from __future__ import annotations

import bisect
import difflib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.make_measurement_manifest import (
    ALL_MEASUREMENT_ROLES, build_manifest, compact_manifest, detect_layout,
    first_existing, load_role_map, mapped_measurement, measurement_spec,
    optional_pair_roles,
)


APP_NAME = "AudioFischer Optimizer"
JOB_FILE = "gui_job.json"
RUN_CLAIM_FILE = ".active_run.json"
RUN_PHASES = ("searching", "merging", "verifying", "reporting", "complete")
RUN_SUCCESS_FILE = ".runner_success"
RUN_FAILURE_FILE = ".runner_failed"


def runtime_root() -> Path:
    bundled = getattr(sys, "_MEIPASS", None)
    return Path(bundled) if bundled else Path(__file__).resolve().parents[1]


def default_target() -> Path:
    return runtime_root() / "ResoNix Target Curve 2026.txt"


def load_target_curve(path: Path, max_points: int = 260) -> dict[str, Any]:
    """Load a two-column target and normalize its shape to 0 dB at 1 kHz."""
    path = Path(path)
    if not path.is_file():
        return {}
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    row_pattern = re.compile(rf"^\s*({number})\s*[,;\t ]+\s*({number})(?:\s|$)")
    points: dict[float, float] = {}
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        match = row_pattern.match(line)
        if not match:
            continue
        frequency, level = float(match.group(1)), float(match.group(2))
        if 10.0 <= frequency <= 50000.0 and math.isfinite(level):
            points[frequency] = level
    if len(points) < 2:
        return {}
    frequencies = sorted(points)
    levels = [points[frequency] for frequency in frequencies]
    logs = [math.log10(frequency) for frequency in frequencies]
    anchor_log = math.log10(1000.0)
    index = bisect.bisect_left(logs, anchor_log)
    if index <= 0:
        anchor = levels[0]
    elif index >= len(logs):
        anchor = levels[-1]
    else:
        span = logs[index] - logs[index - 1]
        ratio = (anchor_log - logs[index - 1]) / span if span else 0.0
        anchor = levels[index - 1] + ratio * (levels[index] - levels[index - 1])
    eligible = [i for i, frequency in enumerate(frequencies) if 20.0 <= frequency <= 20000.0]
    if len(eligible) > max_points:
        step = (len(eligible) - 1) / float(max_points - 1)
        eligible = sorted({eligible[round(i * step)] for i in range(max_points)})
    return {
        "file": path.name,
        "frequency_hz": [round(frequencies[i], 3) for i in eligible],
        "relative_db": [round(levels[i] - anchor, 3) for i in eligible],
        "anchor_hz": 1000.0,
    }

def timestamped_run_root(base: Path | None = None) -> Path:
    """Atomically reserve a unique run folder, including across GUI instances."""
    parent = base or (Path.home() / "Documents" / "AudioFischer Optimizer Runs")
    parent.mkdir(parents=True, exist_ok=True)
    stem = "Optimizer_Run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    for index in range(1000):
        name = stem if index == 0 else f"{stem}_{index:02d}"
        candidate = parent / name
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not reserve a unique run folder under {parent}")


class RunRootBusyError(RuntimeError):
    pass


def process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil
        return bool(psutil.pid_exists(pid) and psutil.Process(pid).is_running())
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start_detached_process(
    program: str, arguments: list[str], working_directory: Path, log_path: Path,
) -> int:
    """Start a console-free runner that is not owned by the GUI process."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("a", encoding="utf-8")
    kwargs: dict[str, Any] = {
        "cwd": str(working_directory),
        "stdin": subprocess.DEVNULL,
        "stdout": stream,
        "stderr": subprocess.STDOUT,
        "close_fds": True,
    }
    if os.name == "nt":
        # A Windows child process does not need DETACHED_PROCESS to survive the
        # GUI closing. That flag also makes powershell.exe exit silently when
        # its standard streams are redirected to the run log. A new process
        # group plus CREATE_NO_WINDOW keeps the runner independent and hidden
        # without losing the launch error/output.
        kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        )
    else:
        kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen([program, *arguments], **kwargs)
    finally:
        stream.close()
    return int(process.pid)


def claim_run_root(run_root: Path, owner_pid: int | None = None) -> Path:
    """Create an exclusive active-run claim or reject a live existing owner."""
    root = Path(run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    claim = root / RUN_CLAIM_FILE
    pid = int(owner_pid or os.getpid())
    payload = {
        "pid": pid,
        "claimed_at": datetime.now().isoformat(timespec="seconds"),
        "run_root": str(root),
    }
    for _attempt in range(2):
        try:
            descriptor = os.open(str(claim), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2)
            return claim
        except FileExistsError:
            try:
                existing = json.loads(claim.read_text(encoding="utf-8-sig"))
                existing_pid = int(existing.get("pid", 0))
            except (OSError, ValueError, TypeError):
                existing_pid = 0
            if process_is_running(existing_pid):
                raise RunRootBusyError(
                    f"This run folder is already active in process {existing_pid}: {root}"
                )
            claim.unlink(missing_ok=True)
    raise RunRootBusyError(f"Could not claim run folder: {root}")


def update_run_claim(run_root: Path, pid: int) -> None:
    claim = Path(run_root).resolve() / RUN_CLAIM_FILE
    payload = {
        "pid": int(pid),
        "claimed_at": datetime.now().isoformat(timespec="seconds"),
        "run_root": str(Path(run_root).resolve()),
    }
    temporary = claim.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(claim)


def release_run_claim(run_root: Path, pid: int | None = None) -> None:
    claim = Path(run_root).resolve() / RUN_CLAIM_FILE
    if pid is not None and claim.exists():
        try:
            owner = int(json.loads(claim.read_text(encoding="utf-8-sig")).get("pid", 0))
        except (OSError, ValueError, TypeError):
            return
        if owner != int(pid):
            return
    claim.unlink(missing_ok=True)


def active_run_pid(run_root: Path) -> int:
    claim = Path(run_root).resolve() / RUN_CLAIM_FILE
    if not claim.exists():
        return 0
    try:
        pid = int(json.loads(claim.read_text(encoding="utf-8-sig")).get("pid", 0))
    except (OSError, ValueError, TypeError):
        return 0
    return pid if process_is_running(pid) else 0


def discover_baseline(folder: Path) -> Path | None:
    preferred = folder / "baseline.afpx"
    if preferred.exists():
        return preferred
    files = sorted(folder.glob("*.afpx"))
    return files[0] if len(files) == 1 else None


@dataclass
class RunConfig:
    data_root: str
    baseline: str
    target: str
    run_root: str
    seconds: int = 1200
    cpu_percent: int = 60
    ram_percent: int = 50
    mode: str = "peq"
    workflow: str = ""
    proposal: str = "beam"
    phase_writes: str = "auto"
    voicing_variants: str = "off"
    sub_blend: str = "off"
    headroom_db: float | None = None
    level_calibration: str = ""
    role_map: str = ""
    status: str = "ready"
    summary_path: str = ""
    error: str = ""
    started_at: str = ""
    completed_at: str = ""

    @property
    def ui_workflow(self) -> str:
        return self.workflow or self.mode

    @property
    def workers(self) -> int:
        if self.mode == "phase":
            return 1
        logical = os.cpu_count() or 4
        return max(1, min(12, round(logical * self.cpu_percent / 100.0)))

    def save(self) -> Path:
        root = Path(self.run_root)
        root.mkdir(parents=True, exist_ok=True)
        path = root / JOB_FILE
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, run_root: Path) -> "RunConfig":
        payload = json.loads((run_root / JOB_FILE).read_text(encoding="utf-8"))
        return cls(**payload)


def _cancel_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True, creationflags=flags,
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()


def validate_config(config: RunConfig, cancel_event: Any = None) -> dict[str, Any]:
    data_root = Path(config.data_root)
    baseline = Path(config.baseline)
    target = Path(config.target)
    errors: list[str] = []
    if not data_root.is_dir():
        errors.append("Measurement folder does not exist")
    if not baseline.is_file() or baseline.suffix.lower() != ".afpx":
        errors.append("A valid baseline AFPX file is required")
    if not target.is_file():
        errors.append("A target curve text file is required")
    diagnostics: dict[str, Any] = {
        "job_config": asdict(config),
        "manifest": None,
        "preflight_command": [],
        "stdout": "",
        "stderr": "",
    }
    if errors:
        diagnostics["errors"] = list(errors)
        return {
            "valid": False, "errors": errors, "manifest": None, "compact": None,
            "diagnostics": diagnostics,
        }
    if cancel_event is not None and cancel_event.is_set():
        return {
            "valid": False, "cancelled": True, "errors": [], "manifest": None,
            "compact": None, "diagnostics": diagnostics,
        }
    manifest = build_manifest(
        data_root.resolve(), baseline.resolve(), target.resolve(), config.role_map or None,
    )
    diagnostics["manifest"] = manifest
    blocking = []
    if manifest["measurements_missing"]:
        blocking.append("Required measurements are missing")
    if not manifest["baseline_exists"]:
        blocking.append("Baseline AFPX is missing")
    if not manifest["target_exists"]:
        blocking.append("Target curve is missing")
    preflight = None
    if not blocking:
        script = runtime_root() / "scripts" / "gui_preflight.py"
        command = [
            worker_executable(), str(script), "--data-root", str(data_root),
            "--baseline", str(baseline), "--target", str(target),
        ]
        if config.level_calibration:
            command.extend(["--level-calibration", config.level_calibration])
        if config.role_map:
            command.extend(["--role-map", config.role_map])
        diagnostics["preflight_command"] = command
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, creationflags=flags,
        )
        while process.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                _cancel_process(process)
                return {
                    "valid": False, "cancelled": True, "errors": [],
                    "manifest": manifest, "compact": compact_manifest(manifest),
                    "preflight": None, "diagnostics": diagnostics,
                }
            time.sleep(0.05)
        stdout, stderr = process.communicate()
        diagnostics["stdout"] = stdout
        diagnostics["stderr"] = stderr
        diagnostics["return_code"] = process.returncode
        try:
            preflight = json.loads(stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            preflight = {
                "valid": False,
                "errors": [
                    "Optimizer preflight did not return a readable result. "
                    "Use Copy Diagnostics to capture stderr, the job configuration, "
                    "and the measurement manifest."
                ],
                "parse_failed": True,
            }
        if not preflight.get("valid"):
            blocking.extend(str(item) for item in preflight.get("errors", []))
        if config.mode == "phase" and not dict(preflight.get("measurement_session", {})).get("phase_valid"):
            blocking.append("Phase stage requires phase-valid sweeps with one shared timing reference")
    return {
        "valid": not blocking,
        "errors": blocking,
        "manifest": manifest,
        "compact": compact_manifest(manifest),
        "preflight": preflight,
        "diagnostics": diagnostics,
    }


def powershell_command(config: RunConfig, executable: str | None = None) -> tuple[str, list[str]]:
    script = runtime_root() / "run_optimizer.ps1"
    python_exe = executable or worker_executable()
    args = [
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
        "-DataRoot", config.data_root,
        "-Baseline", config.baseline,
        "-Target", config.target,
        "-Root", config.run_root,
        "-Seconds", str(config.seconds),
        "-Workers", str(config.workers),
        "-Mode", config.mode,
        "-Proposal", config.proposal,
        "-PhaseWrites", config.phase_writes,
        "-VoicingVariants", config.voicing_variants,
        "-SubBlend", config.sub_blend,
        "-PythonExe", python_exe,
    ]
    if config.headroom_db is not None:
        args.extend(["-HeadroomDb", str(config.headroom_db)])
    if config.level_calibration:
        args.extend(["-LevelCalibration", config.level_calibration])
    if config.role_map:
        args.extend(["-RoleMap", config.role_map])
    return "powershell.exe", args


def worker_executable() -> str:
    if getattr(sys, "frozen", False):
        companion = Path(sys.executable).with_name("AudioFischerOptimizerWorker.exe")
        if companion.exists():
            return str(companion)
    return sys.executable


def collect_progress(run_root: Path) -> dict[str, Any]:
    states = []
    for path in sorted(run_root.glob("worker_*/stream_state.json")):
        try:
            states.append(json.loads(path.read_text(encoding="utf-8-sig")))
        except (OSError, ValueError):
            continue
    objectives = [
        float(row["objective"])
        for state in states
        for row in state.get("best", [])[:1]
        if "objective" in row
    ]
    verified = len(list((run_root / "_merged_top" / "verification").glob("*.json")))
    candidates = len([
        path for path in (run_root / "_merged_top").glob("*.afpx")
        if path.name.startswith(("family_", "voicing_"))
    ])
    phase = next(
        (name for name in reversed(RUN_PHASES) if (run_root / f".phase_{name}").exists()),
        "searching",
    )
    return {
        "workers_reporting": len(states),
        "trials": sum(int(state.get("completed_trials", 0)) for state in states),
        "best_objective": min(objectives) if objectives else None,
        "elapsed_worker_seconds": max(
            (float(state.get("elapsed_seconds", 0.0)) for state in states), default=0.0
        ),
        "phase": phase,
        "verified_candidates": verified,
        "verification_candidates": candidates,
    }


def locate_summary(run_root: Path) -> Path | None:
    preferred = run_root / "_merged_top" / "assistant_summary.json"
    return preferred if preferred.is_file() else None


def runner_completed_successfully(run_root: Path) -> bool:
    return (run_root / RUN_SUCCESS_FILE).is_file() and locate_summary(run_root) is not None


def runner_failure_reason(run_root: Path) -> str:
    """Return a compact durable runner failure for the GUI and recent-runs list."""
    failure_marker = run_root / RUN_FAILURE_FILE
    log_path = run_root / "gui_runner.log"
    source = failure_marker if failure_marker.is_file() else log_path
    try:
        text = source.read_text(encoding="utf-8-sig", errors="replace").strip()
    except OSError:
        text = ""
    if not text:
        return "Optimizer stopped before producing a merged and verified result."
    compact = " ".join(text.split())
    if "PermissionError" in compact and "stream_state.json" in compact:
        return (
            "A worker could not save its checkpoint because Windows kept the state file "
            "locked. The run remains resumable from its last intact checkpoint."
        )
    return compact[-2000:]


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def candidate_files(summary: dict[str, Any], summary_path: Path) -> list[dict[str, Any]]:
    folder = summary_path.parent
    rows: list[dict[str, Any]] = []
    baseline = summary.get("baseline") or {}
    baseline_name = str(
        ((summary.get("inputs") or {}).get("baseline") or {}).get("file")
        or "Loaded baseline tune"
    )
    baseline_path: Path | None = None
    run_root = folder.parent if folder.name == "_merged_top" else folder
    job_path = run_root / JOB_FILE
    if job_path.exists():
        try:
            configured = Path(str(json.loads(
                job_path.read_text(encoding="utf-8-sig")
            ).get("baseline", "")))
            if configured.is_file():
                baseline_path = configured
        except (OSError, ValueError, TypeError):
            baseline_path = None
    rows.append({
        "role": "Current tune (baseline)",
        "file": baseline_name,
        "objective": baseline.get("objective"),
        "path": str(baseline_path) if baseline_path else "",
        "is_baseline": True,
        "exportable": False,
    })
    best = summary.get("best") or {}
    if best.get("file"):
        rows.append({
            "role": "Recommended candidate", "file": best["file"],
            "objective": best.get("objective"), "is_baseline": False,
        })
    for role, data in (summary.get("families") or {}).items():
        rows.append({"role": role.title(), "file": data.get("file", ""), "objective": data.get("objective")})
    for data in summary.get("voicing_variants") or []:
        rows.append({"role": "Voicing: " + str(data.get("label", "")).title(),
                     "file": data.get("file", ""), "objective": None})
    unique = []
    seen = set()
    for row in rows:
        if row.get("is_baseline"):
            continue
        path = folder / str(row["file"])
        if row["file"] and path.exists() and path not in seen:
            seen.add(path)
            row["path"] = str(path)
            row["exportable"] = True
            unique.append(row)
    return [rows[0], *unique]


def default_export_name(source: Path, role: str, started_at: str = "") -> str:
    stamp = ""
    if started_at:
        try:
            stamp = datetime.fromisoformat(started_at).strftime("%Y%m%d_%H%M%S")
        except ValueError:
            stamp = re.sub(r"\D", "", started_at)[:14]
    if not stamp:
        for parent in source.parents:
            match = re.search(r"Optimizer_Run_(\d{8}_\d{6})", parent.name)
            if match:
                stamp = match.group(1)
                break
    if not stamp:
        stamp = datetime.fromtimestamp(source.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
    role_slug = re.sub(r"[^a-z0-9]+", "_", role.lower()).strip("_") or source.stem
    return f"{stamp}_{role_slug}{source.suffix.lower() or '.afpx'}"


def suggest_measurement_role(
    filename: str, remembered: dict[str, str] | None = None,
) -> str:
    remembered = remembered or {}
    exact = remembered.get(filename.lower())
    if exact in ALL_MEASUREMENT_ROLES:
        return exact

    def normalize(value: str) -> str:
        stem = Path(value).stem.lower()
        stem = re.sub(r"\b(sweep|measurement|measure|rew|export|trace|response)\b", " ", stem)
        stem = stem.replace("front left", "fl").replace("front right", "fr")
        stem = stem.replace("left", "l").replace("right", "r")
        return re.sub(r"[^a-z0-9]+", " ", stem).strip()

    source = normalize(filename)
    candidates: list[tuple[float, str]] = []
    full_spec = measurement_spec("front_3way_plus_sub")
    for role in ALL_MEASUREMENT_ROLES:
        names = (role, *full_spec.get(role, ()))
        score = max(
            difflib.SequenceMatcher(None, source, normalize(name)).ratio()
            for name in names
        )
        candidates.append((score, role))
    score, role = max(candidates, default=(0.0, ""))
    return role if score >= 0.56 else ""


def measurement_checklist(
    root: Path, role_map: Path | str | None = None,
) -> dict[str, Any]:
    root = Path(root)
    mapped = load_role_map(role_map)
    layout = detect_layout(root, mapped) if root.is_dir() else "front_2way_plus_sub"
    optional_roles = optional_pair_roles(layout)
    rows = []
    for role, aliases in measurement_spec(layout).items():
        path = mapped_measurement(root, role, mapped) or first_existing(root, aliases)
        rows.append({
            "role": role,
            "expected": aliases[0],
            "path": str(path) if path else "",
            "ready": bool(path and path.stat().st_size > 0),
            "empty": bool(path and path.stat().st_size == 0),
            "required": role not in optional_roles,
        })
    return {"layout": layout, "rows": rows}


def create_measurement_template(destination: Path, layout: str) -> list[Path]:
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Folder is not empty: {destination}")
    spec = measurement_spec(layout)
    readme = destination / "README - Export from REW.txt"
    targets = [destination / aliases[0] for aliases in spec.values()] + [readme]
    conflicts = [path for path in targets if path.exists()]
    if conflicts:
        raise FileExistsError(", ".join(path.name for path in conflicts))
    destination.mkdir(parents=True, exist_ok=True)
    for path in targets[:-1]:
        path.touch(exist_ok=False)
    readme.write_text(
        "Replace the empty TXT placeholders with the matching REW text exports.\n"
        "Required for PEQ: System Sum, Sub, and each individual front driver.\n"
        "Optional but recommended: the left+right Together traces. They unlock measured "
        "pair-summation, null, and phase validation; PEQ can run without them.\n"
        "Keep the microphone position, source volume, sweep level, and timing reference "
        "consistent for the entire session.\n"
        "Do not rename files after validation unless you validate again.\n",
        encoding="utf-8",
    )
    return targets


def save_role_map(
    path: Path, mapping: dict[str, str], layout: str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "audiofischer-role-map-v1",
        "layout": layout,
        "roles": {
            role: filename for role, filename in mapping.items()
            if role in ALL_MEASUREMENT_ROLES and filename
        },
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def export_candidate(
    source: Path, destination: Path, *, filename: str | None = None, overwrite: bool = False
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / (filename or source.name)
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    shutil.copy2(source, target)
    return target


def memory_guard_status() -> tuple[bool, str]:
    try:
        import psutil
        psutil.Process(os.getpid()).memory_info()
        if int(psutil.virtual_memory().total) <= 0:
            raise RuntimeError("physical RAM total is unavailable")
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def process_tree_memory(pid: int) -> tuple[int, int, str]:
    """Return optimizer process-tree RSS, physical RAM bytes, and any failure."""
    try:
        import psutil
        process = psutil.Process(pid)
        members = [process, *process.children(recursive=True)]
        rss = sum(member.memory_info().rss for member in members if member.is_running())
        return rss, int(psutil.virtual_memory().total), ""
    except Exception as exc:
        return 0, 0, f"{type(exc).__name__}: {exc}"


def stop_process_tree(pid: int, timeout_seconds: float = 20.0) -> dict[str, Any]:
    """Wait for a cooperative stop, then terminate the process tree off the UI thread."""
    deadline = time.monotonic() + timeout_seconds
    try:
        import psutil
        while time.monotonic() < deadline:
            if not psutil.pid_exists(pid):
                return {"forced": False, "error": ""}
            time.sleep(0.1)
    except Exception:
        while time.monotonic() < deadline:
            if os.name != "nt":
                try:
                    os.kill(pid, 0)
                except OSError:
                    return {"forced": False, "error": ""}
            time.sleep(0.1)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    completed = subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        capture_output=True, text=True, creationflags=flags,
    )
    error = "" if completed.returncode == 0 else (completed.stderr.strip() or completed.stdout.strip())
    return {"forced": True, "error": error}
