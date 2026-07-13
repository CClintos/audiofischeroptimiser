from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.make_measurement_manifest import build_manifest, compact_manifest


APP_NAME = "AudioFischer Optimizer"
JOB_FILE = "gui_job.json"


def runtime_root() -> Path:
    bundled = getattr(sys, "_MEIPASS", None)
    return Path(bundled) if bundled else Path(__file__).resolve().parents[1]


def default_target() -> Path:
    return runtime_root() / "ResoNix Target Curve 2026.txt"


def timestamped_run_root(base: Path | None = None) -> Path:
    parent = base or (Path.home() / "Documents" / "AudioFischer Optimizer Runs")
    return parent / ("Optimizer_Run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))


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
    proposal: str = "beam"
    phase_writes: str = "auto"
    voicing_variants: str = "off"
    sub_blend: str = "off"
    headroom_db: float | None = None
    level_calibration: str = ""
    status: str = "ready"
    summary_path: str = ""
    error: str = ""
    started_at: str = ""
    completed_at: str = ""

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


def validate_config(config: RunConfig) -> dict[str, Any]:
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
    if errors:
        return {"valid": False, "errors": errors, "manifest": None, "compact": None}
    manifest = build_manifest(data_root.resolve(), baseline.resolve(), target.resolve())
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
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        completed = subprocess.run(command, capture_output=True, text=True, creationflags=flags)
        try:
            preflight = json.loads(completed.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            preflight = {"valid": False, "errors": [completed.stderr.strip() or "Optimizer preflight did not return a result"]}
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
    return {
        "workers_reporting": len(states),
        "trials": sum(int(state.get("completed_trials", 0)) for state in states),
        "best_objective": min(objectives) if objectives else None,
        "elapsed_worker_seconds": max(
            (float(state.get("elapsed_seconds", 0.0)) for state in states), default=0.0
        ),
    }


def locate_summary(run_root: Path) -> Path | None:
    preferred = run_root / "_merged_top" / "assistant_summary.json"
    if preferred.exists():
        return preferred
    found = sorted(run_root.rglob("assistant_summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return found[0] if found else None


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def candidate_files(summary: dict[str, Any], summary_path: Path) -> list[dict[str, Any]]:
    folder = summary_path.parent
    rows: list[dict[str, Any]] = []
    best = summary.get("best") or {}
    if best.get("file"):
        rows.append({"role": "Best objective", "file": best["file"], "objective": best.get("objective")})
    for role, data in (summary.get("families") or {}).items():
        rows.append({"role": role.title(), "file": data.get("file", ""), "objective": data.get("objective")})
    for data in summary.get("voicing_variants") or []:
        rows.append({"role": "Voicing: " + str(data.get("label", "")).title(),
                     "file": data.get("file", ""), "objective": None})
    unique = []
    seen = set()
    for row in rows:
        path = folder / str(row["file"])
        if row["file"] and path.exists() and path not in seen:
            seen.add(path)
            row["path"] = str(path)
            unique.append(row)
    return unique


def export_candidate(source: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / source.name
    shutil.copy2(source, target)
    return target


def process_tree_memory(pid: int) -> tuple[int, int]:
    """Return optimizer process-tree RSS and physical RAM bytes."""
    try:
        import psutil
        process = psutil.Process(pid)
        members = [process, *process.children(recursive=True)]
        rss = sum(member.memory_info().rss for member in members if member.is_running())
        return rss, int(psutil.virtual_memory().total)
    except Exception:
        return 0, 0
