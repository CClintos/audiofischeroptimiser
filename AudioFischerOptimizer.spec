from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


root = Path(SPECPATH)
datas = []
for name in (
    "run_optimizer.ps1",
    "run_guided_stream_workers.ps1",
    "merge_guided_stream_results.ps1",
    "ResoNix Target Curve 2026.txt",
    "_optimizer_stream.py",
    "_merge_stream_results.py",
):
    datas.append((str(root / name), "."))
for folder in ("scripts", "objective_module"):
    for path in (root / folder).glob("*.py"):
        datas.append((str(path), folder))

hiddenimports = (
    collect_submodules("optuna")
    + collect_submodules("cma")
    + [
        "_optimizer", "_optimizer_stream", "_merge_stream_results", "_tunefit",
        "_make_v3", "_devcalc", "afpx", "pct6", "scipy.optimize", "scipy.signal",
    ]
)

a = Analysis(
    [str(root / "audiofischer_gui.py")],
    pathex=[str(root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "IPython", "notebook"],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AudioFischerOptimizer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)
worker_exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AudioFischerOptimizerWorker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
coll = COLLECT(
    exe,
    worker_exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="AudioFischerOptimizer",
)
