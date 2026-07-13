from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _run_bundled_script() -> bool:
    if len(sys.argv) < 2 or not sys.argv[1].lower().endswith(".py"):
        return False
    requested = Path(sys.argv[1])
    if not requested.is_absolute():
        root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
        requested = root / requested
    if not requested.exists():
        return False
    sys.argv = [str(requested), *sys.argv[2:]]
    runpy.run_path(str(requested), run_name="__main__")
    return True


if __name__ == "__main__":
    if not _run_bundled_script():
        from optimizer_gui.window import run_gui
        raise SystemExit(run_gui())
