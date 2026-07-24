"""Isolated objective sessions for callers that score more than one tune set."""
from __future__ import annotations

import importlib.util
import os
import threading
import uuid
from pathlib import Path

_IMPORT_LOCK = threading.Lock()
_OBJECTIVE_PATH = Path(__file__).with_name("afpx_objective.py")


class ScorerSession:
    """Own one independently initialized objective module and its immutable caches."""

    def __init__(self, data_root, baseline, target, level_calibration=None):
        self.data_root = Path(data_root).resolve()
        self.baseline = Path(baseline).resolve()
        self.target = Path(target).resolve()
        self.level_calibration = dict(level_calibration or {})
        self._module = self._load_module()
        self._module.LEVEL_CALIBRATION = dict(self.level_calibration)

    def _load_module(self):
        module_name = "afpx_objective_session_" + uuid.uuid4().hex
        spec = importlib.util.spec_from_file_location(module_name, _OBJECTIVE_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load objective module: %s" % _OBJECTIVE_PATH)
        module = importlib.util.module_from_spec(spec)
        updates = {
            "AFPX_DATA_ROOT": str(self.data_root),
            "AFPX_BASELINE": str(self.baseline),
            "AFPX_TARGET": str(self.target),
        }
        with _IMPORT_LOCK:
            previous = {key: os.environ.get(key) for key in updates}
            try:
                os.environ.update(updates)
                spec.loader.exec_module(module)
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        return module

    def score_bands(self, band_sets):
        return self._module.score_bands(band_sets)

    def score_afpx(self, path):
        return self._module.score_afpx(path)

    def baseline_band_sets(self):
        return self._module.baseline_band_sets()

    def prediction_audit(self):
        self._module._init()
        return dict(self._module._PREDICTION_AUDIT)