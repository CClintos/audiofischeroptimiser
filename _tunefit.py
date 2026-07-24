"""Compatibility import for the canonical DSP helpers.

The implementation lives in objective_module._tunefit so the optimizer and its
standalone objective cannot drift apart again.
"""
from objective_module._tunefit import *  # noqa: F401,F403
