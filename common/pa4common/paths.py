"""Where the course data repo (``atml-pa4-datasets``) lives.

Order: ``$PA4_DATA_ROOT``; else the first existing of ``../atml-pa4-datasets``, ``../../atml-pa4-datasets`` and
``./atml-pa4-datasets`` (relative to the current directory, which is a ``task_N`` folder or the repo root); else the first candidate.
"""
from __future__ import annotations

import os
from pathlib import Path

_CANDIDATES = ("../atml-pa4-datasets", "../../atml-pa4-datasets", "atml-pa4-datasets")


def data_root() -> str:
    env = os.environ.get("PA4_DATA_ROOT")
    if env:
        return env
    for c in _CANDIDATES:
        if (Path(c) / "scripts" / "setup_data.py").exists():
            return c
    return _CANDIDATES[0]
