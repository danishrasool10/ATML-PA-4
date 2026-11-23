"""Puts ``common/`` and this task's ``src/`` on sys.path (the same way the Kaggle notebook does)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "common", ROOT / "task_2" / "src"):
    sys.path.insert(0, str(p))
