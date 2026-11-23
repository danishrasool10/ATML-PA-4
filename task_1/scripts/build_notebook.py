"""Regenerates notebooks/pa4_task1_kaggle.ipynb:  python task_1/scripts/build_notebook.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))
from pa4common.notebook import build  # noqa: E402

build(str(ROOT / "task_1" / "notebooks" / "pa4_task1_kaggle.ipynb"), 1, "pa4t1",
      'Pruning and quantization of Qwen2.5-1.5B',
      '`python -m pa4t1.run_all` runs, one job per stage and in this order: unit tests and `setup_data.py --public` (C4 / WikiText-2 / HellaSwag locked files) -> baseline -> P1 magnitude -> P2 Wanda -> 2:4 diagnostic -> sensitivity tables -> P3 selection (development manifest only, locked before any test run) -> P3 -> Q1 -> Q2 (AWQ-style) -> Q3 tables / selection / Q3 -> analysis (tables, plots, paired bootstrap). Outputs: `task_1/results/`.',
      'the stages share cached statistics and the selection lock, so they run as one sequential chain on GPU 0; GPU 1 stays idle (nothing in Task 1 benefits from a second T4)')
