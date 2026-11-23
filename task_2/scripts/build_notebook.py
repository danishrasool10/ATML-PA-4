"""Regenerates notebooks/pa4_task2_kaggle.ipynb:  python task_2/scripts/build_notebook.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))
from pa4common.notebook import build  # noqa: E402

build(str(ROOT / "task_2" / "notebooks" / "pa4_task2_kaggle.ipynb"), 2, "pa4t2",
      'Efficient attention and context memory (Parts A, B, C)',
      '`python -m pa4t2.run_all` schedules two GPU lanes: Part C (KV-cache study, 180 prompts), MQAR B1 / B2 / B3 (3 seeds each) run concurrently; afterwards, alone on an idle machine, the timing-critical jobs: Part A attention benchmark, MQAR inference sweep, Part C packed-cache benchmark (batch 1 and 8, 2048/4096), then the MQAR analysis. Outputs: `task_2/results/{partA,partB,partC}`.',
      'two lanes (one job per GPU) for training/evaluation, then one quiet GPU for benchmarks')
