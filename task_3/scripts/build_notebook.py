"""Regenerates notebooks/pa4_task3_kaggle.ipynb:  python task_3/scripts/build_notebook.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))
from pa4common.notebook import build  # noqa: E402

build(str(ROOT / "task_3" / "notebooks" / "pa4_task3_kaggle.ipynb"), 3, "pa4t3",
      'Policy distillation and speculative decoding',
      '`python -m pa4t3.run_all` runs unit tests, `setup_data.py --public` and `--prefix-banks`, the Step 1 validation gate (chi-square on 100,000 Markov-toy sequences, KV rollback, merged-vs-unmerged logits; sweeps refuse to start if it fails), the four distillation runs (off-/on-policy x seeds 6304, 6305) on the two GPUs, then prefix-bank alignment and standalone quality; afterwards, alone on an idle machine: pretrained-draft and merged-student speculative benchmarks (gamma 2 and 4), fixed-64-token throughput and the analysis with deployment break-even. Outputs: `task_3/results/`.',
      'two lanes for validation / distillation / banks / standalone, then one quiet GPU for the speculative-decoding benchmarks')
