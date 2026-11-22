"""Shared constants for Task 1."""
from __future__ import annotations

import numpy as np

SEED = 6304
PROJ_NAMES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ATTN_PROJS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJS = ("gate_proj", "up_proj", "down_proj")
# projections that share one input tensor (hence one calibration Gram matrix)
INPUT_GROUP = {
    "q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv",
    "o_proj": "o",
    "gate_proj": "gu", "up_proj": "gu",
    "down_proj": "down",
}
GROUPS = ("qkv", "o", "gu", "down")

GROUP_SIZE = 128
SPARSITIES = (0.25, 0.50, 0.75)
BITS_UNIFORM = (8, 4, 3)
BITS_ALLOWED = (2, 3, 4, 8)
S_MIN, S_MAX = 0.10, 0.85
# sparsity levels used for the P3 loss tables / allocation search
LEVELS = tuple(float(x) for x in np.round(np.arange(S_MIN, S_MAX + 1e-9, 0.05), 2))
AWQ_GRID = tuple(round(i / 10, 1) for i in range(11))  # alpha in {0.0, 0.1, ..., 1.0}
HELLASWAG_SUBSET_N = 512
OWL_M = 5.0  # OWL outlier threshold: score > M * mean(score)

TASK_VERSIONS = {
    # recorded in every HellaSwag result: lm-evaluation-harness hellaswag task, version 1.0
    "hellaswag_task": "lm-eval hellaswag (version 1.0)",
}


def unit_id(layer: int, name: str) -> str:
    return f"L{layer:02d}.{name}"
