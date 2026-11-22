"""Deterministic seeding. The common reference seed for PA4 is 6304."""
from __future__ import annotations

import os
import random

import numpy as np
import torch

DEFAULT_SEED = 6304


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = False) -> int:
    """Seed python, numpy and torch (CPU + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Matmul numerics are part of the matched protocol: never silently use TF32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return seed


def make_generator(seed: int = DEFAULT_SEED, device: str | torch.device = "cpu") -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g
