"""3-symbol Markov toy for the exact-speculative-sampling validation (Step 1).

Target p and draft q are first-order chains with a BOS row (initial distribution). Sequences of length 3 are generated with
``speculative_generate`` (gamma = 2, 100,000 fixed-seed sequences) and compared to the exact target law by Pearson chi-square
over the 27 outcomes (every expected count >= 10). The buggy control replaces the positive residual by p.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch
from scipy import stats

from pa4common.seed import DEFAULT_SEED
from .lm import TableLM
from .specdec import speculative_generate

V, BOS, LEN = 3, 3, 3
P_TARGET = torch.tensor([[0.60, 0.25, 0.15],
                         [0.15, 0.55, 0.30],
                         [0.30, 0.10, 0.60],
                         [0.50, 0.30, 0.20]], dtype=torch.float64)     # row 3 = BOS (initial law)
Q_DRAFT = torch.tensor([[0.25, 0.50, 0.25],
                        [0.45, 0.20, 0.35],
                        [0.20, 0.40, 0.40],
                        [0.20, 0.30, 0.50]], dtype=torch.float64)


def exact_law(trans: torch.Tensor = P_TARGET) -> np.ndarray:
    out = np.zeros(V ** LEN)
    for k, seq in enumerate(itertools.product(range(V), repeat=LEN)):
        pr, last = 1.0, BOS
        for s in seq:
            pr *= float(trans[last, s])
            last = s
        out[k] = pr
    return out


def sample_counts(n: int = 100_000, gamma: int = 2, residual_mode: str = "exact", seed: int = DEFAULT_SEED, p=P_TARGET, q=Q_DRAFT) -> np.ndarray:
    """Counts over the 27 length-3 sequences from ``n`` independent speculative runs (run i uses its own generator seed)."""
    target, draft = TableLM(p, "target"), TableLM(q, "draft")
    counts = np.zeros(V ** LEN, dtype=np.int64)
    pow3 = np.array([9, 3, 1])
    gen = torch.Generator().manual_seed(seed)             # one fixed-seed stream; sequences are consecutive independent runs
    for _ in range(n):
        out, _ = speculative_generate(target, draft, [BOS], gamma, LEN, gen, residual_mode=residual_mode, timed=False)
        counts[int(np.dot(out[:LEN], pow3))] += 1
    return counts


def pearson_test(counts: np.ndarray, law: np.ndarray | None = None) -> dict:
    law = exact_law() if law is None else law
    n = counts.sum()
    exp = law * n
    assert exp.min() >= 10, f"expected counts below 10: {exp.min()}"
    chi2 = float(((counts - exp) ** 2 / exp).sum())
    dof = len(counts) - 1
    return {"chi2": chi2, "dof": dof, "p_value": float(stats.chi2.sf(chi2, dof)), "n": int(n), "min_expected": float(exp.min())}


def run_toy_validation(n: int = 100_000, gamma: int = 2, alpha: float = 0.01) -> dict:
    ok = pearson_test(sample_counts(n, gamma, "exact"))
    bug = pearson_test(sample_counts(n, gamma, "p"))
    # also the draft law itself must be rejected (sanity: the test has power against q)
    return {"correct": ok, "buggy_residual_is_p": bug, "alpha": alpha, "gamma": gamma,
            "passed": bool(ok["p_value"] >= alpha and bug["p_value"] < alpha)}
