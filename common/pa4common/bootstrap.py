"""Paired bootstrap confidence intervals.

* Example-level resampling (HellaSwag): resample example indices with replacement, the same
  indices for both methods, and take the mean of the per-example paired difference.
* Window-level resampling (perplexity): resample whole fixed evaluation windows with
  replacement, *recompute pooled token NLL* (sum NLL / sum token count) for each method on the
  same resample, and only then exponentiate to perplexity.
"""
from __future__ import annotations

import numpy as np

from .seed import DEFAULT_SEED


def _percentile_ci(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _index_batches(n: int, n_boot: int, seed: int, chunk: int = 1000):
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        yield rng.integers(0, n, size=(m, n))
        done += m


def paired_bootstrap_mean(a, b=None, *, n_boot: int = 10000, seed: int = DEFAULT_SEED, alpha: float = 0.05) -> dict:
    """CI of mean(a) - mean(b) with paired resampling (b=None gives CI of mean(a))."""
    a = np.asarray(a, dtype=np.float64)
    d = a if b is None else a - np.asarray(b, dtype=np.float64)
    assert d.ndim == 1 and len(d) > 0
    stats = np.concatenate([d[idx].mean(axis=1) for idx in _index_batches(len(d), n_boot, seed)])
    lo, hi = _percentile_ci(stats, alpha)
    return {
        "estimate": float(d.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "n": int(len(d)),
        "n_boot": n_boot,
        "seed": seed,
        "alpha": alpha,
        "excludes_zero": bool(lo > 0 or hi < 0),
        "resampling": "paired examples",
    }


def pooled_ppl(nll_sum: np.ndarray, count: np.ndarray) -> tuple[float, float]:
    mean_nll = float(nll_sum.sum() / count.sum())
    return mean_nll, float(np.exp(mean_nll))


def paired_bootstrap_ppl(
    nll_a, cnt_a, nll_b, cnt_b, *, n_boot: int = 10000, seed: int = DEFAULT_SEED, alpha: float = 0.05
) -> dict:
    """Paired window-level bootstrap for perplexity. Differences are reported as A - B.

    ``nll_*`` are per-window summed NLL, ``cnt_*`` per-window scored-token counts. Windows must be
    the identical fixed set (same order) for both models.
    """
    nll_a, cnt_a = np.asarray(nll_a, np.float64), np.asarray(cnt_a, np.float64)
    nll_b, cnt_b = np.asarray(nll_b, np.float64), np.asarray(cnt_b, np.float64)
    assert nll_a.shape == nll_b.shape == cnt_a.shape == cnt_b.shape
    assert np.array_equal(cnt_a, cnt_b), "paired evaluation requires identical scored tokens per window"
    n = len(nll_a)
    d_nll, d_ppl, ppl_a_s, ppl_b_s = [], [], [], []
    for idx in _index_batches(n, n_boot, seed):
        ma = nll_a[idx].sum(1) / cnt_a[idx].sum(1)
        mb = nll_b[idx].sum(1) / cnt_b[idx].sum(1)
        d_nll.append(ma - mb)
        d_ppl.append(np.exp(ma) - np.exp(mb))
        ppl_a_s.append(np.exp(ma))
        ppl_b_s.append(np.exp(mb))
    d_nll, d_ppl = np.concatenate(d_nll), np.concatenate(d_ppl)
    ppl_a_s, ppl_b_s = np.concatenate(ppl_a_s), np.concatenate(ppl_b_s)
    nll_pt_a, ppl_pt_a = pooled_ppl(nll_a, cnt_a)
    nll_pt_b, ppl_pt_b = pooled_ppl(nll_b, cnt_b)
    out = {
        "n_windows": n,
        "n_tokens": int(cnt_a.sum()),
        "n_boot": n_boot,
        "seed": seed,
        "alpha": alpha,
        "resampling": "paired whole windows; pooled token NLL recomputed before exp",
        "mean_nll_a": nll_pt_a,
        "mean_nll_b": nll_pt_b,
        "ppl_a": ppl_pt_a,
        "ppl_b": ppl_pt_b,
        "ppl_a_ci": _percentile_ci(ppl_a_s, alpha),
        "ppl_b_ci": _percentile_ci(ppl_b_s, alpha),
    }
    for name, samples, est in (("delta_nll", d_nll, nll_pt_a - nll_pt_b), ("delta_ppl", d_ppl, ppl_pt_a - ppl_pt_b)):
        lo, hi = _percentile_ci(samples, alpha)
        out[name] = {"estimate": est, "ci_low": lo, "ci_high": hi, "excludes_zero": bool(lo > 0 or hi < 0)}
    return out
