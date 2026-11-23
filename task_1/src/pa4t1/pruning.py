"""Unstructured / 2:4 pruning on the eligible projections (P1, P2, P3 application, 2:4 diagnostic).

Conventions
-----------
* per-output-row ranking: every row of every eligible matrix independently zeroes its ``k``
  lowest-scoring weights (``k = round(s * d_in)`` for a uniform sparsity ``s``),
* magnitude score  S_ij = |W_ij|,   Wanda score  S_ij = |W_ij| * ||X_{:,j}||_2 (dense calibration
  activations of the 128 calibration sequences), ties broken by column index (stable sort),
* only eligible weights are touched (embeddings, norms, biases, LM head are never pruned and are
  never counted towards the sparsity target),
* 2:4: in every group of four contiguous input channels of a row, the two lowest-scoring weights
  are zeroed (the two highest-ranked are retained).
"""
from __future__ import annotations

import time

import torch

from .calibration import quad_error, run_layerwise
from .constants import PROJ_NAMES, S_MAX, S_MIN, unit_id
from .model_utils import get_proj


def k_from_sparsity(s: float, d_in: int) -> int:
    return int(round(s * d_in))


def score_magnitude(W: torch.Tensor) -> torch.Tensor:
    return W.abs().float()


def score_wanda(W: torch.Tensor, act_norm: torch.Tensor) -> torch.Tensor:
    return W.abs().float() * act_norm.float()[None, :]


def rowwise_prune_mask(score: torch.Tensor, k: int) -> torch.Tensor:
    """Boolean mask, True = prune: the k lowest scores of every row (stable, index tie-break)."""
    mask = torch.zeros_like(score, dtype=torch.bool)
    if k > 0:
        idx = torch.argsort(score, dim=1, stable=True)[:, :k]
        mask.scatter_(1, idx, True)
    return mask


def nm_prune_mask(score: torch.Tensor, n_keep: int = 2, m: int = 4) -> torch.Tensor:
    """N:M mask, True = prune: in each group of ``m`` contiguous input channels keep the ``n_keep`` best."""
    dout, din = score.shape
    assert din % m == 0, f"d_in={din} not divisible by {m}"
    g = score.reshape(dout, din // m, m)
    idx = torch.argsort(g, dim=2, stable=True)[:, :, : m - n_keep]
    mask = torch.zeros_like(g, dtype=torch.bool)
    mask.scatter_(2, idx, True)
    return mask.reshape(dout, din)


def rank_matrix(score: torch.Tensor) -> torch.Tensor:
    """rank[i, j] = position of column j in row i's ascending order (0 = lowest score)."""
    order = torch.argsort(score, dim=1, stable=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(score.shape[1], device=score.device).expand_as(order).contiguous())
    return rank


def score_summary(score: torch.Tensor, ks: dict[float, int], seed: int = 6304) -> dict:
    """Distribution summary of a score matrix (kept for the P2 per-projection score distributions)."""
    flat = score.reshape(-1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    if flat.numel() > 2_000_000:
        idx = torch.randint(0, flat.numel(), (2_000_000,), generator=g).to(flat.device)
        sample = flat[idx]
    else:
        sample = flat
    qs = torch.tensor([0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999], device=flat.device)
    quant = torch.quantile(sample.float(), qs).tolist()
    logs = torch.log10(sample.float().clamp(min=1e-12))
    hist = torch.histc(logs, bins=60, min=-9.0, max=3.0)
    mean = float(flat.mean())
    thr = {}
    srt = torch.sort(score, dim=1).values
    for s, k in ks.items():
        thr[str(s)] = float(srt[:, k - 1].mean()) if k > 0 else 0.0  # mean per-row cut-off score
    return {"quantiles": dict(zip(["q0.001", "q0.01", "q0.05", "q0.1", "q0.25", "q0.5", "q0.75", "q0.9", "q0.95", "q0.99", "q0.999"], quant)),
            "mean": mean, "max": float(flat.max()), "log10_hist": {"min": -9.0, "max": 3.0, "bins": 60, "counts": hist.tolist()},
            "mean_row_cutoff_score": thr}


def prune_model(model, units, batches, plan: dict[str, dict], *, diagnostics: bool = False,
                sparsity_for_diag=(0.25, 0.5, 0.75)) -> dict:
    """Apply a pruning ``plan`` layer-wise and measure calibration layer-output errors.

    ``plan[uid]`` = {"rule": "magnitude" | "wanda", "k": int}  (row-wise)
                 or {"rule": ..., "nm": (2, 4)}                 (N:M)
    Returns per-unit records; weights are modified in place.
    """
    records: dict[str, dict] = {}
    t0 = time.perf_counter()

    def layer_fn(li, layer, stats, ctx):
        for name in PROJ_NAMES:
            uid = unit_id(li, name)
            spec = plan[uid]
            lin = get_proj(layer, name)
            W = lin.weight.data
            Wf = W.float()
            G = stats.gram(name)
            norm = stats.act_norm(name)
            score = score_magnitude(W) if spec["rule"] == "magnitude" else score_wanda(W, norm)
            if "nm" in spec:
                mask = nm_prune_mask(score, *spec["nm"])
            else:
                mask = rowwise_prune_mask(score, spec["k"])
            dW = Wf * mask  # the removed part
            ref = quad_error(Wf, G)
            err = quad_error(dW, G)
            rec = {"rule": spec["rule"], "k": spec.get("k"), "nm": spec.get("nm"), "numel": W.numel(), "shape": list(W.shape),
                   "n_pruned_by_mask": int(mask.sum()), "sparsity_by_mask": float(mask.float().mean()),
                   "abs_err": err, "ref_energy": ref, "rel_err": err / max(ref, 1e-30),
                   "nonzero_before": int(torch.count_nonzero(W))}
            if diagnostics:
                ks = {s: k_from_sparsity(s, W.shape[1]) for s in sparsity_for_diag}
                rec["score_summary"] = score_summary(score, ks)
                ov = {}
                for s, k in ks.items():
                    mm = rowwise_prune_mask(score_magnitude(W), k)
                    mw = rowwise_prune_mask(score_wanda(W, norm), k)
                    ov[str(s)] = {"overlap_of_pruned": float((mm & mw).sum() / max(int(mm.sum()), 1)),
                                  "err_magnitude": quad_error(Wf * mm, G) / max(ref, 1e-30),
                                  "err_wanda": quad_error(Wf * mw, G) / max(ref, 1e-30)}
                rec["magnitude_vs_wanda"] = ov
                an = norm
                rec["act_norm_stats"] = {"mean": float(an.mean()), "max": float(an.max()), "median": float(an.median()),
                                         "max_over_median": float(an.max() / an.median().clamp(min=1e-12))}
            W.masked_fill_(mask, 0)
            rec["nonzero_after"] = int(torch.count_nonzero(W))
            records[uid] = rec

    info = run_layerwise(model, batches, layer_fn)
    info["wall_s"] = time.perf_counter() - t0
    return {"records": records, "info": info}


def uniform_plan(units, s: float, rule: str) -> dict[str, dict]:
    return {u.uid: {"rule": rule, "k": k_from_sparsity(s, u.shape[1])} for u in units}


def nm_plan(units, rule: str = "wanda", n_keep: int = 2, m: int = 4) -> dict[str, dict]:
    return {u.uid: {"rule": rule, "nm": (n_keep, m)} for u in units}


def plan_zero_count(units, plan) -> int:
    z = 0
    for u in units:
        sp = plan[u.uid]
        if "nm" in sp:
            n_keep, m = sp["nm"]
            z += u.shape[0] * (u.shape[1] // m) * (m - n_keep)
        else:
            z += u.shape[0] * sp["k"]
    return z


def k_bounds(d_in: int) -> tuple[int, int]:
    import math
    return math.ceil(S_MIN * d_in - 1e-9), math.floor(S_MAX * d_in + 1e-9)
