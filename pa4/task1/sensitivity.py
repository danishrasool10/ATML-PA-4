"""Sensitivity probes for P3 (pruning) and Q3 (mixed precision). All use calibration data only.

Pruning sensitivities (unit = individual projection; 7 x n_layers units), all for the *Wanda* ranking:
  owl        OWL-style outlier ratio: fraction of the unit's Wanda scores exceeding M x the unit mean
             (Yin et al. 2024, M = 5). Scalar -> power-law allocation (more outliers = prune less).
  local_rel  calibration layer-output error ||X dW^T||^2 / ||X W^T||^2 of Wanda pruning at each
             sparsity level in LEVELS (exact, from the Gram matrix). A loss table L_u(s).
  local_abs  the same error without the normalisation (absolute squared output perturbation).
  taylor     Fisher/second-order Taylor prediction of the loss increase of pruning the Wanda set,
             0.5 * sum_{pruned} F_ij w_ij^2 with the empirical diagonal Fisher F_ij = E_seq[g_ij^2]
             of the calibration NLL (Molchanov et al. 2019 / OBD-style), also tabulated over LEVELS.
Quantization sensitivities (see ``q3_tables``):
  local error tables E[u, b] of the AWQ quantizer (from the homogeneous AWQ passes) and the
  HAWQ-V2-style Omega[u, b] = Tr(F_u) * ||W - W_hat_b||^2 / n_u * ... = fisher_trace_u * weight_mse_{u,b}.

Limitation (assignment "What to watch for"): all of these are *isolated* measurements -- each unit is
perturbed alone with dense inputs -- and cannot capture interactions that appear when many layers
are compressed together.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .calibration import quad_error, run_layerwise
from .constants import LEVELS, OWL_M, PROJ_NAMES, unit_id
from .model_utils import get_proj
from .pruning import k_from_sparsity, rank_matrix, score_wanda


def collect_pruning_tables(model, units, batches, levels=LEVELS) -> dict:
    """Layer-wise pass (no weight changes): Wanda local error tables, OWL outlier ratios, activation
    norms and per-row threshold scores (needed by the Fisher pass to reconstruct the Wanda masks)."""
    n = len(units)
    L = len(levels)
    idx = {u.uid: i for i, u in enumerate(units)}
    out = {"levels": list(levels), "uids": [u.uid for u in units],
           "local_abs": np.zeros((n, L)), "local_rel": np.zeros((n, L)), "ref_energy": np.zeros(n),
           "owl_ratio": np.zeros(n), "act_norm": {}, "thresholds": {}, "mean_score": np.zeros(n)}
    t0 = time.perf_counter()

    def layer_fn(li, layer, stats, ctx):
        for name in PROJ_NAMES:
            uid = unit_id(li, name)
            i = idx[uid]
            W = get_proj(layer, name).weight.data
            Wf = W.float()
            norm = stats.act_norm(name)
            G = stats.gram(name)
            score = score_wanda(W, norm)
            rank = rank_matrix(score)
            ref = quad_error(Wf, G)
            out["ref_energy"][i] = ref
            thr = torch.sort(score, dim=1).values
            ths = []
            for l, s in enumerate(levels):
                k = k_from_sparsity(s, W.shape[1])
                err = quad_error(Wf * (rank < k), G)
                out["local_abs"][i, l] = err
                out["local_rel"][i, l] = err / max(ref, 1e-30)
                ths.append(thr[:, k - 1].cpu() if k > 0 else torch.full((W.shape[0],), -1.0))
            out["thresholds"][uid] = torch.stack(ths)  # [levels, d_out]
            mean = float(score.mean())
            out["mean_score"][i] = mean
            out["owl_ratio"][i] = float((score > OWL_M * mean).float().mean())
            out["act_norm"][uid] = norm.cpu()
            del rank, thr

    info = run_layerwise(model, batches, layer_fn)
    info["wall_s"] = time.perf_counter() - t0
    out["info"] = info
    return out


def fisher_pass(model, units, calib_seqs: list[list[int]], tables: dict, device=None, max_grad_scale_check=True) -> dict:
    """One backward pass per calibration sequence (batch size 1, mean token NLL), using per-parameter
    post-accumulate-grad hooks so no gradient tensor is kept alive.

    For every eligible weight it accumulates, averaged over sequences,
        trace_u   = sum_ij g_ij^2                                  (HAWQ-V2-style Fisher trace)
        taylor_u[l] = 0.5 * sum_{(i,j) in Wanda mask at level l} g_ij^2 w_ij^2
    The model should be in float32 for this pass (the caller upcasts and restores).
    """
    device = device or next(model.parameters()).device
    levels = tables["levels"]
    L = len(levels)
    uid_of = {id(u.module.weight): u.uid for u in units}
    acc_trace = {u.uid: 0.0 for u in units}
    acc_taylor = {u.uid: torch.zeros(L, dtype=torch.float64) for u in units}
    thr = {uid: t.to(device) for uid, t in tables["thresholds"].items()}
    norms = {uid: a.to(device) for uid, a in tables["act_norm"].items()}
    handles, params = [], []

    def make_hook(uid):
        def hook(p):
            g = p.grad.float()
            w = p.data.float()
            acc_trace[uid] += float((g * g).sum(dtype=torch.float64))
            gw2 = (g * w) ** 2
            score = w.abs() * norms[uid][None, :]
            for l in range(L):
                m = score <= thr[uid][l][:, None]
                acc_taylor[uid][l] += 0.5 * float((gw2 * m).sum(dtype=torch.float64))
            p.grad = None
        return hook

    for u in units:
        p = u.module.weight
        p.requires_grad_(True)
        handles.append(p.register_post_accumulate_grad_hook(make_hook(u.uid)))
        params.append(p)
    t0 = time.perf_counter()
    n_ok = 0
    try:
        for seq in calib_seqs:
            ids = torch.tensor([seq], dtype=torch.long, device=device)
            with torch.enable_grad():
                logits = model(input_ids=ids, use_cache=False).logits
                loss = torch.nn.functional.cross_entropy(logits[0, :-1].float(), ids[0, 1:])
                loss.backward()
            n_ok += 1
    finally:
        for h in handles:
            h.remove()
        for p in params:
            p.requires_grad_(False)
            p.grad = None
    return {"trace": np.array([acc_trace[u.uid] / n_ok for u in units]),
            "taylor": np.stack([(acc_taylor[u.uid] / n_ok).numpy() for u in units]),
            "n_sequences": n_ok, "wall_s": time.perf_counter() - t0,
            "loss": "mean token NLL per sequence (empirical Fisher with sequence-level gradients)"}


def build_prune_sigma(tables: dict, fisher: dict | None, design: str, s_global: float) -> np.ndarray | None:
    """Scalar sensitivity per unit (used by the power-law rule) -- None for table-based designs."""
    if design == "owl":
        return tables["owl_ratio"]
    raise KeyError(design)


def loss_table(tables: dict, fisher: dict | None, design: str) -> np.ndarray:
    if design == "local_rel":
        return tables["local_rel"]
    if design == "local_abs":
        return tables["local_abs"]
    if design == "taylor":
        assert fisher is not None
        return fisher["taylor"]
    raise KeyError(design)
