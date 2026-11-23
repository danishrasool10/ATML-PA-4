"""Q1 (uniform group-wise PTQ), Q2 (AWQ-style activation-aware scaling) and the application of a
per-projection bit assignment (Q3) -- all share the same quantizer from ``quant.py``.

What this implementation reproduces from AWQ (Lin et al., 2024), and what it does not
---------------------------------------------------------------------------------------
Reproduced:
  * per-input-channel scaling s_j = x_mean_j^alpha / sqrt(max(s) * min(s)) derived from the mean
    absolute calibration activation of that channel, weights multiplied by s before quantization,
  * inverse compensation folded into the preceding operator, algebraically exactly:
        input_layernorm  -> q/k/v   (RMSNorm weight /= s; one scale shared by q, k, v)
        v_proj           -> o_proj  (v_proj rows & bias /= s; with GQA the scale is tied across the
                                     query heads that share a KV head so the fold stays exact)
        post_attn_norm   -> gate/up (RMSNorm weight /= s)
        up_proj          -> down    (up_proj rows /= s; exact because SwiGLU multiplies elementwise)
    hence no compensation vectors need to be serialized (their byte count is reported = 0),
  * a grid search of alpha over a small, recorded grid (default {0.0, 0.1, ..., 1.0}; alpha = 0 is
    the unscaled Q1 solution so the search can never be worse than Q1 on its own criterion),
  * the full-precision invariance of the rescale/inverse-compensation is verified per block.
Not reproduced (this is NOT the optimized AWQ inference engine):
  * no weight-clipping search (AWQ's second stage), the quantizer always uses the plain min/max
    group range so Q1/Q2/Q3 share one quantizer,
  * the search criterion is the exact *local* output error of each scaled-projection group on the
    128 calibration sequences, tr(dW G dW^T) with dense calibration inputs, not the error of the
    whole attention / MLP block output with the nonlinearity in between,
  * no fused W4A16 CUDA kernels: weights are de-quantized to FP16 for evaluation (fake quantization)
    and executed with dense FP16 matmuls.
"""
from __future__ import annotations

import copy
import time

import torch

from .calibration import layer_forward, quad_error, run_layerwise
from .constants import AWQ_GRID, GROUP_SIZE, PROJ_NAMES, unit_id
from .model_utils import get_proj
from .quant import pack_matrix, quant_packed_bytes, quantize_groupwise, unpack_matrix


def make_scale(x_mean: torch.Tensor, alpha: float) -> torch.Tensor:
    s = x_mean.float().clamp(min=1e-4).pow(alpha)
    s = s / (s.max() * s.min()).sqrt()
    return s.clamp(min=1e-4)


def _eff(W: torch.Tensor, col_scale: torch.Tensor, bits: int, gs: int):
    """Quantize W*diag(col_scale), return (QuantResult, effective weight in the original basis)."""
    qr = quantize_groupwise(W * col_scale[None, :], bits, gs)
    return qr, qr.deq / col_scale[None, :]


def search_scale(G: torch.Tensor, x_mean: torch.Tensor, Ws: list[torch.Tensor], bits: list[int], gs: int,
                 grid=AWQ_GRID, expand=None) -> dict:
    """Grid search of alpha minimising the summed local output error of the projections in the group.

    ``expand`` maps a scale vector of length d' to the length-d_in column scale (used for the
    v_proj -> o_proj fold with grouped-query attention).
    """
    losses = []
    best = (float("inf"), None, None)
    for a in grid:
        s = make_scale(x_mean, a)
        sf = expand(s) if expand is not None else s
        loss = 0.0
        for W, b in zip(Ws, bits):
            _, What = _eff(W, sf, b, gs)
            loss += quad_error(What - W, G)
        losses.append(loss)
        if loss < best[0]:
            best = (loss, a, s)
    return {"alpha": best[1], "scale": best[2], "grid": list(grid), "grid_losses": losses, "best_loss": best[0],
            "loss_at_alpha0": losses[0]}


def _cast(obj, dtype):
    if torch.is_tensor(obj):
        return obj.to(dtype) if obj.is_floating_point() else obj
    if isinstance(obj, tuple):
        return tuple(_cast(o, dtype) for o in obj)
    if isinstance(obj, list):
        return [_cast(o, dtype) for o in obj]
    if isinstance(obj, dict):
        return {k: _cast(v, dtype) for k, v in obj.items()}
    return obj


def invariance_check(layer, scaled: dict, sample, tol: float) -> float:
    """Verify (in fp32, or fp64 if the model is fp64, *before any rounding to low bits*) that applying
    the scaling + inverse compensation leaves the block output unchanged. Returns the max relative error."""
    dt = torch.float64 if next(layer.parameters()).dtype == torch.float64 else torch.float32
    ref_layer = copy.deepcopy(layer).to(dt)
    new_layer = copy.deepcopy(layer).to(dt)
    with torch.no_grad():
        for name, w in scaled["weights"].items():
            get_proj(new_layer, name).weight.copy_(w)
        new_layer.self_attn.v_proj.bias.copy_(scaled["v_bias"])
        new_layer.input_layernorm.weight.copy_(scaled["ln1"])
        new_layer.post_attention_layernorm.weight.copy_(scaled["ln2"])
        h, args, kw = sample
        h, args, kw = _cast(h, dt), _cast(args, dt), _cast(kw, dt)
        a = layer_forward(ref_layer, h, args, kw)
        b = layer_forward(new_layer, h, args, kw)
    rel = float((a - b).abs().max() / a.abs().max().clamp(min=1e-12))
    assert rel < tol, f"AWQ rescaling changed the full-precision block output (rel err {rel:.3e} >= {tol})"
    return rel


def compress_quant_model(model, units, batches, bits_by_unit: dict[str, int], *, mode: str, group_size: int = GROUP_SIZE,
                         grid=AWQ_GRID, apply: bool = True, verify_invariance: bool = True, inv_tol: float = 1e-4,
                         verify_packing: bool = False, fold: bool = True) -> dict:
    """Quantize every eligible projection of ``model`` in place (``apply=True``).

    mode "q1": plain group-wise PTQ (alpha = 0, nothing folded). mode "awq": Q2/Q3 scaling search.
    ``bits_by_unit`` gives the integer width of each projection (uniform for Q1/Q2, mixed for Q3).
    Returns per-unit records (calibration errors in the *original* basis, packed bytes, chosen alpha),
    group search records and the number of serialized compensation-scale bytes.
    """
    assert mode in ("q1", "awq")
    cfg = model.config
    n_heads, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = cfg.hidden_size // n_heads
    n_rep = n_heads // n_kv
    records: dict[str, dict] = {}
    groups: dict[str, dict] = {}
    totals = {"compensation_scale_bytes": 0, "max_invariance_rel_err": 0.0, "packing_verified": 0}
    t0 = time.perf_counter()

    def expand_kv(sv: torch.Tensor) -> torch.Tensor:  # [n_kv*hd] -> [n_heads*hd]
        return sv.view(n_kv, 1, hd).expand(n_kv, n_rep, hd).reshape(-1)

    def layer_fn(li, layer, stats, ctx):
        bits = {n: bits_by_unit[unit_id(li, n)] for n in PROJ_NAMES}
        W = {n: get_proj(layer, n).weight.data.float().clone() for n in PROJ_NAMES}
        G = {n: stats.gram(n) for n in PROJ_NAMES}
        dev = W["q_proj"].device
        ones = lambda n: torch.ones(n, device=dev)
        d_h, d_i = W["q_proj"].shape[1], W["down_proj"].shape[1]
        s_qkv, s_v, s_gu, s_dn = ones(d_h), ones(n_kv * hd), ones(d_h), ones(d_i)
        ginfo = {}
        if mode == "awq":
            r = search_scale(G["q_proj"], stats.abs_mean("qkv"), [W["q_proj"], W["k_proj"], W["v_proj"]],
                             [bits["q_proj"], bits["k_proj"], bits["v_proj"]], group_size, grid)
            s_qkv, ginfo["qkv"] = r["scale"], r
            xm_o = stats.abs_mean("o").view(n_kv, n_rep, hd).mean(1).reshape(-1)
            r = search_scale(G["o_proj"], xm_o, [W["o_proj"]], [bits["o_proj"]], group_size, grid, expand=expand_kv)
            s_v, ginfo["o"] = r["scale"], r
            r = search_scale(G["gate_proj"], stats.abs_mean("gu"), [W["gate_proj"], W["up_proj"]],
                             [bits["gate_proj"], bits["up_proj"]], group_size, grid)
            s_gu, ginfo["gu"] = r["scale"], r
            r = search_scale(G["down_proj"], stats.abs_mean("down"), [W["down_proj"]], [bits["down_proj"]], group_size, grid)
            s_dn, ginfo["down"] = r["scale"], r
        s_o = expand_kv(s_v)
        # pre-quantization (scaled) weights with the inverse folded into the adjacent operator
        pre = {
            "q_proj": W["q_proj"] * s_qkv[None, :],
            "k_proj": W["k_proj"] * s_qkv[None, :],
            "v_proj": (W["v_proj"] * s_qkv[None, :]) / s_v[:, None],
            "o_proj": W["o_proj"] * s_o[None, :],
            "gate_proj": W["gate_proj"] * s_gu[None, :],
            "up_proj": (W["up_proj"] * s_gu[None, :]) / s_dn[:, None],
            "down_proj": W["down_proj"] * s_dn[None, :],
        }
        v_bias = layer.self_attn.v_proj.bias.data.float() / s_v if layer.self_attn.v_proj.bias is not None else None
        ln1 = layer.input_layernorm.weight.data.float() / s_qkv
        ln2 = layer.post_attention_layernorm.weight.data.float() / s_gu
        if mode == "awq" and verify_invariance and v_bias is not None:
            rel = invariance_check(layer, {"weights": pre, "v_bias": v_bias, "ln1": ln1, "ln2": ln2}, ctx["sample"], inv_tol)
            totals["max_invariance_rel_err"] = max(totals["max_invariance_rel_err"], rel)
        if mode == "awq" and not fold:
            # alternative accounting: serialize the compensating FP16 channel scales instead of folding
            totals["compensation_scale_bytes"] += 2 * (d_h + n_kv * hd + d_h + d_i)
        # effective (original-basis) weights after quantization, and the column/row scale of each
        col = {"q_proj": s_qkv, "k_proj": s_qkv, "v_proj": s_qkv, "o_proj": s_o, "gate_proj": s_gu, "up_proj": s_gu, "down_proj": s_dn}
        row_inv = {"v_proj": s_v, "up_proj": s_dn}
        for n in PROJ_NAMES:
            uid = unit_id(li, n)
            qr = quantize_groupwise(pre[n], bits[n], group_size)
            What = qr.deq / col[n][None, :]
            if n in row_inv:
                What = What * row_inv[n][:, None]
            dW = What - W[n]
            ref = quad_error(W[n], G[n])
            err = quad_error(dW, G[n])
            pk = quant_packed_bytes(tuple(W[n].shape), bits[n], group_size)
            rec = {"bits": bits[n], "abs_err": err, "ref_energy": ref, "rel_err": err / max(ref, 1e-30),
                   "weight_mse": float((dW ** 2).mean()), "packed_bytes": pk["total"], "numel": W[n].numel()}
            if mode == "awq":
                grp = {"q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv", "o_proj": "o", "gate_proj": "gu", "up_proj": "gu", "down_proj": "down"}[n]
                rec["alpha"] = ginfo[grp]["alpha"]
            if verify_packing:
                blobs = pack_matrix(qr)
                got = sum(len(v) for v in blobs.values())
                assert got == pk["total"], (uid, got, pk["total"])
                back = unpack_matrix(blobs, tuple(W[n].shape), bits[n], group_size)
                assert torch.allclose(back, qr.deq.cpu(), atol=0, rtol=0), f"pack/unpack mismatch {uid}"
                totals["packing_verified"] += 1
            records[uid] = rec
            if apply:
                get_proj(layer, n).weight.data.copy_(qr.deq.to(get_proj(layer, n).weight.dtype))
        if apply:
            if v_bias is not None:
                layer.self_attn.v_proj.bias.data.copy_(v_bias.to(layer.self_attn.v_proj.bias.dtype))
            layer.input_layernorm.weight.data.copy_(ln1.to(layer.input_layernorm.weight.dtype))
            layer.post_attention_layernorm.weight.data.copy_(ln2.to(layer.post_attention_layernorm.weight.dtype))
        if mode == "awq":
            groups[f"L{li:02d}"] = {g: {k: v for k, v in r.items() if k != "scale"} for g, r in ginfo.items()}

    info = run_layerwise(model, batches, layer_fn)
    info.update(totals)
    info["wall_s"] = time.perf_counter() - t0
    info["mode"] = mode
    info["awq_grid"] = list(grid)
    return {"records": records, "groups": groups, "info": info}
