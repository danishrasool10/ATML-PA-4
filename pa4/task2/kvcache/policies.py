"""The five compressed-cache methods of Part C (plus the FP16 reference).

All methods start from the identical FP16 prefix cache (``L`` tokens, post-RoPE keys with their original absolute
positions) and receive the same byte budget ``M_4(L)`` (INT4, channel-K / token-V, group 64, trailing ``L mod 64`` in FP16).

  fp16            reference, nothing removed
  sink_recent     C2(i)  FP16 eviction: 8 sink + 64 recent, remainder filled with the most recent remaining positions
  question_aware  C2(ii) FP16 eviction: 8 sink + 64 recent, remainder filled by FP32 question-suffix attention scores
  int4_token      C3(i)  INT4, token-wise K and token-wise V
  int4_channel    C3(ii) INT4, channel-wise K and token-wise V  (== the matched-budget reference cache)
  hybrid          C4     8 sink + 64 recent + top question-aware positions in FP16, every other position INT2
                         channel-K / token-V (complete 64-groups; incomplete tail FP16), sized to the same budget

Retention is *shared* across layers and heads (one index set per prompt), so only ``2 B x kept-tokens`` of index data are
charged against the budget. Kept K/V elements keep their original rotary positions (K is stored post-RoPE; new tokens get
absolute positions L, L+1, ...), the physical cache is simply shorter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .engine import KVCache, LayerCache, QuantSpec
from .quant import (GROUP, INDEX_BYTES, PackedTensor, budget_bytes, fake_quantize, fp16_bytes, fp16_token_capacity, hybrid_bytes,
                    hybrid_capacity)

SINK, RECENT = 8, 64
METHODS = ["sink_recent", "question_aware", "int4_token", "int4_channel", "hybrid"]
NEEDS_SCORES = {"question_aware", "hybrid"}


@dataclass
class Compressed:
    cache: KVCache
    packed_bytes: int                         # realized bytes of the compressed prefix cache (+ indices)
    budget_bytes: int
    kept_fp16: list[int] | None = None        # prefix positions held in FP16 (None: all, i.e. quantization-only methods)
    kept_any: list[int] | None = None         # prefix positions still present at any precision
    k_err: float | None = None                # relative L2 error of dequantized prefix K / V (quantized methods)
    v_err: float | None = None
    info: dict = field(default_factory=dict)


def _select(cache: KVCache, idx: torch.Tensor) -> KVCache:
    return KVCache([LayerCache(l.K.index_select(-2, idx).contiguous(), l.V.index_select(-2, idx).contiguous()) for l in cache.layers])


def safeguards(L: int) -> torch.Tensor:
    s = torch.arange(0, min(SINK, L))
    r = torch.arange(max(L - RECENT, SINK), L)
    return torch.unique(torch.cat([s, r]))


def fill_by_scores(L: int, n_total: int, scores: torch.Tensor) -> torch.Tensor:
    """Safeguards + highest-score remaining positions up to ``n_total`` kept tokens (sorted positional order)."""
    keep = torch.zeros(L, dtype=torch.bool, device=scores.device)
    keep[safeguards(L).to(scores.device)] = True
    rest = n_total - int(keep.sum())
    if rest > 0:
        sc = scores.clone().float()
        sc[keep] = float("-inf")
        keep[torch.topk(sc, min(rest, int((~keep).sum()))).indices] = True
    return torch.nonzero(keep).flatten()


def fill_by_recency(L: int, n_total: int, device) -> torch.Tensor:
    return fill_by_scores(L, n_total, torch.arange(L, device=device, dtype=torch.float32))   # later position = higher score


def _quantize_prefix(cache: KVCache, bits: int, k_layout: str, v_layout: str, n_full: int | None = None):
    """Fake-quantize the first ``n_full`` (default floor(L/64)*64) tokens of every layer; returns realized packed bytes and
    the relative reconstruction errors of K and V over the quantized region."""
    L = cache.n
    n_full = (L // GROUP) * GROUP if n_full is None else n_full
    total, ek, ev, nk, nv = 0, 0.0, 0.0, 0.0, 0.0
    for lc in cache.layers:
        for name, layout in (("K", k_layout), ("V", v_layout)):
            x = getattr(lc, name)
            if n_full > 0:
                pt = PackedTensor(x[..., :n_full, :], bits, layout)
                total += pt.nbytes()
                xq = pt.decode()
                err = (xq.float() - x[..., :n_full, :].float()).pow(2).sum().item()
                den = x[..., :n_full, :].float().pow(2).sum().item()
                x[..., :n_full, :] = xq
                if name == "K":
                    ek, nk = ek + err, nk + den
                else:
                    ev, nv = ev + err, nv + den
            total += 2 * x[..., n_full:, :].numel()                                   # FP16 tail
    rel = lambda e, n: float((e / max(n, 1e-30)) ** 0.5)
    return total, rel(ek, nk), rel(ev, nv), n_full


def compress(method: str, cache: KVCache, scores: torch.Tensor | None, cfg_dims: tuple[int, int, int]) -> Compressed:
    """``cache`` is a *fresh clone* of the FP16 prefix cache and is modified/consumed. ``cfg_dims`` = (n_layers, H_kv, d_h)."""
    n_l, H, d = cfg_dims
    L = cache.n
    B4 = budget_bytes(4, L, n_l, H, d)
    dev = cache.layers[0].K.device
    if method == "fp16":
        return Compressed(cache, fp16_bytes(L, n_l, H, d), B4, None, list(range(L)))
    if method in ("sink_recent", "question_aware"):
        T = min(fp16_token_capacity(B4, n_l, H, d), L)
        idx = fill_by_recency(L, T, dev) if method == "sink_recent" else fill_by_scores(L, T, scores)
        out = _select(cache, idx)
        nb = 4 * n_l * H * d * idx.numel() + INDEX_BYTES * idx.numel()
        assert nb <= B4
        il = idx.tolist()
        return Compressed(out, nb, B4, il, il, info={"n_kept": len(il), "n_safeguard": int(safeguards(L).numel())})
    if method in ("int4_token", "int4_channel"):
        k_layout = "token" if method == "int4_token" else "channel"
        nb, ek, ev, n_full = _quantize_prefix(cache, 4, k_layout, "token")
        for lc in cache.layers:
            lc.spec, lc.n_quant, lc.quantize_new = QuantSpec(4, k_layout, "token"), n_full, True
        return Compressed(cache, nb, B4, None, list(range(L)), ek, ev, {"n_quantized": n_full, "n_fp16_tail": L - n_full})
    if method == "hybrid":
        n_fp = hybrid_capacity(B4, L, n_l, H, d, bits=2)
        fp_idx = fill_by_scores(L, n_fp, scores)
        mask = torch.ones(L, dtype=torch.bool, device=dev)
        mask[fp_idx] = False
        q_idx = torch.nonzero(mask).flatten()                                          # INT2 stream, positional order
        n_q = q_idx.numel()
        n_full = (n_q // GROUP) * GROUP
        order = torch.cat([q_idx, fp_idx])                                             # [INT2 stream (tail FP16) | FP16 set]
        out = _select(cache, order)
        nb, ek, ev, _ = _quantize_prefix(out, 2, "channel", "token", n_full)
        nb += INDEX_BYTES * fp_idx.numel()
        for lc in out.layers:
            lc.spec, lc.n_quant, lc.quantize_new = QuantSpec(2, "channel", "token"), n_full, False   # new tokens stay FP16
        assert nb <= B4 and nb == hybrid_bytes(n_fp, L, n_l, H, d, 2), (nb, B4, hybrid_bytes(n_fp, L, n_l, H, d, 2))
        return Compressed(out, nb, B4, fp_idx.tolist(), list(range(L)), ek, ev,
                          {"n_fp16": n_fp, "n_int2": n_q, "n_int2_quantized": n_full, "n_selected_extra": n_fp - int(safeguards(L).numel())})
    raise ValueError(method)
