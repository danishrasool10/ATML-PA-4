"""Group-wise asymmetric KV quantizers with real bit packing, plus the Part C budget / byte accounting.

Conventions (assignment, Part C):
  * group size 64 (``GROUP``); each group stores an FP16 scale and an FP16 offset (4 metadata bytes),
  * ``token`` layout: a group is 64 consecutive channels of one token of one head,
  * ``channel`` layout: a group is 64 consecutive tokens of one channel of one head,
  * the quantizer uses the FP16-rounded scale/offset for both encoding and decoding, so
    ``dequantize(quantize(x))`` is exactly what a packed cache would reproduce.
All functions accept any number of leading dims ``(..., n_tokens, d)``.
"""
from __future__ import annotations

import torch

GROUP = 64
META_BYTES = 4          # FP16 scale + FP16 offset
INDEX_BYTES = 2         # one uint16 position index per explicitly indexed token (L < 65536)


# --------------------------------------------------------------------------- budget math
def budget_bytes(bits: int, L: int, n_layers: int, n_kv: int, d_h: int) -> int:
    """M_b(L) = 2 n_l H_kv d_h [ floor(L/64) (64 b/8 + 4) + 2 (L mod 64) ]   (K and V, per channel)."""
    full, r = divmod(L, GROUP)
    return 2 * n_layers * n_kv * d_h * (full * (GROUP * bits // 8 + META_BYTES) + 2 * r)


def fp16_bytes(L: int, n_layers: int, n_kv: int, d_h: int) -> int:
    """M_16(L) = 4 n_l H_kv d_h L."""
    return 4 * n_layers * n_kv * d_h * L


def fp16_token_capacity(budget: int, n_layers: int, n_kv: int, d_h: int, index_bytes: int = INDEX_BYTES) -> int:
    """Largest T with 4 n_l H d T + index_bytes * T <= budget (shared retention: one index set for all layers/heads)."""
    return int(budget // (4 * n_layers * n_kv * d_h + index_bytes))


def hybrid_bytes(n_fp: int, L: int, n_layers: int, n_kv: int, d_h: int, bits: int = 2) -> int:
    """Bytes of the hybrid cache: ``n_fp`` FP16 tokens (indexed) + the other ``L - n_fp`` tokens as a channel-K / token-V
    INT``bits`` stream in complete 64-token groups, whose incomplete tail stays FP16 (its position is implied by the
    stream order, so only the FP16-set indices are stored)."""
    n_q = L - n_fp
    full, r = divmod(n_q, GROUP)
    per_channel = full * (GROUP * bits // 8 + META_BYTES) + 2 * (r + n_fp)
    return 2 * n_layers * n_kv * d_h * per_channel + INDEX_BYTES * n_fp


def hybrid_capacity(budget: int, L: int, n_layers: int, n_kv: int, d_h: int, bits: int = 2) -> int:
    """Largest feasible n_fp (not monotone in n_fp because of the FP16 tail, so every value is checked)."""
    best = 0
    for n_fp in range(0, L + 1):
        if hybrid_bytes(n_fp, L, n_layers, n_kv, d_h, bits) <= budget:
            best = n_fp
    return best


# --------------------------------------------------------------------------- quantizer
def _group_view(x: torch.Tensor, layout: str):
    """Reshape so the last dim is the group axis; returns (view, restore_fn)."""
    *lead, n, d = x.shape
    if layout == "token":
        assert d % GROUP == 0, f"token layout needs head_dim % {GROUP} == 0, got {d}"
        g = x.reshape(*lead, n, d // GROUP, GROUP)
        return g, lambda y: y.reshape(*lead, n, d)
    if layout == "channel":
        assert n % GROUP == 0, f"channel layout needs a multiple of {GROUP} tokens, got {n}"
        g = x.reshape(*lead, n // GROUP, GROUP, d).transpose(-1, -2)      # (..., n/64, d, 64)
        return g, lambda y: y.transpose(-1, -2).reshape(*lead, n, d)
    raise ValueError(layout)


def quantize(x: torch.Tensor, bits: int, layout: str):
    """Returns (codes uint8, scale fp16, offset fp16, restore_fn); codes live in the group view."""
    levels = (1 << bits) - 1
    g, restore = _group_view(x.float(), layout)
    lo = g.amin(dim=-1, keepdim=True)
    hi = g.amax(dim=-1, keepdim=True)
    scale = ((hi - lo) / levels).to(torch.float16)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)           # constant group: all codes 0
    off = lo.to(torch.float16)
    codes = torch.round((g - off.float()) / scale.float()).clamp_(0, levels).to(torch.uint8)
    return codes, scale, off, restore


def dequantize(codes, scale, off, restore) -> torch.Tensor:
    return restore((codes.float() * scale.float() + off.float()).to(torch.float16))


def fake_quantize(x: torch.Tensor, bits: int, layout: str) -> torch.Tensor:
    return dequantize(*quantize(x, bits, layout))


# --------------------------------------------------------------------------- bit packing
def pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack uint8 codes along the last dim (b in {2,4,8}) into uint8 bytes (little-endian within a byte)."""
    if bits == 8:
        return codes.contiguous()
    per = 8 // bits
    assert codes.shape[-1] % per == 0
    c = codes.reshape(*codes.shape[:-1], codes.shape[-1] // per, per).to(torch.int32)
    shifts = torch.arange(per, device=codes.device, dtype=torch.int32) * bits
    return (c << shifts).sum(dim=-1).to(torch.uint8)


def unpack_codes(packed: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 8:
        return packed
    per = 8 // bits
    shifts = torch.arange(per, device=packed.device, dtype=torch.int32) * bits
    mask = (1 << bits) - 1
    c = (packed.to(torch.int32).unsqueeze(-1) >> shifts) & mask
    return c.reshape(*packed.shape[:-1], packed.shape[-1] * per).to(torch.uint8)


class PackedTensor:
    """Serialised group-quantized tensor: packed payload + FP16 scale/offset (+ the shape needed to decode)."""

    def __init__(self, x: torch.Tensor, bits: int, layout: str):
        codes, scale, off, self._restore = quantize(x, bits, layout)
        self.bits, self.layout, self.shape = bits, layout, tuple(x.shape)
        self.payload = pack_codes(codes, bits)
        self.scale, self.off = scale, off

    def nbytes(self) -> int:
        return self.payload.numel() + 2 * self.scale.numel() + 2 * self.off.numel()

    def decode(self) -> torch.Tensor:
        return dequantize(unpack_codes(self.payload, self.bits), self.scale, self.off, self._restore)


def layout_bytes(n_groups_tokens: int, lead: int, d: int, bits: int, layout: str) -> int:
    """Analytic packed bytes for ``lead`` independent (head) slices of ``n_groups_tokens`` tokens x d channels."""
    n_el = lead * n_groups_tokens * d
    n_grp = n_el // GROUP
    return n_el * bits // 8 + META_BYTES * n_grp
