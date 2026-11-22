"""Group-wise asymmetric weight-only quantizer (the single quantizer shared by Q1, Q2 and Q3).

Conventions (fixed across Q1/Q2/Q3, see README "Quantization conventions"):
  * groups of 128 *consecutive input columns* within each output row; a trailing partial group
    (when d_in % 128 != 0) is quantized on its real columns only -- no padding values are stored,
  * per group: w_min' = min(w_min, 0), w_max' = max(w_max, 0) (so the integer zero-point lies in
    [0, 2^b - 1] and exact zeros stay exactly representable),
    s = max((w_max' - w_min') / (2^b - 1), 1e-5) rounded to FP16 (the stored precision),
    z = clip(-round(w_min' / s), 0, 2^b - 1) (integer),
    q = clip(round(w / s) + z, 0, 2^b - 1) with round-half-to-even (torch.round),
    w_hat = s (q - z),
  * no clipping search (min/max range), no outlier handling,
  * packing: the b-bit codes of a matrix are written row-major as one little-endian bit stream
    (value i occupies bits [i*b, (i+1)*b)), zero-points are packed the same way with b bits per
    group, FP16 scales are stored as 2 bytes per group; each of the three streams is padded to a
    whole byte.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .constants import GROUP_SIZE


@dataclass
class QuantResult:
    deq: torch.Tensor      # float32 dequantized weights, same shape as input
    q: torch.Tensor        # uint8 codes [dout, din]
    scale: torch.Tensor    # float16 [dout, n_groups]
    zero: torch.Tensor     # uint8 integer zero-points [dout, n_groups]
    bits: int
    group_size: int


def _quant_block(w: torch.Tensor, bits: int):
    levels = float(2 ** bits - 1)
    wmax = w.amax(dim=-1, keepdim=True).clamp(min=0.0)
    wmin = w.amin(dim=-1, keepdim=True).clamp(max=0.0)
    scale = ((wmax - wmin) / levels).clamp(min=1e-5).to(torch.float16).float()  # stored precision
    zero = (-torch.round(wmin / scale)).clamp(0, levels)
    q = torch.clamp(torch.round(w / scale) + zero, 0, levels)
    deq = (q - zero) * scale
    return deq, q, scale, zero


def quantize_groupwise(w: torch.Tensor, bits: int, group_size: int = GROUP_SIZE) -> QuantResult:
    assert w.ndim == 2 and 2 <= bits <= 8
    w = w.float()
    dout, din = w.shape
    n_full, rem = divmod(din, group_size)
    deqs, qs, scs, zs = [], [], [], []
    if n_full:
        wf = w[:, : n_full * group_size].reshape(dout, n_full, group_size)
        d, q, s, z = _quant_block(wf, bits)
        deqs.append(d.reshape(dout, -1))
        qs.append(q.reshape(dout, -1))
        scs.append(s.reshape(dout, n_full))
        zs.append(z.reshape(dout, n_full))
    if rem:
        d, q, s, z = _quant_block(w[:, n_full * group_size :], bits)
        deqs.append(d)
        qs.append(q)
        scs.append(s)
        zs.append(z)
    return QuantResult(
        deq=torch.cat(deqs, 1),
        q=torch.cat(qs, 1).to(torch.uint8),
        scale=torch.cat(scs, 1).to(torch.float16),
        zero=torch.cat(zs, 1).to(torch.uint8),
        bits=bits,
        group_size=group_size,
    )


def n_groups(dout: int, din: int, group_size: int = GROUP_SIZE) -> int:
    return dout * (-(-din // group_size))


def quant_packed_bytes(shape: tuple[int, int], bits: int, group_size: int = GROUP_SIZE) -> dict:
    """Exact packed size of one quantized matrix (payload + FP16 scales + packed zero-points)."""
    dout, din = shape
    ng = n_groups(dout, din, group_size)
    payload = -(-(dout * din * bits) // 8)
    scale = 2 * ng
    zp = -(-(ng * bits) // 8)
    return {"payload": payload, "scale": scale, "zero_point": zp, "total": payload + scale + zp,
            "n_groups": ng, "bits_per_weight_effective": 8 * (payload + scale + zp) / (dout * din)}


# ---------- bit packing (used to *verify* the byte formula, and to serialize) ----------

def pack_bits(values: np.ndarray, bits: int) -> bytes:
    """Little-endian bit-stream packing of unsigned ints < 2**bits."""
    v = np.asarray(values, dtype=np.uint16).reshape(-1)
    assert v.max(initial=0) < (1 << bits)
    shifts = np.arange(bits, dtype=np.uint16)
    bit_matrix = ((v[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
    return np.packbits(bit_matrix.reshape(-1), bitorder="little").tobytes()


def unpack_bits(data: bytes, bits: int, count: int) -> np.ndarray:
    raw = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")[: count * bits]
    m = raw.reshape(count, bits).astype(np.uint16)
    return (m << np.arange(bits, dtype=np.uint16)[None, :]).sum(1).astype(np.uint8)


def pack_matrix(res: QuantResult) -> dict[str, bytes]:
    return {
        "payload": pack_bits(res.q.cpu().numpy(), res.bits),
        "scale": res.scale.cpu().numpy().astype("<f2").tobytes(),
        "zero_point": pack_bits(res.zero.cpu().numpy(), res.bits),
    }


def unpack_matrix(blobs: dict[str, bytes], shape: tuple[int, int], bits: int, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Dequantize from packed bytes (inverse of ``pack_matrix``) to float32."""
    dout, din = shape
    ngr = -(-din // group_size)
    q = unpack_bits(blobs["payload"], bits, dout * din).reshape(dout, din).astype(np.float32)
    scale = np.frombuffer(blobs["scale"], dtype="<f2").astype(np.float32).reshape(dout, ngr)
    zero = unpack_bits(blobs["zero_point"], bits, dout * ngr).reshape(dout, ngr).astype(np.float32)
    col_group = np.arange(din) // group_size
    return torch.from_numpy((q - zero[:, col_group]) * scale[:, col_group])
