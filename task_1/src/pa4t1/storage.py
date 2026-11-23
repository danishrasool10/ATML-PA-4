"""Storage accounting: dense FP16, unstructured 16-bit sparse, and packed low-bit weights.

All numbers are *logical / serialized* bytes. They are not GPU allocation and none of them implies
a measured speed-up: the evaluated models are dense FP16 tensors holding zeros or de-quantized
values.

Models
------
* dense FP16:  2 bytes per unique parameter (tied parameters once; buffers such as the rotary
  ``inv_freq`` are derived and not counted).
* unstructured 16-bit sparse (assignment-specified): per eligible matrix
  ``ceil((numel + 16 * nnz) / 8)`` bytes = 1 mask bit per weight + 16 bits per non-zero,
  byte-aligned per matrix. Embeddings, norms, biases and the (tied) LM head stay dense FP16.
* packed low-bit: see ``quant.quant_packed_bytes``; uncompressed parameters stay FP16, plus any
  serialized AWQ compensation scales that could not be folded.
"""
from __future__ import annotations

import torch

from .constants import GROUP_SIZE
from .model_utils import Unit, classify_param, unique_params
from .quant import quant_packed_bytes

FP16_BYTES = 2


def sparse_matrix_bytes(numel: int, nnz: int) -> int:
    return -(-(numel + 16 * nnz) // 8)


def _split_bytes(model, units: list[Unit]) -> dict:
    elig = {id(u.module.weight) for u in units}
    n_elig = sum(u.numel for u in units)
    other = sum(p.numel() for _, p in unique_params(model) if id(p) not in elig)
    total = n_elig + other
    return {"eligible_params": n_elig, "other_params": other, "total_params_unique": total,
            "other_fp16_bytes": FP16_BYTES * other, "eligible_fp16_bytes": FP16_BYTES * n_elig,
            "dense_fp16_total_bytes": FP16_BYTES * total}


def dense_report(model, units: list[Unit]) -> dict:
    return _split_bytes(model, units)


def nonzero_counts(model, units: list[Unit]) -> dict:
    nnz_e = sum(int(torch.count_nonzero(u.weight)) for u in units)
    n_e = sum(u.numel for u in units)
    elig = {id(u.module.weight) for u in units}
    nnz_o = sum(int(torch.count_nonzero(p)) for _, p in unique_params(model) if id(p) not in elig)
    return {"eligible_nnz": nnz_e, "eligible_numel": n_e, "eligible_nonzero_fraction": nnz_e / n_e,
            "eligible_zero_fraction": 1 - nnz_e / n_e, "other_nnz": nnz_o, "total_nnz": nnz_e + nnz_o}


def sparse_report(model, units: list[Unit]) -> dict:
    base = _split_bytes(model, units)
    elig_bytes = 0
    for u in units:
        elig_bytes += sparse_matrix_bytes(u.numel, int(torch.count_nonzero(u.weight)))
    total = elig_bytes + base["other_fp16_bytes"]
    return {**base, "format": "1 mask bit/weight + 16 bits/nonzero, byte-aligned per matrix",
            "eligible_sparse_bytes": elig_bytes, "sparse_total_bytes": total,
            "nominal_bits_per_eligible_weight": 8 * elig_bytes / base["eligible_params"],
            "compression_vs_dense_fp16": base["dense_fp16_total_bytes"] / total}


def quant_report(model, units: list[Unit], bits_by_unit: dict[str, int], group_size: int = GROUP_SIZE,
                 compensation_bytes: int = 0, detail: bool = False) -> dict:
    base = _split_bytes(model, units)
    elig_bytes, per_unit = 0, {}
    parts = {"payload": 0, "scale": 0, "zero_point": 0}
    for u in units:
        r = quant_packed_bytes(u.shape, bits_by_unit[u.uid], group_size)
        elig_bytes += r["total"]
        for k in parts:
            parts[k] += r[k]
        if detail:
            per_unit[u.uid] = {"bits": bits_by_unit[u.uid], **r}
    total = elig_bytes + base["other_fp16_bytes"] + compensation_bytes
    out = {**base, "format": f"packed group-wise asymmetric, group={group_size}, FP16 scale + b-bit zero-point",
           "eligible_packed_bytes": elig_bytes, "eligible_breakdown_bytes": parts,
           "compensation_scale_bytes": compensation_bytes, "packed_total_bytes": total,
           "nominal_bits_per_eligible_weight": 8 * elig_bytes / base["eligible_params"],
           "mean_integer_bits": sum(bits_by_unit[u.uid] * u.numel for u in units) / base["eligible_params"],
           "compression_vs_dense_fp16": base["dense_fp16_total_bytes"] / total}
    if detail:
        out["per_unit"] = per_unit
    return out


def baseline_report(model, units: list[Unit]) -> dict:
    d = dense_report(model, units)
    d["nominal_bits_per_eligible_weight"] = 16.0
    d["sparse_model_estimate_at_current_nnz"] = sparse_report(model, units)
    return d
