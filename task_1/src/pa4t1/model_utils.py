"""Model loading, eligible-projection discovery, snapshot/restore and fingerprinting."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .constants import ATTN_PROJS, MLP_PROJS, PROJ_NAMES, unit_id


def load_model(name: str, revision: str | None = None, dtype=torch.float16, device: str = "cuda",
               attn_implementation: str = "sdpa"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype, attn_implementation=attn_implementation)
    model = model.to(device).eval()
    model.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    return model, tok


@dataclass
class Unit:
    """One eligible projection (the allocation unit for P3/Q3 is an individual projection)."""
    uid: str
    layer: int
    name: str
    module: nn.Linear

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.module.weight.shape)  # (d_out, d_in)

    @property
    def numel(self) -> int:
        return self.module.weight.numel()

    @property
    def weight(self) -> torch.Tensor:
        return self.module.weight


def get_layers(model) -> nn.ModuleList:
    return model.model.layers


def get_proj(layer: nn.Module, name: str) -> nn.Linear:
    if name in ATTN_PROJS:
        return getattr(layer.self_attn, name)
    if name in MLP_PROJS:
        return getattr(layer.mlp, name)
    raise KeyError(name)


def enumerate_units(model) -> list[Unit]:
    units = []
    for li, layer in enumerate(get_layers(model)):
        for name in PROJ_NAMES:
            mod = get_proj(layer, name)
            assert isinstance(mod, nn.Linear), (li, name, type(mod))
            units.append(Unit(unit_id(li, name), li, name, mod))
    return units


def eligible_param_ids(model) -> set[int]:
    return {id(u.module.weight) for u in enumerate_units(model)}


def unique_params(model) -> list[tuple[str, nn.Parameter]]:
    """All parameters, with tied/shared tensors listed (and later counted) exactly once."""
    seen, out = set(), []
    for n, p in model.named_parameters(remove_duplicate=False):
        if id(p) in seen:
            continue
        seen.add(id(p))
        out.append((n, p))
    return out


def classify_param(name: str, eligible_names: set[str]) -> str:
    if name in eligible_names:
        return "eligible_weight"
    if name.endswith(".bias"):
        return "bias"
    if "embed_tokens" in name:
        return "embedding"
    if "lm_head" in name:
        return "lm_head"
    if "norm" in name:
        return "norm"
    return "other"


def enumerate_report(model) -> dict:
    """Verifies the eligible set and reports counts (used by Step 1 and the unit tests)."""
    units = enumerate_units(model)
    n_layers = len(get_layers(model))
    assert len(units) == 7 * n_layers
    elig_names = {f"model.layers.{u.layer}.{'self_attn' if u.name in ATTN_PROJS else 'mlp'}.{u.name}.weight" for u in units}
    named = dict(model.named_parameters(remove_duplicate=False))
    assert elig_names <= set(named), elig_names - set(named)
    uniq = unique_params(model)
    by_class: dict[str, dict] = {}
    for n, p in uniq:
        c = classify_param(n, elig_names)
        d = by_class.setdefault(c, {"tensors": 0, "params": 0, "names_sample": []})
        d["tensors"] += 1
        d["params"] += p.numel()
        if len(d["names_sample"]) < 3:
            d["names_sample"].append(n)
    tied = None
    if hasattr(model, "lm_head") and hasattr(model.model, "embed_tokens"):
        tied = model.lm_head.weight is model.model.embed_tokens.weight
    total_unique = sum(p.numel() for _, p in uniq)
    total_listed = sum(p.numel() for _, p in model.named_parameters(remove_duplicate=False))
    return {
        "n_layers": n_layers,
        "n_units": len(units),
        "eligible_params": sum(u.numel for u in units),
        "unique_params_total": total_unique,
        "params_if_tied_double_counted": total_listed,
        "tied_lm_head_embedding": tied,
        "by_class": by_class,
        "eligible_shapes": {u.name: list(u.shape) for u in units[:7]},
    }


class ModelSnapshot:
    """Pristine copy of the unchanged weights; every compression condition restarts from it."""

    def __init__(self, model):
        self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.fingerprint = fingerprint(model)

    def restore(self, model):
        model.load_state_dict(self.state, strict=True)
        model.requires_grad_(False)
        fp = fingerprint(model)
        assert fp == self.fingerprint, "model was not restored to the pristine checkpoint"


def fingerprint(model) -> str:
    """Cheap exact-ish checksum of every unique parameter (float64 sums of values and squares)."""
    s = 0.0
    q = 0.0
    for _, p in unique_params(model):
        x = p.detach().double()
        s += float(x.sum())
        q += float((x * x).sum())
    return f"{s:.10e}|{q:.10e}"


def count_nonzero_eligible(units: list[Unit]) -> int:
    return int(sum(torch.count_nonzero(u.weight).item() for u in units))
