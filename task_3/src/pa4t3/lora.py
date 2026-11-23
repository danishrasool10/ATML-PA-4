"""Hand-rolled LoRA (no ``peft`` dependency) for q/k/v/o projections, with exact merging.

Adapters are FP32 parameters ``A (r, in)``, ``B (out, r)``; the frozen backbone stays in its own dtype (FP16 on GPU).
``y = W x + b + (alpha / r) * B A x``. ``B`` starts at zero, ``A`` follows peft's Kaiming-uniform(a = sqrt 5).
Merging computes ``W' = W + (alpha / r) B A`` in FP32 and rounds once to the backbone dtype.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base, self.r, self.scaling, self.enabled = base, r, alpha / r, True
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        d = F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B) * self.scaling
        return y + d.to(y.dtype)

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B.float() @ self.lora_A.float()) * self.scaling


def inject_lora(model: nn.Module, r: int = 8, alpha: float = 16.0, targets=TARGETS) -> dict[str, LoRALinear]:
    """Freeze ``model`` and replace the attention projections by LoRALinear modules; returns {name: module}."""
    model.requires_grad_(False)
    out = {}
    for lname, layer in enumerate(model.model.layers):
        for t in targets:
            base = getattr(layer.self_attn, t)
            mod = LoRALinear(base, r, alpha)
            setattr(layer.self_attn, t, mod)
            out[f"model.layers.{lname}.self_attn.{t}"] = mod
    return out


def lora_parameters(model: nn.Module):
    return [p for n, p in model.named_parameters() if "lora_" in n]


def lora_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if "lora_" in n}


def load_lora_state(model: nn.Module, state: dict[str, torch.Tensor]):
    params = dict(model.named_parameters())
    for n, v in state.items():
        params[n].data.copy_(v.to(params[n].device))


def set_lora_enabled(model: nn.Module, enabled: bool):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.enabled = enabled


@torch.no_grad()
def merge_into(fresh_model: nn.Module, state: dict[str, torch.Tensor], r: int = 8, alpha: float = 16.0) -> nn.Module:
    """Add the adapter delta to the matching projections of a *fresh* (adapter-free) backbone, in place; returns it."""
    scaling = alpha / r
    mods = dict(fresh_model.named_modules())
    for n in state:
        if not n.endswith("lora_A"):
            continue
        prefix = n[: -len(".lora_A")]
        # state names look like model.layers.0.self_attn.q_proj.lora_A (possibly with a base.-less wrapper prefix)
        lin = mods[prefix]
        A, B = state[n].float().to(lin.weight.device), state[prefix + ".lora_B"].float().to(lin.weight.device)
        lin.weight.copy_((lin.weight.float() + scaling * (B @ A)).to(lin.weight.dtype))
    return fresh_model
