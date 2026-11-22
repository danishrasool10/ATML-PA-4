"""Layer-wise calibration driver.

All calibration-time statistics are collected by running the *dense, unmodified* transformer
blocks one at a time over the 128 calibration sequences and propagating the dense hidden states
(the same dense inputs are used for Wanda norms, AWQ, sensitivity probes and every error metric,
independent of the compression applied to earlier blocks).

For every input tensor X of the projection groups (qkv / o / gate+up / down) we accumulate
    G = X^T X          (fp32 Gram matrix, d_in x d_in)
    sum|x|             (for AWQ's mean activation magnitude)
so that, for any weight perturbation dW, the exact layer-output error on the calibration inputs is
    ||X dW^T||_F^2 = tr(dW G dW^T)
without ever storing X. ``diag(G)`` = ||X_{:,j}||_2^2 are the Wanda activation norms.
"""
from __future__ import annotations

from typing import Callable

import torch

from .constants import GROUPS, INPUT_GROUP, PROJ_NAMES
from .model_utils import get_layers, get_proj


class _Stop(Exception):
    pass


class _Catcher(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.attention_type = getattr(layer, "attention_type", "full_attention")
        self.captured: list = []

    def forward(self, hidden_states, *args, **kwargs):
        self.captured.append((hidden_states.detach(), args, kwargs))
        raise _Stop


@torch.no_grad()
def capture_first_layer_inputs(model, batches: list[torch.Tensor]):
    layers = get_layers(model)
    orig = layers[0]
    catcher = _Catcher(orig)
    layers[0] = catcher
    dev = next(model.parameters()).device
    try:
        for b in batches:
            try:
                model(input_ids=b.to(dev), use_cache=False)
            except _Stop:
                pass
    finally:
        layers[0] = orig
    return catcher.captured


def make_batches(seqs: list[list[int]], batch_size: int) -> list[torch.Tensor]:
    return [torch.tensor(seqs[i : i + batch_size], dtype=torch.long) for i in range(0, len(seqs), batch_size)]


def layer_forward(layer, h, args, kwargs):
    out = layer(h, *args, **kwargs)
    return out[0] if isinstance(out, tuple) else out


class LayerStats:
    """Accumulates Gram matrices / mean |x| of the inputs of one block's four projection groups."""

    def __init__(self, layer):
        self.layer = layer
        self.G: dict[str, torch.Tensor] = {}
        self.absum: dict[str, torch.Tensor] = {}
        self.n_tokens = 0
        self._handles = []
        self._modules = {"qkv": get_proj(layer, "q_proj"), "o": get_proj(layer, "o_proj"),
                         "gu": get_proj(layer, "gate_proj"), "down": get_proj(layer, "down_proj")}
        for g, mod in self._modules.items():
            self._handles.append(mod.register_forward_pre_hook(self._make_hook(g)))

    def _make_hook(self, group):
        def hook(module, args):
            x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
            if group not in self.G:
                self.G[group] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=x.device)
                self.absum[group] = torch.zeros(x.shape[1], dtype=torch.float32, device=x.device)
            self.G[group].addmm_(x.T, x)
            self.absum[group] += x.abs().sum(0)
            if group == "qkv":
                self.n_tokens += x.shape[0]
        return hook

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    # --- accessors by projection name ---
    def gram(self, proj: str) -> torch.Tensor:
        return self.G[INPUT_GROUP[proj]]

    def act_norm(self, proj: str) -> torch.Tensor:
        """||X_{:,j}||_2 (Wanda)."""
        return self.gram(proj).diagonal().clamp(min=0).sqrt()

    def abs_mean(self, group_or_proj: str) -> torch.Tensor:
        g = INPUT_GROUP.get(group_or_proj, group_or_proj)
        return self.absum[g] / self.n_tokens


def quad_error(dW: torch.Tensor, G: torch.Tensor, chunk_rows: int = 4096) -> float:
    """tr(dW G dW^T) = || X dW^T ||_F^2 (float64 accumulation of fp32 row chunks)."""
    total = 0.0
    for s in range(0, dW.shape[0], chunk_rows):
        d = dW[s : s + chunk_rows].float()
        total += float(((d @ G) * d).sum(dtype=torch.float64))
    return total


def run_layerwise(model, batches: list[torch.Tensor], layer_fn: Callable, *, first_batch_ctx: bool = True) -> dict:
    """Run every block densely over ``batches``; after block ``li`` has produced its dense outputs
    (and its input statistics are complete) call ``layer_fn(li, layer, stats, ctx)``. ``layer_fn``
    may modify the block's weights in place -- subsequent blocks still receive the dense hidden
    states because the outputs were computed before the modification.
    """
    layers = get_layers(model)
    cur = capture_first_layer_inputs(model, batches)
    n_tok = sum(int(b.numel()) for b in batches)
    for li, layer in enumerate(layers):
        stats = LayerStats(layer)
        nxt = []
        with torch.no_grad():
            for (h, args, kw) in cur:
                nxt.append((layer_forward(layer, h, args, kw), args, kw))
        stats.close()
        assert stats.n_tokens == n_tok, (stats.n_tokens, n_tok)
        ctx = {"sample": cur[0], "n_tokens": n_tok} if first_batch_ctx else {"n_tokens": n_tok}
        layer_fn(li, layer, stats, ctx)
        cur = nxt
        del stats
    return {"n_tokens": n_tok, "n_layers": len(layers)}
