"""Incremental language-model wrappers with explicit KV cache rollback (shared interface for real models and the toy).

Interface used by ``specdec``:  ``len(lm)`` (tokens in cache), ``lm.tokens`` (their ids), ``lm.feed(ids) -> logits (len(ids), V)``
(float32, row i = distribution of the token *after* ids[i]), ``lm.truncate(n)``, ``lm.reset()``.
Cache positions are always the absolute positions 0..n-1 of ``lm.tokens``, so after a rollback the next fed token gets
position ``n`` exactly as in a fresh forward pass.
"""
from __future__ import annotations

import torch

from ..task2.kvcache.engine import KVEngine


class CachedLM:
    def __init__(self, model, name: str = "lm"):
        self.engine = KVEngine(model)
        self.model, self.name = model, name
        self.device = self.engine.device
        self.reset()

    def reset(self):
        self.cache = self.engine.empty_cache(1)
        self.tokens: list[int] = []

    def __len__(self):
        return len(self.tokens)

    @torch.no_grad()
    def feed(self, ids: list[int]) -> torch.Tensor:
        n0 = len(self.tokens)
        t = torch.tensor([ids], device=self.device)
        pos = torch.arange(n0, n0 + len(ids), device=self.device)[None]
        logits = self.engine.forward(t, pos, self.cache, update=True, logits_last=len(ids))[0].float()
        self.tokens.extend(ids)
        return logits

    def truncate(self, n: int):
        if n < len(self.tokens):
            self.cache.truncate(n)
            del self.tokens[n:]
        assert self.cache.n == len(self.tokens)

    @torch.no_grad()
    def full_logits(self, ids: list[int]) -> torch.Tensor:
        """Reference: one fresh forward over ``ids`` (no cache); row i = distribution after ids[i]."""
        t = torch.tensor([ids], device=self.device)
        pos = torch.arange(len(ids), device=self.device)[None]
        return self.engine.forward(t, pos, None, update=False, logits_last=len(ids))[0].float()


class TableLM:
    """First-order Markov 'language model': ``trans`` has V+1 rows (row V is the BOS token and holds the initial
    distribution) and V columns (next symbol in 0..V-1). Conditioning is on the last token only. Same interface as
    ``CachedLM`` (``feed`` returns log-probabilities, i.e. logits at temperature 1)."""

    def __init__(self, trans: torch.Tensor, name="table"):
        self.trans, self.name, self.device = trans.double(), name, trans.device
        self.reset()

    def reset(self):
        self.tokens: list[int] = []

    def __len__(self):
        return len(self.tokens)

    def feed(self, ids):
        self.tokens.extend(ids)
        return torch.log(self.trans[torch.tensor(ids)])

    def truncate(self, n):
        del self.tokens[n:]
