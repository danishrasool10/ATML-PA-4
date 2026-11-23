"""Explicit-cache Qwen2 forward pass (shared by task_2 Part C and task_3 speculative decoding).

The engine re-implements the decoder forward on top of the HF module weights (``q_proj`` ... ``mlp``, RoPE from the
model's own ``rotary_emb``) so that
  * the cache is a plain per-layer tensor pair we can evict / quantize / reorder,
  * every token carries its original absolute rotary position id regardless of its physical cache slot,
  * attention is PyTorch SDPA (the working model path), except in the *scoring pass*, which additionally computes the
    FP32 attention probabilities of the question suffix against the cached keys (``capture``),
  * C1 validates this engine against the unmodified HF forward / ``generate``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F



@dataclass
class QuantSpec:
    bits: int
    k_layout: str            # 'channel' | 'token'
    v_layout: str = "token"
    group: int = 64
    fake_quantize: Callable | None = None     # (x, bits, layout) -> dequantized x; supplied by the KV-cache study (task_2)


class LayerCache:
    """K, V: (B, H_kv, n, d) post-RoPE FP16 (dequantized values if fake-quantized). Tokens ``[0, n_quant)`` are
    quantized; the rest are FP16. When ``quantize_new`` a group is quantized as soon as 64 aligned slots are full."""

    def __init__(self, K, V, spec: QuantSpec | None = None, n_quant: int = 0, quantize_new: bool = False):
        self.K, self.V, self.spec, self.n_quant, self.quantize_new = K, V, spec, n_quant, quantize_new

    @property
    def n(self) -> int:
        return self.K.shape[-2]

    def append(self, k, v):
        self.K = torch.cat([self.K, k], dim=-2)
        self.V = torch.cat([self.V, v], dim=-2)
        if self.spec is not None and self.quantize_new:
            g, fq = self.spec.group, self.spec.fake_quantize
            while self.n_quant + g <= self.n:
                s, e = self.n_quant, self.n_quant + g
                self.K[..., s:e, :] = fq(self.K[..., s:e, :], self.spec.bits, self.spec.k_layout)
                self.V[..., s:e, :] = fq(self.V[..., s:e, :], self.spec.bits, self.spec.v_layout)
                self.n_quant = e

    def clone(self):
        return LayerCache(self.K.clone(), self.V.clone(), self.spec, self.n_quant, self.quantize_new)

    def truncate(self, n: int):
        assert 0 <= n <= self.n
        if n < self.n:
            self.K, self.V = self.K[..., :n, :], self.V[..., :n, :]
            self.n_quant = min(self.n_quant, n)


class KVCache:
    def __init__(self, layers: list[LayerCache], meta: dict | None = None):
        self.layers, self.meta = layers, meta or {}

    @property
    def n(self) -> int:
        return self.layers[0].n

    def clone(self):
        return KVCache([l.clone() for l in self.layers], dict(self.meta))

    def truncate(self, n: int):
        for l in self.layers:
            l.truncate(n)

    def expand_batch(self, B: int):
        return KVCache([LayerCache(l.K.expand(B, -1, -1, -1).contiguous(), l.V.expand(B, -1, -1, -1).contiguous(), l.spec, l.n_quant,
                                   l.quantize_new) for l in self.layers], dict(self.meta))

    def fp16_nbytes(self) -> int:
        return sum(l.K.numel() + l.V.numel() for l in self.layers) * 2


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


class KVEngine:
    def __init__(self, model):
        self.model = model
        c = model.config
        self.n_layers = c.num_hidden_layers
        self.Hq = c.num_attention_heads
        self.Hkv = c.num_key_value_heads
        self.d = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def forward(self, ids, pos, cache: KVCache | None = None, update: bool = True, logits_last: int = 1, capture: dict | None = None,
                return_hidden: bool = False):
        """ids, pos: (B, s) token ids and *absolute* rotary position ids. New tokens attend to every cached token and
        causally to each other. ``capture``: dict; filled with ``attn_to_cache`` (s-averaged FP32 attention mass on each
        cached key, averaged over layers and heads, shape (B, n_cache)) when given (cache must be non-empty)."""
        m = self.model.model
        B, s = ids.shape
        h = m.embed_tokens(ids)
        cos, sin = m.rotary_emb(h, pos)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        n_c = cache.n if cache is not None else 0
        mask = None
        if n_c > 0 and s > 1:
            col = torch.arange(n_c + s, device=ids.device)[None, :]
            row = torch.arange(s, device=ids.device)[:, None] + n_c
            mask = col <= row
        score_acc = None
        for li, layer in enumerate(m.layers):
            a = layer.self_attn
            x = layer.input_layernorm(h)
            q = a.q_proj(x).view(B, s, self.Hq, self.d).transpose(1, 2)
            k = a.k_proj(x).view(B, s, self.Hkv, self.d).transpose(1, 2)
            v = a.v_proj(x).view(B, s, self.Hkv, self.d).transpose(1, 2)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            if cache is not None:
                lc = cache.layers[li]
                K = torch.cat([lc.K, k], dim=-2) if n_c > 0 else k
                V = torch.cat([lc.V, v], dim=-2) if n_c > 0 else v
            else:
                K, V = k, v
            if n_c == 0:
                o = F.scaled_dot_product_attention(q, K, V, is_causal=s > 1, enable_gqa=True)
            elif s == 1:
                o = F.scaled_dot_product_attention(q, K, V, enable_gqa=True)
            else:
                o = F.scaled_dot_product_attention(q, K, V, attn_mask=mask, enable_gqa=True)
            if capture is not None:
                assert n_c > 0
                rep = self.Hq // self.Hkv
                Kf = K.float().repeat_interleave(rep, dim=1)
                sc = torch.matmul(q.float(), Kf.transpose(-1, -2)) / (self.d ** 0.5)          # (B, Hq, s, n_c + s)
                if mask is not None:
                    sc = sc.masked_fill(~mask, float("-inf"))
                p = torch.softmax(sc, dim=-1)[..., :n_c]                                       # mass on the cached keys
                contrib = p.mean(dim=(1, 2))                                                   # mean over heads, suffix queries
                score_acc = contrib if score_acc is None else score_acc + contrib
            o = o.transpose(1, 2).reshape(B, s, self.Hq * self.d)
            h = h + a.o_proj(o)
            h = h + layer.mlp(layer.post_attention_layernorm(h))
            if cache is not None and update:
                lc.append(k, v)
        if capture is not None:
            capture["attn_to_cache"] = score_acc / self.n_layers
        h = m.norm(h[:, -logits_last:])
        logits = self.model.lm_head(h)
        return (logits, h) if return_hidden else logits

    # ------------------------------------------------------------------ cache construction
    def empty_cache(self, B: int = 1) -> KVCache:
        z = lambda: torch.empty(B, self.Hkv, 0, self.d, device=self.device, dtype=self.dtype)
        return KVCache([LayerCache(z(), z()) for _ in range(self.n_layers)])

    @torch.no_grad()
    def prefill(self, ids, chunk: int = 2048):
        """Full-precision prefill of the pre-question prefix; returns (logits of the last token, KVCache)."""
        cache = self.empty_cache(ids.shape[0])
        logits = None
        for s in range(0, ids.shape[1], chunk):
            part = ids[:, s:s + chunk]
            pos = torch.arange(s, s + part.shape[1], device=ids.device)[None].expand(ids.shape[0], -1)
            logits = self.forward(part, pos, cache, update=True)
        return logits, cache

    @torch.no_grad()
    def score_suffix(self, suffix_ids, cache: KVCache, start_pos: int):
        """C2(ii)/C4 scoring pass: FP32 attention of the question suffix on the (unchanged) FP16 prefix cache.
        Does not modify the cache. Returns (n_cache,) scores (shared over layers and heads)."""
        cap: dict = {}
        pos = torch.arange(start_pos, start_pos + suffix_ids.shape[1], device=suffix_ids.device)[None]
        self.forward(suffix_ids, pos, cache, update=False, capture=cap)
        return cap["attn_to_cache"][0]

    @torch.no_grad()
    def run_suffix_and_generate(self, suffix_ids, cache: KVCache, start_pos: int, max_new: int, eos_ids: set[int]):
        """Feed the question suffix on (a clone of) ``cache`` and greedily decode. Returns (generated ids, first logits)."""
        pos = torch.arange(start_pos, start_pos + suffix_ids.shape[1], device=suffix_ids.device)[None]
        logits = self.forward(suffix_ids, pos, cache, update=True)
        first = logits[0, -1].float()
        out, nxt, p = [], logits[0, -1].argmax().item(), start_pos + suffix_ids.shape[1]
        for _ in range(max_new):
            out.append(nxt)
            if nxt in eos_ids:
                break
            lg = self.forward(torch.tensor([[nxt]], device=suffix_ids.device), torch.tensor([[p]], device=suffix_ids.device), cache, update=True)
            nxt, p = lg[0, -1].argmax().item(), p + 1
        return out, first

    @torch.no_grad()
    def teacher_forced(self, suffix_ids, answer_ids, cache: KVCache, start_pos: int):
        """Suffix + gold answer tokens in one pass; returns logits (len(answer)+1, V): row 0 is the post-prefill
        distribution (predicts answer[0]), row i predicts answer[i] (row len(answer) predicts the token after it)."""
        full = torch.cat([suffix_ids, answer_ids], dim=1)
        pos = torch.arange(start_pos, start_pos + full.shape[1], device=full.device)[None]
        lg = self.forward(full, pos, cache, update=True, logits_last=answer_ids.shape[1] + 1)
        return lg[0].float()
