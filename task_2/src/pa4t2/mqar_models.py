"""Task 2 Part B: the small decoder-only Transformer and its attention variants.

Fixed architecture (identical across variants -- only the attention *operator* changes):
  2 pre-LN blocks, d=128, 4 heads x 32, FFN 4x (GELU), learned token embeddings, fixed sinusoidal positions,
  256-token vocabulary, untied output head, bias-free q/k/v/o projections.

Variants (``attn``):
  dense      causal softmax attention
  window     causal softmax attention restricted to the previous W positions (W=64; includes the current token)
  linear     phi(x)=ELU(x)+1 causal linear attention  a_t = phi(q_t)^T S_t / (phi(q_t)^T z_t + eps)
  sparse     heads 0,1: local window (W=64); heads 2,3: 64 keys at causal offsets {0,4,...,252}
  linear_l2  B3 control: additive linear attention with L2-normalised phi(q), phi(k), read-out S_t phi(q_t) (no z)
  delta      B3 intervention: same as linear_l2 but the write is the delta rule
                 S_t = S_{t-1} + beta_t (v_t - S_{t-1} phi(k_t)) phi(k_t)^T   (beta_t = sigmoid(w^T x_t), +H params)

Two implementations of the structured variants exist and are checked to agree in the tests: the *reference* (mask /
parallel form, used for training at T=256) and the *efficient* kernels used by the inference sweep (chunked
banded window, gathered dilated keys, cumulative-sum linear attention).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6
VARIANTS = ("dense", "window", "linear", "sparse", "linear_l2", "delta")


@dataclass
class MQARConfig:
    attn: str = "dense"
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    vocab: int = 256
    mlp_mult: int = 4
    window: int = 64
    dilation: int = 4
    n_dilated_keys: int = 64
    max_len: int = 8192

    @property
    def d_head(self):
        return self.d_model // self.n_heads


def sinusoidal(max_len: int, d: int) -> torch.Tensor:
    pos = torch.arange(max_len)[:, None].float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(max_len, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def phi_elu(x):
    return F.elu(x) + 1.0


def head_masks(cfg: MQARConfig, T: int, device, variant: str | None = None, window: int | None = None) -> torch.Tensor | None:
    """(H, T, T) boolean allow-masks for softmax-style variants; None = plain causal."""
    variant = variant or cfg.attn
    W = window or cfg.window
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    causal = j <= i
    delta = i - j
    local = causal & (delta < W)
    if variant == "dense":
        return None
    if variant == "window":
        return local.expand(cfg.n_heads, T, T)
    if variant == "sparse":
        dil = causal & (delta % cfg.dilation == 0) & (delta // cfg.dilation < cfg.n_dilated_keys)
        masks = [local, local, dil, dil]
        return torch.stack(masks[: cfg.n_heads])
    raise ValueError(variant)


def linear_attention_parallel(q, k, v, normalised: bool = True, l2: bool = False):
    """Parallel (masked matrix) form of causal linear attention; q,k,v (B,H,T,d)."""
    fq, fk = phi_elu(q), phi_elu(k)
    if l2:
        fq, fk = F.normalize(fq, dim=-1), F.normalize(fk, dim=-1)
    T = q.shape[2]
    A = torch.matmul(fq, fk.transpose(-1, -2)) * torch.ones(T, T, device=q.device).tril()
    out = torch.matmul(A, v)
    if normalised:
        out = out / (A.sum(-1, keepdim=True) + EPS)
    return out


def linear_attention_recurrent(q, k, v, normalised: bool = True, l2: bool = False):
    """Explicit recurrence S_t = S_{t-1} + phi(k_t) v_t^T, z_t = z_{t-1} + phi(k_t) (reference for tests)."""
    fq, fk = phi_elu(q), phi_elu(k)
    if l2:
        fq, fk = F.normalize(fq, dim=-1), F.normalize(fk, dim=-1)
    B, H, T, d = q.shape
    S = torch.zeros(B, H, d, d, device=q.device, dtype=q.dtype)
    z = torch.zeros(B, H, d, device=q.device, dtype=q.dtype)
    outs = []
    for t in range(T):
        S = S + fk[:, :, t, :, None] * v[:, :, t, None, :]
        z = z + fk[:, :, t]
        o = torch.einsum("bhd,bhde->bhe", fq[:, :, t], S)
        if normalised:
            o = o / ((fq[:, :, t] * z).sum(-1, keepdim=True) + EPS)
        outs.append(o)
    return torch.stack(outs, 2)


def delta_rule_attention(q, k, v, beta):
    """S_t = S_{t-1} + beta_t (v_t - S_{t-1} k_t) k_t^T ; o_t = S_t q_t with L2-normalised phi(q), phi(k)."""
    fq, fk = F.normalize(phi_elu(q), dim=-1), F.normalize(phi_elu(k), dim=-1)
    B, H, T, d = q.shape
    S = torch.zeros(B, H, d, d, device=q.device, dtype=q.dtype)        # (value dim, key dim)
    outs = []
    for t in range(T):
        kt, vt, bt = fk[:, :, t], v[:, :, t], beta[:, :, t, None]
        pred = torch.einsum("bhef,bhf->bhe", S, kt)
        S = S + torch.einsum("bhe,bhf->bhef", bt * (vt - pred), kt)
        outs.append(torch.einsum("bhef,bhf->bhe", S, fq[:, :, t]))
    return torch.stack(outs, 2)


# ----------------------------------------------------------------------------- efficient inference kernels

def window_attention_chunked(q, k, v, W: int):
    """Causal banded attention (keys i-W+1..i) in O(T*W): each chunk of W queries sees [prev chunk, own chunk]."""
    B, H, T, d = q.shape
    pad = (-T) % W
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    n = q.shape[2] // W
    qc = q.view(B, H, n, W, d)
    kc, vc = k.view(B, H, n, W, d), v.view(B, H, n, W, d)
    kp = torch.cat([torch.zeros_like(kc[:, :, :1]), kc[:, :, :-1]], 2)
    vp = torch.cat([torch.zeros_like(vc[:, :, :1]), vc[:, :, :-1]], 2)
    kk, vv = torch.cat([kp, kc], 3), torch.cat([vp, vc], 3)                # (B,H,n,2W,d)
    s = torch.matmul(qc, kk.transpose(-1, -2)) / math.sqrt(d)               # (B,H,n,W,2W)
    qi = torch.arange(W, device=q.device)[:, None] + W                      # position inside [prev|cur]
    kj = torch.arange(2 * W, device=q.device)[None, :]
    allowed = (kj <= qi) & (qi - kj < W)
    first = torch.zeros(n, 1, 2 * W, dtype=torch.bool, device=q.device)
    first[0, :, :W] = True                                                  # no previous chunk for chunk 0
    allowed = allowed[None] & ~first
    s = s.masked_fill(~allowed[None, None], float("-inf"))
    p = torch.softmax(s.float() if s.dtype in (torch.float16, torch.bfloat16) else s, -1).to(v.dtype)
    out = torch.matmul(p, vv).view(B, H, n * W, d)
    return out[:, :, :T]


def dilated_attention_gather(q, k, v, dilation: int, n_keys: int):
    """Each query attends to keys at causal offsets 0, dilation, ..., (n_keys-1)*dilation (gathered, O(T*n_keys))."""
    B, H, T, d = q.shape
    off = torch.arange(n_keys, device=q.device) * dilation
    idx = torch.arange(T, device=q.device)[:, None] - off[None, :]           # (T, n_keys)
    valid = idx >= 0
    idx = idx.clamp(min=0)
    kg, vg = k[:, :, idx], v[:, :, idx]                                       # (B,H,T,n_keys,d)
    s = torch.einsum("bhtd,bhtnd->bhtn", q, kg) / math.sqrt(d)
    s = s.masked_fill(~valid, float("-inf"))
    p = torch.softmax(s.float() if s.dtype in (torch.float16, torch.bfloat16) else s, -1).to(v.dtype)
    return torch.einsum("bhtn,bhtnd->bhtd", p, vg)


def linear_attention_cumsum(q, k, v, normalised: bool = True, l2: bool = False):
    fq, fk = phi_elu(q), phi_elu(k)
    if l2:
        fq, fk = F.normalize(fq, dim=-1), F.normalize(fk, dim=-1)
    S = torch.cumsum(fk[..., :, None] * v[..., None, :], dim=2)               # (B,H,T,dk,dv)
    out = torch.einsum("bhtd,bhtde->bhte", fq, S)
    if normalised:
        z = torch.cumsum(fk, dim=2)
        out = out / ((fq * z).sum(-1, keepdim=True) + EPS)
    return out


# ----------------------------------------------------------------------------- model

class Attention(nn.Module):
    def __init__(self, cfg: MQARConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.beta = nn.Linear(d, cfg.n_heads) if cfg.attn == "delta" else None

    def forward(self, x, variant: str | None = None, window: int | None = None, efficient: bool = False):
        cfg = self.cfg
        B, T, D = x.shape
        H, dh = cfg.n_heads, cfg.d_head
        q, k, v = self.qkv(x).view(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        variant = variant or cfg.attn
        W = window or cfg.window
        if variant == "dense":
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif variant in ("window", "sparse"):
            if efficient:
                if variant == "window":
                    o = window_attention_chunked(q, k, v, W)
                else:
                    loc = window_attention_chunked(q[:, :2], k[:, :2], v[:, :2], W)
                    dil = dilated_attention_gather(q[:, 2:], k[:, 2:], v[:, 2:], cfg.dilation, cfg.n_dilated_keys)
                    o = torch.cat([loc, dil], 1)
            else:
                m = head_masks(cfg, T, x.device, variant, W)
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=m[None])
        elif variant == "linear":
            o = (linear_attention_cumsum if efficient else linear_attention_parallel)(q, k, v)
        elif variant == "linear_l2":
            o = (linear_attention_cumsum if efficient else linear_attention_parallel)(q, k, v, normalised=False, l2=True)
        elif variant == "delta":
            beta = torch.sigmoid(self.beta(x)).transpose(1, 2)                  # (B,H,T)
            o = delta_rule_attention(q, k, v, beta)
        else:
            raise ValueError(variant)
        return self.out(o.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.mlp_mult * cfg.d_model), nn.GELU(), nn.Linear(cfg.mlp_mult * cfg.d_model, cfg.d_model))

    def forward(self, x, **kw):
        x = x + self.attn(self.ln1(x), **kw)
        return x + self.mlp(self.ln2(x))


class MQARTransformer(nn.Module):
    def __init__(self, cfg: MQARConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.register_buffer("pos", sinusoidal(cfg.max_len, cfg.d_model), persistent=False)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, tokens=None, inputs_embeds=None, **kw):
        x = self.emb(tokens) if inputs_embeds is None else inputs_embeds
        x = x + self.pos[: x.shape[1]].to(x.dtype)
        for b in self.blocks:
            x = b(x, **kw)
        return self.head(self.ln_f(x))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------- theoretical accounting

def accounting(cfg: MQARConfig, T: int = 256, bytes_per_elem: int = 4) -> dict:
    """Theoretical attention pair counts and *persistent inference state* per layer at length T."""
    H, dh = cfg.n_heads, cfg.d_head
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    causal = j <= i
    delta = i - j
    local = (causal & (delta < cfg.window)).sum().item()
    dil = (causal & (delta % cfg.dilation == 0) & (delta // cfg.dilation < cfg.n_dilated_keys)).sum().item()
    dense = causal.sum().item()
    a = cfg.attn
    if a == "dense":
        pairs, state = [dense] * H, [2 * T * dh] * H
    elif a == "window":
        pairs, state = [local] * H, [2 * min(T, cfg.window) * dh] * H
    elif a == "sparse":
        pairs = [local, local, dil, dil][:H]
        reach = (cfg.n_dilated_keys - 1) * cfg.dilation + 1
        state = [2 * min(T, cfg.window) * dh] * 2 + [2 * min(T, reach) * dh] * 2
    elif a == "linear":
        pairs, state = [0] * H, [dh * dh + dh] * H           # S (dh x dh) + z (dh)
    else:                                                      # linear_l2, delta: S only
        pairs, state = [0] * H, [dh * dh] * H
    model = MQARTransformer(cfg)
    return {"attn": a, "T": T, "trainable_params": model.n_params(), "pairs_per_head": pairs, "pairs_per_layer": sum(pairs),
            "pairs_fraction_of_dense": sum(pairs) / (dense * H) if dense else None,
            "state_elements_per_layer": sum(state), "state_bytes_per_layer_fp32": sum(state) * 4, "state_bytes_per_layer_fp16": sum(state) * 2,
            "note": "state = elements kept across steps by an incremental decoder (KV cache for softmax variants, (S,z) for linear)"}
