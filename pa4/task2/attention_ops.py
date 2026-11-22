"""Task 2 Part A: three exact causal attention operators on (B, H, L, d) tensors.

1. ``dense_causal_attention``   explicit s_ij = q_i k_j^T / sqrt(d) scores, causal mask, numerically
                                stable softmax (max-subtracted, computed in fp32), times V.
2. ``tiled_online_softmax_attention``  exact blockwise forward with running (m, l, o); only a
                                (block_q x block_k) score tile exists at any time (no L x L matrix).
                                IMPORTANT: this is a *Python-level* implementation. Its backward
                                is plain PyTorch autograd which retains every tile's intermediates,
                                so training memory is still O(L^2) -- it is NOT a fused FlashAttention
                                kernel (no recomputation in backward). ``TILED_BACKWARD_NOTE`` is
                                written into every benchmark record.
3. ``sdpa_attention``           torch.nn.functional.scaled_dot_product_attention, optionally under a
                                forced backend; the backend that is *actually dispatched* is detected
                                (``detect_sdpa_backend``) and recorded, never assumed.

All operators accept an optional boolean ``mask`` (True = may attend) broadcastable to (B, H, L, L),
combined with the causal mask. A query row with no allowed key produces an all-zero output (dense and
tiled agree); SDPA's behaviour for such rows is recorded by the edge-case tests, not assumed.
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F

TILED_BACKWARD_NOTE = ("tiled forward is exact online-softmax; backward = Python autograd through the tile loop "
                       "(retains per-tile scores/probabilities/accumulators; not a fused FlashAttention backward)")


def dense_causal_attention(q, k, v, mask=None):
    """q,k,v: (B,H,L,d). Scores are formed in the input dtype (as a plain implementation would), the
    softmax is evaluated in fp32 after subtracting the row max."""
    B, H, L, d = q.shape
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)
    allowed = torch.ones(L, L, dtype=torch.bool, device=q.device).tril()
    if mask is not None:
        allowed = allowed & mask
    scores = scores.float().masked_fill(~allowed, float("-inf"))
    m = scores.amax(dim=-1, keepdim=True)
    m = torch.where(torch.isinf(m), torch.zeros_like(m), m)           # fully masked rows
    p = torch.exp(scores - m)
    denom = p.sum(dim=-1, keepdim=True)
    probs = p / denom.clamp(min=1e-30)                                   # all-masked rows -> zeros
    return torch.matmul(probs.to(v.dtype), v)


def tiled_online_softmax_attention(q, k, v, block_q: int = 128, block_k: int = 128, mask=None, score_dtype=torch.float32):
    """Exact tiled causal attention with the online-softmax recurrence

        m' = max(m, max_j s_j)           l' = e^{m-m'} l + sum_j e^{s_j - m'}
        o' = e^{m-m'} o + sum_j e^{s_j - m'} V_j          A_i = o' / l'

    Accumulators and tile scores are fp32 (``score_dtype``; tiles are up-cast before the QK^T product, as fused
    kernels keep scores in fp32 registers), so fp16 inputs cannot overflow in the score matrix the way the plain
    dense operator can. Tiles strictly above the diagonal are skipped. Works for any L (ragged last tile).
    """
    B, H, L, d = q.shape
    scale = 1.0 / math.sqrt(d)
    out = []
    for qs in range(0, L, block_q):
        qe = min(qs + block_q, L)
        qt = q[:, :, qs:qe]
        nq = qe - qs
        m = torch.full((B, H, nq, 1), float("-inf"), device=q.device, dtype=torch.float32)
        l = torch.zeros((B, H, nq, 1), device=q.device, dtype=torch.float32)
        o = torch.zeros((B, H, nq, d), device=q.device, dtype=torch.float32)
        qpos = torch.arange(qs, qe, device=q.device)[:, None]
        for ks in range(0, qe, block_k):          # keys beyond the last query of the tile are never needed
            ke = min(ks + block_k, L)
            s = torch.matmul(qt.to(score_dtype), k[:, :, ks:ke].to(score_dtype).transpose(-1, -2)) * scale
            allowed = qpos >= torch.arange(ks, ke, device=q.device)[None, :]
            if mask is not None:
                allowed = allowed & mask[..., qs:qe, ks:ke]
            s = s.masked_fill(~allowed, float("-inf"))
            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
            m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
            p = torch.exp(s - m_safe)
            alpha = torch.exp(m - m_safe)                                    # 0 while m = -inf
            l = alpha * l + p.sum(dim=-1, keepdim=True)
            o = alpha * o + torch.matmul(p.to(v.dtype), v[:, :, ks:ke]).float()
            m = m_new
        out.append((o / l.clamp(min=1e-30)).to(v.dtype))
    return torch.cat(out, dim=2)


def sdpa_attention(q, k, v, mask=None, backend: str | None = None):
    """SDPA with causal masking. ``backend`` in {None, 'flash', 'efficient', 'math', 'cudnn'} forces a kernel
    (error if unsupported for these inputs); None lets PyTorch dispatch."""
    with force_sdpa_backend(backend):
        if mask is None:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        L = q.shape[-2]
        allowed = torch.ones(L, L, dtype=torch.bool, device=q.device).tril() & mask
        return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)


@contextmanager
def force_sdpa_backend(backend: str | None):
    if backend is None:
        yield
        return
    from torch.nn.attention import SDPBackend, sdpa_kernel

    table = {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION, "math": SDPBackend.MATH,
             "cudnn": getattr(SDPBackend, "CUDNN_ATTENTION", None)}
    sel = table[backend]
    if sel is None:
        raise RuntimeError(f"backend {backend} unavailable in this torch build")
    with sdpa_kernel([sel]):
        yield


def detect_sdpa_backend(q, k, v, mask=None) -> dict:
    """Identify the kernel PyTorch dispatches for these inputs by inspecting profiler kernel names
    (CUDA) / operator names (CPU). Returns {'dispatched': name, 'evidence': [...]}."""
    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if q.is_cuda else [])
    with torch.no_grad(), profile(activities=acts) as prof:
        sdpa_attention(q, k, v, mask)
        if q.is_cuda:
            torch.cuda.synchronize()
    names = [e.key for e in prof.key_averages()]
    low = " ".join(names).lower()
    if "flash" in low and ("fwd" in low or "flash_attention" in low or "flash_fwd" in low):
        d = "flash"
    elif "efficient_attention" in low or "mem_eff" in low or "fmha" in low or "attention_forward_kernel" in low:
        d = "efficient"
    elif "cudnn" in low and "attention" in low:
        d = "cudnn"
    elif "_scaled_dot_product_flash_attention_for_cpu" in low or "flash_attention_for_cpu" in low:
        d = "cpu_flash"
    elif "softmax" in low or "bmm" in low or "matmul" in low:
        d = "math"
    else:
        d = "unknown"
    ev = [n for n in names if any(t in n.lower() for t in ("attention", "flash", "fmha", "softmax", "scaled_dot"))][:8]
    return {"dispatched": d, "evidence": ev}


def supported_backends(q, k, v) -> dict:
    """Try forcing each backend on these inputs; record supported / error."""
    res = {}
    for b in ("flash", "efficient", "math", "cudnn"):
        try:
            with torch.no_grad():
                sdpa_attention(q, k, v, backend=b)
            res[b] = {"supported": True}
        except Exception as e:  # noqa: BLE001
            res[b] = {"supported": False, "error": str(e).splitlines()[0][:160]}
    return res


OPERATORS = {
    "dense": dense_causal_attention,
    "tiled": tiled_online_softmax_attention,
    "sdpa": sdpa_attention,
}
