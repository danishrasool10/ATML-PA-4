"""Latency / memory profiling for dense FP16 execution.

Prefill: forward of a fixed random prompt of 512 / 2048 tokens (batch 1), no cache returned.
Decode : 64 greedy new tokens (KV cache) after a 512-token prefill; the reported decode time is
         the time of the 64-step loop only (prefill excluded) plus the per-token mean.
All timings: >= 5 warmed-up repetitions, CUDA synchronised, unit = ms, fixed input shapes.
Every condition is *executed as dense FP16* (zeros / de-quantized values); no sparse or low-bit
kernels are used, so measured differences between conditions are not kernel speed-ups.
"""
from __future__ import annotations

import torch

from pa4common.seed import make_generator
from pa4common.timing import peak_memory, summarize_ms, sync, time_callable
import time


def _prompt(vocab: int, n: int, device, seed: int = 6304, lo: int = 3) -> torch.Tensor:
    g = make_generator(seed)
    return torch.randint(lo, vocab, (1, n), generator=g).to(device)


@torch.no_grad()
def profile_model(model, *, warmup: int = 2, reps: int = 5, prefill_lengths=(512, 2048), decode_prompt_len: int = 512,
                  decode_new_tokens: int = 64) -> dict:
    device = next(model.parameters()).device
    vocab = model.config.vocab_size
    out: dict = {"execution": "dense_fp16_tensors (no sparse/low-bit kernels)", "device": str(device),
                 "dtype": str(next(model.parameters()).dtype), "unit": "ms", "reps": reps, "warmup": warmup,
                 "prefill": {}, "decode": {}}
    with peak_memory(device) as mem_all:
        for L in prefill_lengths:
            ids = _prompt(vocab, L, device)
            with peak_memory(device) as m:
                t = time_callable(lambda: model(input_ids=ids, use_cache=False), device=device, warmup=warmup, reps=reps)
            t["peak_allocated_bytes"] = m["peak_allocated_bytes"]
            out["prefill"][str(L)] = t
        ids = _prompt(vocab, decode_prompt_len, device)

        def decode_once():
            o = model(input_ids=ids, use_cache=True)
            past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
            sync(device)
            t0 = time.perf_counter()
            for _ in range(decode_new_tokens):
                o = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
            sync(device)
            return (time.perf_counter() - t0) * 1000.0

        for _ in range(warmup):
            decode_once()
        with peak_memory(device) as m:
            times = [decode_once() for _ in range(reps)]
        d = {"unit": "ms", "reps": reps, "warmup": warmup, "prompt_len": decode_prompt_len, "new_tokens": decode_new_tokens,
             "times_ms": times, "per_token_mean_ms": sum(times) / len(times) / decode_new_tokens,
             "peak_allocated_bytes": m["peak_allocated_bytes"]}
        d.update(summarize_ms(times))
        out["decode"] = d
    out["peak_allocated_bytes_overall"] = mem_all["peak_allocated_bytes"]
    out["peak_reserved_bytes_overall"] = mem_all["peak_reserved_bytes"]
    return out
