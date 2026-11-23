"""C5 benchmark: real packed-cache encode / decode and full-model decode-step timing.

For batch sizes {1, 8} x context lengths {2048, 4096} (synthetic fixed-length prompts; no quality is measured here):

  encode            quantize + bit-pack the whole K/V cache of all layers      (INT4 channel-K / token-V, group 64)
  decode            unpack + dequantize the whole packed cache back to FP16
  step_fp16         one decode step of the model with the unchanged FP16 cache
  step_evict        one decode step with the FP16 eviction cache at the matched budget (T tokens)
  step_packed_int4  one decode step where every layer first unpacks its packed INT4 cache (naive "decode then attend";
                    this is the real cost of the packed layout without a fused kernel)

Each condition: >= 10 timed repetitions after warm-up, CUDA synchronised on both edges, median/mean/std/min/max reported;
allocated and reserved peak GPU memory are recorded; OOM and unsupported conditions are recorded, not skipped silently.
Packed *storage* bytes are logical (payload + metadata); allocated GPU memory is reported separately.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pa4common.env_meta import collect_environment
from pa4common.results import write_json
from pa4common.seed import DEFAULT_SEED, set_seed
from pa4common.timing import summarize_ms
from pa4common.qwen_engine import KVCache, KVEngine, LayerCache
from .policies import _select
from .quant import PackedTensor, budget_bytes, fp16_bytes, fp16_token_capacity

REPS, WARMUP = 10, 3


def _time(fn, device, reps=REPS, warmup=WARMUP):
    cuda = torch.device(device).type == "cuda"
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    ts = []
    for _ in range(reps):
        if cuda:
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            fn()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        else:
            import time
            t = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t) * 1000)
    out = summarize_ms(ts)
    out["reps"] = reps
    out["unit"] = "ms"
    if cuda:
        out["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        out["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    return out


class PackedCache:
    """Real packed INT4 channel-K / token-V cache (complete 64-token groups) + FP16 tail, per layer."""

    def __init__(self, cache: KVCache, bits: int = 4):
        self.layers = []
        for lc in cache.layers:
            n_full = (lc.n // 64) * 64
            self.layers.append((PackedTensor(lc.K[..., :n_full, :], bits, "channel"), PackedTensor(lc.V[..., :n_full, :], bits, "token"),
                                lc.K[..., n_full:, :].clone(), lc.V[..., n_full:, :].clone()))

    def nbytes(self) -> int:
        return sum(k.nbytes() + v.nbytes() + 2 * (kt.numel() + vt.numel()) for k, v, kt, vt in self.layers)

    def decode(self) -> KVCache:
        return KVCache([LayerCache(torch.cat([k.decode(), kt], -2), torch.cat([v.decode(), vt], -2)) for k, v, kt, vt in self.layers])


def bench_condition(engine: KVEngine, B: int, L: int, device, reps=REPS) -> dict:
    rec = {"batch": B, "context": L, "reps": reps, "device": str(device)}
    dims = (engine.n_layers, engine.Hkv, engine.d)
    try:
        g = torch.Generator(device="cpu").manual_seed(DEFAULT_SEED + L)
        ids = torch.randint(10, engine.model.config.vocab_size - 10, (1, L), generator=g).to(device)
        _, cache = engine.prefill(ids)
        cache = cache.expand_batch(B)
        rec["fp16_cache_bytes"] = cache.fp16_nbytes()
        rec["formula_M16"] = fp16_bytes(L, *dims) * B
        rec["formula_M4_budget"] = budget_bytes(4, L, *dims) * B
        packed = PackedCache(cache)
        rec["packed_int4_bytes"] = packed.nbytes()
        rec["packed_over_fp16"] = rec["packed_int4_bytes"] / rec["fp16_cache_bytes"]
        T = fp16_token_capacity(budget_bytes(4, L, *dims), *dims)
        evict = _select(cache, torch.cat([torch.arange(8), torch.arange(L - (T - 8), L)]).to(device))
        rec["evict_tokens"] = T
        tok = torch.tensor([[17]], device=device).expand(B, -1).contiguous()
        pos = torch.full((B, 1), L, device=device)

        def step(c):
            def f():
                cc = KVCache([LayerCache(l.K, l.V) for l in c.layers])          # shallow: decode step appends to a throw-away list
                engine.forward(tok, pos, cc, update=False)
            return f

        rec["encode"] = _time(lambda: PackedCache(cache), device, reps)
        rec["decode"] = _time(lambda: packed.decode(), device, reps)
        rec["step_fp16"] = _time(step(cache), device, reps)
        rec["step_evict"] = _time(step(evict), device, reps)

        def packed_step():
            engine.forward(tok, pos, packed.decode(), update=False)

        rec["step_packed_int4"] = _time(packed_step, device, reps)
        rec["status"] = "ok"
    except torch.cuda.OutOfMemoryError as e:
        rec["status"] = "oom"
        rec["error"] = str(e)[:200]
        torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/partC/packed_benchmark.json")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--reps", type=int, default=REPS)
    a = ap.parse_args()
    set_seed(DEFAULT_SEED)
    if a.smoke:
        from pa4common.tiny import make_tiny_model
        model, dev, contexts, batches = make_tiny_model(layers=2, hidden=128, heads=2, kv_heads=1), "cpu", [256, 512], [1, 2]
        tok = None
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .evaluate import MODEL_ID
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, attn_implementation="sdpa").to(dev).eval()
        tok, contexts, batches = AutoTokenizer.from_pretrained(MODEL_ID), [2048, 4096], [1, 8]
    eng = KVEngine(model)
    res = {"env": collect_environment(model, tok), "reps": a.reps, "warmup": WARMUP,
           "note": "decode-step 'packed' includes unpack+dequantize of every layer (no fused kernel); storage bytes are logical packed bytes, "
                   "peak_allocated/reserved are CUDA allocator figures; ms are CUDA-event timings (wall clock on CPU).",
           "conditions": [bench_condition(eng, B, L, dev, a.reps) for B in batches for L in contexts]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(a.out, res)
    for r in res["conditions"]:
        print(r["batch"], r["context"], r["status"], {k: round(r[k]["median_ms"], 2) for k in ("encode", "decode", "step_fp16", "step_evict", "step_packed_int4") if k in r})


if __name__ == "__main__":
    main()
