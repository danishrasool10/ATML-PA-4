"""Task 2 Part A: correctness checks and benchmarks for dense / tiled / SDPA causal attention.

python -m pa4t2.bench_attention --out results/partA            (GPU: full grid)
python -m pa4t2.bench_attention --out /tmp/a --smoke                  (CPU: tiny grid)

Protocol: B=2, H=4, d_h=64, L in {256,...,8192}, FP16 and FP32 where supported; accuracy vs the dense FP32
reference at feasible lengths (beyond ``ref_max_L`` the reference is SDPA-FP32 and the record says so --
no infeasible dense baseline is forced); 10 warmed-up CUDA-synchronised repetitions per condition;
forward latency, forward+backward latency, peak allocated / reserved memory, OOM / unsupported recorded.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from pa4common.env_meta import collect_environment
from pa4common.results import write_json
from pa4common.seed import DEFAULT_SEED, set_seed
from pa4common.timing import peak_memory, summarize_ms, sync
from .attention_ops import (TILED_BACKWARD_NOTE, dense_causal_attention, detect_sdpa_backend, sdpa_attention, supported_backends,
                            tiled_online_softmax_attention)

B, H, D = 2, 4, 64
LENGTHS = (256, 512, 1024, 2048, 4096, 8192)
DTYPES = {"fp16": torch.float16, "fp32": torch.float32}


def make_qkv(L, dtype, device, seed=DEFAULT_SEED, requires_grad=True, scale=1.0):
    g = torch.Generator(device="cpu").manual_seed(seed + L)
    mk = lambda: (torch.randn(B, H, L, D, generator=g) * scale).to(device=device, dtype=dtype).requires_grad_(requires_grad)
    return mk(), mk(), mk()


def make_grad_out(L, dtype, device, seed=DEFAULT_SEED):
    g = torch.Generator(device="cpu").manual_seed(seed + 7 * L)
    return torch.randn(B, H, L, D, generator=g).to(device=device, dtype=dtype)


def is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def make_op(name: str, block: int = 128):
    if name == "dense":
        return dense_causal_attention
    if name == "tiled":
        return lambda q, k, v, mask=None: tiled_online_softmax_attention(q, k, v, block, block, mask)
    if name == "sdpa":
        return lambda q, k, v, mask=None: sdpa_attention(q, k, v, mask)
    if name.startswith("sdpa_"):
        be = name.split("_", 1)[1]
        return lambda q, k, v, mask=None: sdpa_attention(q, k, v, mask, backend=be)
    raise KeyError(name)


def run_fwd_bwd(op, q, k, v, go):
    out = op(q, k, v)
    out.backward(go)
    return out, q.grad, k.grad, v.grad


def err_stats(x, ref):
    x, ref = x.detach().float(), ref.detach().float()
    d = (x - ref).abs()
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "rel_l2": float((x - ref).norm() / ref.norm().clamp(min=1e-30)), "has_nan": bool(torch.isnan(x).any())}


def accuracy_record(name, L, dtype_name, device, ref_max_L, grad_ref_max_L):
    dtype = DTYPES[dtype_name]
    op = make_op(name)
    rec = {"op": name, "L": L, "dtype": dtype_name}
    try:
        q, k, v = make_qkv(L, dtype, device)
        go = make_grad_out(L, dtype, device)
        q32, k32, v32 = (t.detach().float().clone().requires_grad_(True) for t in (q, k, v))
        if L <= ref_max_L:
            ref_name, ref_op = "dense_fp32", dense_causal_attention
        else:
            ref_name, ref_op = "sdpa_fp32", lambda a, b, c: sdpa_attention(a, b, c)
        with_grad = L <= grad_ref_max_L
        if with_grad:
            ref, rq, rk, rv = run_fwd_bwd(ref_op, q32, k32, v32, go.float())
            out, gq, gk, gv = run_fwd_bwd(op, q, k, v, go)
            rec["grad_error"] = {"dq": err_stats(gq, rq), "dk": err_stats(gk, rk), "dv": err_stats(gv, rv)}
        else:
            with torch.no_grad():
                ref, out = ref_op(q32, k32, v32), op(q, k, v)
        rec["reference"] = ref_name
        rec["output_error"] = err_stats(out, ref)
        rec["grad_checked"] = with_grad
    except Exception as e:  # noqa: BLE001
        rec["status"] = "oom" if is_oom(e) else "unsupported_or_error"
        rec["error"] = str(e).splitlines()[0][:200]
        if device == "cuda":
            torch.cuda.empty_cache()
    return rec


def causality_check(name, L, dtype_name, device, block=64):
    """Perturb keys/values at positions > t: outputs at positions <= t must not change (bit-exact for
    these row-independent ops)."""
    dtype = DTYPES[dtype_name]
    op = make_op(name, block=block)
    q, k, v = make_qkv(L, dtype, device, requires_grad=False)
    t = L // 2
    with torch.no_grad():
        base = op(q, k, v)
        k2, v2 = k.clone(), v.clone()
        g = torch.Generator(device="cpu").manual_seed(1)
        k2[:, :, t + 1:] = torch.randn(k2[:, :, t + 1:].shape, generator=g).to(device, dtype) * 5
        v2[:, :, t + 1:] = torch.randn(v2[:, :, t + 1:].shape, generator=g).to(device, dtype) * 5
        pert = op(q, k2, v2)
    d_past = float((base[:, :, : t + 1].float() - pert[:, :, : t + 1].float()).abs().max())
    d_future = float((base[:, :, t + 1:].float() - pert[:, :, t + 1:].float()).abs().max())
    return {"op": name, "L": L, "dtype": dtype_name, "max_change_at_positions_le_t": d_past, "max_change_future": d_future,
            "causal": d_past <= (0.0 if name != "sdpa" else 1e-3), "perturbation_has_effect": d_future > 0}


def tile_size_check(L, dtype_name, device, tiles=(32, 64, 100, 128, 256)):
    dtype = DTYPES[dtype_name]
    q, k, v = make_qkv(L, dtype, device, requires_grad=False)
    with torch.no_grad():
        ref = dense_causal_attention(q.float(), k.float(), v.float())
        res = {}
        for bt in tiles:
            for bk in (bt, max(16, bt // 2)):
                out = tiled_online_softmax_attention(q, k, v, bt, bk)
                res[f"{bt}x{bk}"] = err_stats(out, ref)
    return {"L": L, "dtype": dtype_name, "tiles": res}


def edge_cases(device):
    out = {}
    L = 256
    for dtype_name in ("fp32", "fp16"):
        dtype = DTYPES[dtype_name]
        rec = {}
        for scale in (1.0, 4.0, 16.0, 64.0):   # increasingly extreme score magnitudes (scores ~ scale^2 * sqrt(d))
            q, k, v = make_qkv(L, dtype, device, requires_grad=False, scale=scale)
            with torch.no_grad():
                ref = dense_causal_attention(q.float(), k.float(), v.float())
                r = {}
                for n in ("dense", "tiled", "sdpa"):
                    try:
                        r[n] = err_stats(make_op(n)(q, k, v), ref)
                    except Exception as e:  # noqa: BLE001
                        r[n] = {"error": str(e).splitlines()[0][:160]}
            rec[f"scale_{scale}"] = r
        # explicit mask that removes every key of some query rows (all-masked rows)
        q, k, v = make_qkv(L, dtype, device, requires_grad=False)
        mask = torch.ones(L, L, dtype=torch.bool, device=device)
        mask[10:20, :] = False
        r = {}
        with torch.no_grad():
            ref = dense_causal_attention(q.float(), k.float(), v.float(), mask)
            for n in ("dense", "tiled", "sdpa"):
                try:
                    o = make_op(n)(q, k, v, mask)
                    r[n] = {"nan_rows": int(torch.isnan(o).any(-1).any(-1).sum()), "masked_rows_output_abs_max": float(torch.nan_to_num(o[:, :, 10:20].float()).abs().max()),
                            "other_rows_err": err_stats(torch.cat([o[:, :, :10], o[:, :, 20:]], 2), torch.cat([ref[:, :, :10], ref[:, :, 20:]], 2))}
                except Exception as e:  # noqa: BLE001
                    r[n] = {"error": str(e).splitlines()[0][:160]}
        rec["all_masked_rows"] = r
        out[dtype_name] = rec
    return out


def bench_condition(name, L, dtype_name, device, reps=10, warmup=3):
    dtype = DTYPES[dtype_name]
    op = make_op(name)
    rec = {"op": name, "L": L, "dtype": dtype_name, "reps": reps, "warmup": warmup, "unit": "ms", "device": device}
    if name == "tiled":
        rec["backward_implementation"] = TILED_BACKWARD_NOTE
    try:
        q, k, v = make_qkv(L, dtype, device, requires_grad=True)
        go = make_grad_out(L, dtype, device)

        def fwd():
            with torch.no_grad():
                return op(q, k, v)

        def fb():
            q.grad = k.grad = v.grad = None
            o = op(q, k, v)
            o.backward(go)

        for label, fn in (("forward", fwd), ("forward_backward", fb)):
            try:
                for _ in range(warmup):
                    fn()
                sync(device)
                with peak_memory(device) as m:
                    ts = []
                    for _ in range(reps):
                        sync(device)
                        t0 = time.perf_counter()
                        fn()
                        sync(device)
                        ts.append((time.perf_counter() - t0) * 1000)
                rec[label] = {"times_ms": ts, **summarize_ms(ts), "peak_allocated_bytes": m["peak_allocated_bytes"],
                              "peak_reserved_bytes": m["peak_reserved_bytes"]}
            except Exception as e:  # noqa: BLE001
                rec[label] = {"status": "oom" if is_oom(e) else "unsupported_or_error", "error": str(e).splitlines()[0][:200]}
            if device == "cuda":
                torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        rec["status"] = "oom" if is_oom(e) else "unsupported_or_error"
        rec["error"] = str(e).splitlines()[0][:200]
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/partA")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    set_seed(DEFAULT_SEED)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    lengths = (64, 128, 192) if args.smoke else LENGTHS
    if device == "cpu" and not args.smoke:
        lengths = tuple(L for L in LENGTHS if L <= 2048)   # CPU runs are plumbing checks only
    reps, warmup = (2, 1) if args.smoke else (10, 3)
    ref_max, grad_max = (192, 192) if args.smoke else (4096, 2048)
    dtypes = ["fp32", "fp16"]
    methods = ["dense", "tiled", "sdpa", "sdpa_flash", "sdpa_efficient", "sdpa_math"]

    env = collect_environment(extra={"part": "task2-A", "device_used": device, "B_H_d": [B, H, D], "lengths": lengths})
    write_json(out / "environment.json", env)

    q, k, v = make_qkv(lengths[-1] if args.smoke else 1024, torch.float16 if device == "cuda" else torch.float32, device, requires_grad=False)
    backend_info = {"forced_support": supported_backends(q, k, v), "default_dispatch": detect_sdpa_backend(q, k, v)}
    for dn in dtypes:
        qq, kk, vv = make_qkv(256, DTYPES[dn], device, requires_grad=False)
        backend_info[f"default_dispatch_{dn}_L256"] = detect_sdpa_backend(qq, kk, vv)
        backend_info[f"forced_support_{dn}_L256"] = supported_backends(qq, kk, vv)
    write_json(out / "sdpa_backends.json", backend_info)

    verification = {"causality": [], "tile_sizes": [], "edge_cases": edge_cases(device), "accuracy": []}
    for dn in dtypes:
        for n in ("dense", "tiled", "sdpa"):
            verification["causality"].append(causality_check(n, lengths[min(1, len(lengths) - 1)], dn, device))
        verification["tile_sizes"].append(tile_size_check(lengths[min(1, len(lengths) - 1)] + 5, dn, device))   # ragged last tile
        for L in lengths:
            for n in ("dense", "tiled", "sdpa"):
                verification["accuracy"].append(accuracy_record(n, L, dn, device, ref_max, grad_max))
    write_json(out / "verification.json", verification)

    bench = []
    for dn in dtypes:
        for L in lengths:
            for n in methods:
                bench.append(bench_condition(n, L, dn, device, reps, warmup))
                write_json(out / "benchmark.json", {"note": "dense baseline not forced where infeasible: OOM is recorded", "records": bench})
    print(f"Part A results written to {out}")


if __name__ == "__main__":
    main()
