"""Task 2 Part B: training, evaluation and run registry for MQAR (steps B1-B3), plus the inference sweep.

python -m pa4t2.mqar_train --stage {b1,b2,b3,sweep,all,analyze} --out results/partB [--smoke]

Fixed protocol (assignment): AdamW (peak LR 8e-4, wd 0.01 on matrices), batch 32, 4,500 updates, 200 warm-up updates
then linear decay to 15% of peak, global-norm clipping 1.0, checkpoint evaluations at {500,1000,1500,2000,3000,4500},
seeds {6304,6305,6306}. Training sequences are freshly generated from a per-run RNG stream. Checkpoint/learning-curve
evaluation uses the frozen *development* split (256 per K x regime); the frozen 1,024-per-(K, regime) test split
are evaluated at every checkpoint too and stored, but nothing is selected on them: the reported model is always the
final update-4500 checkpoint.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pa4common.env_meta import collect_environment
from pa4common.results import read_json, write_json
from pa4common.seed import DEFAULT_SEED, set_seed
from pa4common.timing import peak_memory, summarize_ms, sync
from .mqar import DEFAULT_DATA_ROOT, N_DEV, N_TEST, PrefetchStream, REGIMES, check_invariants, evaluate, loss_fn, make_eval_set, to_torch, train_batch
from .mqar_models import MQARConfig, MQARTransformer, accounting

SEEDS = (6304, 6305, 6306)
CHECKPOINTS = (500, 1000, 1500, 2000, 3000, 4500)
PROTOCOL = dict(peak_lr=8e-4, weight_decay=0.01, batch_size=32, updates=4500, warmup=200, final_lr_frac=0.15, clip=1.0, betas=(0.9, 0.999))

# (run name, attention variant, K, training regime)
B1_RUNS = [("dense_K4_mixed", "dense", 4, "mixed"), ("dense_K8_mixed", "dense", 8, "mixed"), ("dense_K16_mixed", "dense", 16, "mixed"),
           ("dense_K16_near", "dense", 16, "near"),
           ("window64_K4_near", "window", 4, "near"), ("window64_K8_near", "window", 8, "near"), ("window64_K16_near", "window", 16, "near"),
           ("window64_K16_mixed", "window", 16, "mixed")]
B2_RUNS = [("linear_K4_mixed", "linear", 4, "mixed"), ("linear_K8_mixed", "linear", 8, "mixed"), ("linear_K16_mixed", "linear", 16, "mixed"),
           ("sparse_K4_mixed", "sparse", 4, "mixed"), ("sparse_K16_mixed", "sparse", 16, "mixed")]
B3_RUNS = [("linear_l2_K16_mixed", "linear_l2", 16, "mixed"), ("delta_K16_mixed", "delta", 16, "mixed")]

B3_PREDICTION = {
    "written_before_any_B3_training_or_evaluation": True,
    "intervention": "delta-rule write (S_t = S_{t-1} + beta_t (v_t - S_{t-1} phi(k_t)) phi(k_t)^T) with L2-normalised positive features, same persistent state size as additive linear attention",
    "control": "linear_l2: identical features/normalisation/read-out, additive write -- the write rule is the only difference",
    "baselines_from_B2": "linear (ELU+1, normalised read-out) K=16 mixed",
    "mechanism": "additive linear attention superposes all K=16 key-value outer products into one 32x32 state per head; cross-talk between non-orthogonal "
                 "keys is never removed, and the state is shared by every distance, so recall at K=16 is capacity/interference limited. The delta rule "
                 "overwrites the value currently stored under a key (error-correcting write), removing the interference of repeated/non-orthogonal keys without growing the state.",
    "predictions": [
        "P1: delta K=16 mixed accuracy (64-way, final test, mean of 3 seeds) exceeds the additive control by more than 10 percentage points overall.",
        "P2: the gain is not restricted to near queries: accuracy in the [128,256) bin also improves over the control (distance-independent state).",
        "P3: delta remains below dense attention at K=16 mixed, and with 4 heads x 32 dims (rank 32 per head) it will not reach 100%.",
        "P4: parameter count is within 0.3% of the baseline and persistent state bytes per layer do not exceed the linear baseline's.",
    ],
    "development_selection_rule": "No hyper-parameter search. If any choice must be made (e.g. a failed/NaN run), it is made on development-stream far-bin accuracy at K=16 averaged over seeds; "
                                   "the reported checkpoint is always update 4500. Test sets are never used for choices.",
}


def lr_at(step: int, p=PROTOCOL) -> float:
    if step < p["warmup"]:
        return p["peak_lr"] * (step + 1) / p["warmup"]
    frac = (step - p["warmup"]) / max(1, p["updates"] - p["warmup"])
    return p["peak_lr"] * (1 - (1 - p["final_lr_frac"]) * min(1.0, frac))


def make_optimizer(model, p=PROTOCOL):
    decay = [x for n, x in model.named_parameters() if x.ndim >= 2]
    no_decay = [x for n, x in model.named_parameters() if x.ndim < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": p["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
                             lr=p["peak_lr"], betas=p["betas"])


def eval_suite(model, K, device, n_test=None, n_dev=None, data_root=None):
    """Evaluate on all three regimes: frozen dev split (curve / selection) and frozen test split."""
    out = {"dev": {}, "test": {}}
    for split, n in (("dev", n_dev), ("test", n_test)):
        for r in REGIMES:
            res = evaluate(model, make_eval_set(K, r, split, data_root or DEFAULT_DATA_ROOT, n), device)
            res.pop("_per_query")
            out[split][r] = res
    return out


def train_run(name, variant, K, train_regime, seed, out_dir: Path, device, updates=None, checkpoints=None, n_test=None, n_dev=None,
              save_final=True, log=print, data_root=None) -> dict:
    d = out_dir / "runs" / name / f"seed{seed}"
    if (d / "metrics.json").exists():
        return read_json(d / "metrics.json")
    d.mkdir(parents=True, exist_ok=True)
    updates = updates or PROTOCOL["updates"]
    checkpoints = checkpoints or CHECKPOINTS
    p = dict(PROTOCOL, updates=updates)
    set_seed(seed)
    cfg = MQARConfig(attn=variant)
    model = MQARTransformer(cfg).to(device)
    opt = make_optimizer(model, p)
    # staff generator invariants are verified on the first training batch before training starts
    check_invariants(train_batch(K, train_regime, seed, 0, 8))
    stream = PrefetchStream(K, train_regime, seed, updates, p["batch_size"])
    curve, losses, t0 = [], [], time.perf_counter()
    for step in range(updates):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, p)
        model.train()
        batch = to_torch(stream.next(), device)
        loss = loss_fn(model(batch["tokens"]), batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), p["clip"])
        opt.step()
        losses.append(float(loss.detach()))
        if not np.isfinite(losses[-1]):
            log(f"[{name}/{seed}] non-finite loss at step {step + 1}")
        if (step + 1) in checkpoints or step + 1 == updates:
            ev = eval_suite(model, K, device, n_test, n_dev, data_root)
            curve.append({"update": step + 1, "train_loss_last50": float(np.mean(losses[-50:])), "grad_norm": float(gn), "eval": ev})
            log(f"[{name}/{seed}] step {step + 1}: loss {np.mean(losses[-50:]):.4f} dev acc64 " + " ".join(f"{r}={ev['dev'][r]['acc64']:.3f}" for r in REGIMES))
    metrics = {"run": name, "variant": variant, "K": K, "train_regime": train_regime, "seed": seed, "protocol": p, "curve": curve,
               "final": curve[-1]["eval"], "train_wall_s": time.perf_counter() - t0, "n_params": model.n_params(),
               "accounting_T256": accounting(cfg, 256), "loss_over": "value64", "train_losses_every50": losses[::50]}
    if save_final:
        torch.save(model.state_dict(), d / "final.pt")
    write_json(d / "metrics.json", metrics)
    return metrics


def window_control(out_dir: Path, device, seeds=SEEDS, W=64, n_test=None, log=print, data_root=None):
    """Test-time-only W=64 mask on the learned dense K=16 mixed model.

    Receptive-field accounting: with two layers each attending to the previous W positions, information can travel
    at most 2(W-1)=126 positions, so any query whose distance exceeds 126 cannot (structurally) be answered through
    the key-value pair stored at the start of the sequence; the record reports that structural fraction."""
    res = {}
    K = 16
    for seed in seeds:
        f = out_dir / "runs" / "dense_K16_mixed" / f"seed{seed}" / "final.pt"
        if not f.exists():
            continue
        model = MQARTransformer(MQARConfig(attn="dense")).to(device)
        model.load_state_dict(torch.load(f, map_location=device))
        r = {}
        for regime in REGIMES:
            b = make_eval_set(K, regime, "test", data_root or DEFAULT_DATA_ROOT, n_test)
            dense = evaluate(model, b, device)
            win = evaluate(model, b, device, variant="window", window=W)
            reach = float((b["dist"] <= 2 * (W - 1)).mean())
            r[regime] = {"dense": {k: v for k, v in dense.items() if k != "_per_query"}, "window_test_time": {k: v for k, v in win.items() if k != "_per_query"},
                         "fraction_queries_within_2(W-1)": reach, "receptive_field_limit": 2 * (W - 1),
                         "window_acc64_on_reachable": float(win["_per_query"]["correct64"][b["dist"] <= 2 * (W - 1)].mean()) if reach > 0 else None,
                         "window_acc64_on_unreachable": float(win["_per_query"]["correct64"][b["dist"] > 2 * (W - 1)].mean()) if reach < 1 else None}
        res[str(seed)] = r
        log(f"[window-control seed {seed}] done")
    write_json(out_dir / "window_testtime_control.json", res)
    return res


def run_stage(runs, out_dir, device, seeds=SEEDS, **kw):
    for name, variant, K, regime in runs:
        for seed in seeds:
            train_run(name, variant, K, regime, seed, out_dir, device, **kw)


def inference_sweep(out_dir: Path, device, Ts=(256, 1024, 4096, 8192), variants=("dense", "window", "linear", "sparse"), reps=5, warmup=2, B=2,
                    include_delta=False, log=print):
    """Forward-only sweep with fixed random embeddings (matched B, H, d_h); efficient kernels for window / sparse /
    linear (reference mask forms would be O(T^2) memory). Dense = SDPA causal."""
    res = []
    variants = list(variants) + (["delta"] if include_delta else [])
    for v in variants:
        cfg = MQARConfig(attn=v)
        set_seed(DEFAULT_SEED)
        model = MQARTransformer(cfg).to(device).eval()
        for T in Ts:
            g = torch.Generator().manual_seed(DEFAULT_SEED)
            x = torch.randn(B, T, cfg.d_model, generator=g).to(device)
            rec = {"variant": v, "T": T, "B": B, "H": cfg.n_heads, "d_head": cfg.d_head, "reps": reps, "warmup": warmup, "unit": "ms",
                   "kernel": "efficient" if v != "dense" else "sdpa_causal", "theory": accounting(cfg, T)}
            try:
                with torch.no_grad():
                    for _ in range(warmup):
                        model(inputs_embeds=x, efficient=True)
                    sync(device)
                    with peak_memory(device) as m:
                        ts = []
                        for _ in range(reps):
                            sync(device)
                            t0 = time.perf_counter()
                            model(inputs_embeds=x, efficient=True)
                            sync(device)
                            ts.append((time.perf_counter() - t0) * 1000)
                rec.update(times_ms=ts, **summarize_ms(ts), peak_allocated_bytes=m["peak_allocated_bytes"])
            except Exception as e:  # noqa: BLE001
                rec["status"] = "oom" if "out of memory" in str(e).lower() else "error"
                rec["error"] = str(e).splitlines()[0][:200]
                if device == "cuda":
                    torch.cuda.empty_cache()
            res.append(rec)
            log(f"[sweep] {v} T={T}: {rec.get('median_ms', rec.get('status'))}")
    write_json(out_dir / "inference_sweep.json", {"note": "random fixed embeddings; efficient kernels; peak memory is allocated bytes on CUDA, None on CPU", "records": res})
    return res


def accounting_table(out_dir: Path):
    rows = [accounting(MQARConfig(attn=a), 256) for a in ("dense", "window", "sparse", "linear", "linear_l2", "delta")]
    base = rows[3]["trainable_params"]
    for r in rows:
        r["params_vs_linear_baseline_pct"] = 100 * (r["trainable_params"] - base) / base
    write_json(out_dir / "accounting_T256.json", rows)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["b1", "b2", "b3", "sweep", "all", "analyze", "accounting"])
    ap.add_argument("--out", default="results/partB")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="checkout of atml-pa4-datasets (frozen MQAR splits)")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    args = ap.parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kw = {}
    if args.smoke:
        kw = dict(updates=6, checkpoints=(2, 4, 6), n_test=32, n_dev=16)
    kw["data_root"] = args.data_root
    write_json(out / "environment.json", collect_environment(extra={"part": "task2-B", "protocol": PROTOCOL, "seeds": args.seeds, "device_used": device}))
    accounting_table(out)
    seeds = tuple(args.seeds)
    if not args.smoke:
        kw["data_root"] = args.data_root
    if args.stage in ("b1", "all"):
        run_stage(B1_RUNS, out, device, seeds, **kw)
        window_control(out, device, seeds, n_test=kw.get("n_test"), data_root=args.data_root)
    if args.stage in ("b2", "all"):
        run_stage(B2_RUNS, out, device, seeds, **kw)
    if args.stage in ("b3", "all"):
        pred = out / "b3_prediction.json"
        if not pred.exists():                       # prediction is fixed (and timestamped) BEFORE any B3 run
            write_json(pred, {**B3_PREDICTION, "written_at_unix": time.time()})
        run_stage(B3_RUNS, out, device, seeds, **kw)
    if args.stage in ("sweep", "all"):
        Ts = (64, 128) if args.smoke else (256, 1024, 4096, 8192)
        if device == "cpu" and not args.smoke:
            Ts = (256, 1024)
        inference_sweep(out, device, Ts=Ts, reps=2 if args.smoke else 5)
    if args.stage in ("analyze", "all"):
        from .mqar_analysis import analyze
        analyze(out)


if __name__ == "__main__":
    main()
