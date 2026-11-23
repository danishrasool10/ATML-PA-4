"""Task 2 Part B analysis: final-test tables, learning curves, distance-bin plots, paired bootstrap for B3, sweep plot."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pa4common.bootstrap import paired_bootstrap_mean
from pa4common.results import read_json, write_json
from .mqar import DEFAULT_DATA_ROOT, REGIMES, evaluate, make_eval_set
from .mqar_models import MQARConfig, MQARTransformer
from .mqar_train import B1_RUNS, B2_RUNS, B3_RUNS, SEEDS

ALL_RUNS = B1_RUNS + B2_RUNS + B3_RUNS
BIN_KEYS = ["[0,32)", "[32,64)", "[64,128)", "[128,256)"]


def load_runs(out: Path):
    res = {}
    for name, variant, K, tr in ALL_RUNS:
        ms = []
        for s in SEEDS:
            f = out / "runs" / name / f"seed{s}" / "metrics.json"
            if f.exists():
                ms.append(read_json(f))
        if ms:
            res[name] = ms
    return res


def final_table(runs: dict) -> list[dict]:
    rows = []
    for name, ms in runs.items():
        m0 = ms[0]
        for regime in REGIMES:
            accs = [m["final"]["test"][regime]["acc64"] for m in ms]
            accK = [m["final"]["test"][regime]["accK"] for m in ms]
            row = {"run": name, "variant": m0["variant"], "K": m0["K"], "train_regime": m0["train_regime"], "eval_regime": regime, "n_seeds": len(ms),
                   "acc64_mean": float(np.mean(accs)), "acc64_std": float(np.std(accs)), "accK_mean": float(np.mean(accK)),
                   "chance64": 1 / 64, "chanceK": 1 / m0["K"], "ce_mean": float(np.mean([m["final"]["test"][regime]["ce"] for m in ms])),
                   "n_params": m0["n_params"], "state_bytes_per_layer_fp32": m0["accounting_T256"]["state_bytes_per_layer_fp32"],
                   "pairs_per_layer": m0["accounting_T256"]["pairs_per_layer"]}
            for b in BIN_KEYS:
                v = [m["final"]["test"][regime]["bins"][b]["acc64"] for m in ms if b in m["final"]["test"][regime]["bins"]]
                row[f"acc64_bin_{b}"] = float(np.mean(v)) if v else None
            rows.append(row)
    return rows


def plots(runs, out: Path):
    # learning curves (dev stream, mixed-regime evaluation)
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, regime in zip(axs, REGIMES):
        for name, ms in runs.items():
            ups = [c["update"] for c in ms[0]["curve"]]
            ys = np.mean([[c["eval"]["dev"][regime]["acc64"] for c in m["curve"]] for m in ms], 0)
            ax.plot(ups, ys, marker=".", label=name, lw=1)
        ax.set_title(f"dev-stream accuracy, eval regime = {regime}")
        ax.set_xlabel("optimizer updates")
        ax.set_ylabel("64-way accuracy")
        ax.axhline(1 / 64, color="k", ls=":", lw=0.8)
    axs[0].legend(fontsize=5, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=140)
    plt.close(fig)
    # distance-bin accuracy of final models on the mixed test set
    fig, ax = plt.subplots(figsize=(11, 4.6))
    names = list(runs)
    w = 0.8 / 4
    for bi, b in enumerate(BIN_KEYS):
        ys = []
        for n in names:
            v = [m["final"]["test"]["mixed"]["bins"].get(b, {}).get("acc64", np.nan) for m in runs[n]]
            ys.append(np.nanmean(v))
        ax.bar(np.arange(len(names)) + bi * w, ys, w, label=f"distance {b}")
    ax.set_xticks(np.arange(len(names)) + 1.5 * w)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("final-test 64-way accuracy (mixed eval set)")
    ax.axhline(1 / 64, color="k", ls=":", lw=0.8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "accuracy_by_distance_bin.png", dpi=140)
    plt.close(fig)
    f = out / "inference_sweep.json"
    if f.exists():
        recs = read_json(f)["records"]
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
        for v in sorted({r["variant"] for r in recs}):
            rr = [r for r in recs if r["variant"] == v and "median_ms" in r]
            axs[0].plot([r["T"] for r in rr], [r["median_ms"] for r in rr], marker="o", label=v)
            mm = [r for r in rr if r.get("peak_allocated_bytes")]
            if mm:
                axs[1].plot([r["T"] for r in mm], [r["peak_allocated_bytes"] / 2 ** 20 for r in mm], marker="o", label=v)
        for a, t in zip(axs, ("forward latency (ms, median)", "peak allocated memory (MiB)")):
            a.set_xscale("log", base=2)
            a.set_yscale("log")
            a.set_xlabel("sequence length T")
            a.set_title(t)
        axs[0].legend()
        fig.tight_layout()
        fig.savefig(out / "inference_sweep.png", dpi=140)
        plt.close(fig)


def per_query_correct(out: Path, run: str, variant: str, K: int, seed: int, regime: str, n=None, device="cpu", data_root=None):
    f = out / "runs" / run / f"seed{seed}" / "final.pt"
    if not f.exists():
        return None
    model = MQARTransformer(MQARConfig(attn=variant)).to(device)
    model.load_state_dict(torch.load(f, map_location=device))
    b = make_eval_set(K, regime, "test", data_root or DEFAULT_DATA_ROOT, n)
    r = evaluate(model, b, device)
    return r["_per_query"]["correct64"], b["dist"]


def b3_bootstrap(out: Path, n_boot=10000, device="cpu"):
    """Paired (identical sequences and queries) bootstrap of per-query correctness, seeds pooled."""
    pairs = [("delta_K16_mixed", "linear_l2_K16_mixed"), ("delta_K16_mixed", "linear_K16_mixed"), ("linear_l2_K16_mixed", "linear_K16_mixed"),
             ("delta_K16_mixed", "dense_K16_mixed")]
    variants = {n: v for n, v, _, _ in ALL_RUNS}
    res = {}
    for a, b in pairs:
        for regime in ("mixed", "far"):
            ca, cb = [], []
            for s in SEEDS:
                xa = per_query_correct(out, a, variants[a], 16, s, regime, device=device)
                xb = per_query_correct(out, b, variants[b], 16, s, regime, device=device)
                if xa is None or xb is None:
                    continue
                ca.append(xa[0].astype(float).ravel())
                cb.append(xb[0].astype(float).ravel())
            if ca:
                r = paired_bootstrap_mean(np.concatenate(ca), np.concatenate(cb), n_boot=n_boot)
                r["note"] = "resampling unit = query; sequences are identical across models (same frozen test split), seeds pooled"
                res[f"{a}__minus__{b}__{regime}"] = r
    write_json(out / "b3_paired_bootstrap.json", res)
    return res


def analyze(out: Path, device="cpu"):
    out = Path(out)
    runs = load_runs(out)
    if not runs:
        print("no Part B runs found")
        return
    rows = final_table(runs)
    write_json(out / "final_test_table.json", rows)
    cols = ["run", "eval_regime", "acc64_mean", "acc64_std", "accK_mean", "chance64", "chanceK"] + [f"acc64_bin_{b}" for b in BIN_KEYS]
    with open(out / "final_test_table.md", "w") as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r.get(c), float) else str(r.get(c)) for c in cols) + " |\n")
    plots(runs, out)
    if any(n.startswith("delta") for n in runs):
        b3_bootstrap(out, device=device)
    print("Part B analysis written to", out)
