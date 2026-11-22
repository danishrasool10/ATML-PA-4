"""Step 9: summary tables, paired bootstrap CIs, sensitivity / allocation maps, calibration-vs-held-out
diagnostics and qualitative HellaSwag cases. Reads only saved result files; every displayed number
traces to ``results/task1/conditions/*/result.json`` (+ npz / jsonl) or ``results/task1/selection|sensitivity``."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..common.bootstrap import paired_bootstrap_mean, paired_bootstrap_ppl
from ..common.results import read_json, read_jsonl, write_json
from .constants import LEVELS, PROJ_NAMES

PRUNE_ORDER = ["baseline", "p1_s25", "p2_s25", "p1_s50", "p2_s50", "p3_s50", "p2_24", "p1_s75", "p2_s75", "p3_s75"]
QUANT_ORDER = ["q1_int8", "q2_int8", "q1_int4", "q2_int4", "q3_int4", "q1_int3", "q2_int3", "q3_int3"]
COMPARISONS = [("p3_s50", "p2_s50"), ("p3_s75", "p2_s75"), ("q3_int4", "q2_int4"), ("q3_int3", "q2_int3"),   # required
               ("p2_s25", "p1_s25"), ("p2_s50", "p1_s50"), ("p2_s75", "p1_s75"),                           # Wanda vs magnitude
               ("p2_24", "p2_s50"),                                                                         # 2:4 vs unstructured
               ("q2_int8", "q1_int8"), ("q2_int4", "q1_int4"), ("q2_int3", "q1_int3")]                      # AWQ vs plain PTQ
COLORS = {"P1": "#7a7a7a", "P2": "#1f77b4", "P3": "#d95f02", "Q1": "#7a7a7a", "Q2": "#1f77b4", "Q3": "#d95f02", "2:4": "#2ca02c"}


class Results:
    def __init__(self, root: Path):
        self.root = root
        self.cond = {}
        for d in sorted((root / "conditions").glob("*")):
            if (d / "result.json").exists():
                r = read_json(d / "result.json")
                z = np.load(d / "per_window_and_example.npz")
                r["_arrays"] = {k: z[k] for k in z.files}
                r["_dir"] = d
                self.cond[d.name] = r

    def __contains__(self, k):
        return k in self.cond


def _family_color(cid: str) -> str:
    for k, c in (("p1", COLORS["P1"]), ("p2_24", COLORS["2:4"]), ("p2", COLORS["P2"]), ("p3", COLORS["P3"]),
                 ("q1", COLORS["Q1"]), ("q2", COLORS["Q2"]), ("q3", COLORS["Q3"])):
        if cid.startswith(k):
            return c
    return "black"


def bytes_of(r: dict) -> int | None:
    st = r["storage"]
    if "quantized_model" in st:
        return st["quantized_model"]["packed_total_bytes"]
    if r["family"].startswith(("P", "S")):
        return st["sparse_16bit_model"]["sparse_total_bytes"]
    return st["dense_fp16"]["dense_fp16_total_bytes"]


def summary_table(R: Results, out: Path) -> list[dict]:
    base = R.cond.get("baseline")
    rows = []
    for cid in PRUNE_ORDER + QUANT_ORDER:
        if cid not in R:
            continue
        r = R.cond[cid]
        q = r["quality"]
        c = r["compression"]
        st = r["storage"]
        row = {
            "condition": cid, "family": r["family"],
            "eligible_nonzero_fraction": st["nonzeros"]["eligible_nonzero_fraction"],
            "nominal_bits_per_eligible_weight": (st.get("quantized_model") or st["sparse_16bit_model"] if cid != "baseline" else st["dense_fp16"]).get("nominal_bits_per_eligible_weight", 16.0),
            "dense_fp16_bytes": st["dense_fp16"]["dense_fp16_total_bytes"],
            "packed_or_sparse_bytes": bytes_of(r),
            "calib_mean_rel_err": c.get("mean_rel_err"), "calib_energy_weighted_rel_err": c.get("energy_weighted_rel_err"),
            "calib_nll": q["calibration_nll"]["mean_nll"], "dev_nll": q["development_nll"]["mean_nll"],
            "wikitext2_N": q["wikitext2_test"]["N"], "wikitext2_nll": q["wikitext2_test"]["mean_nll"], "wikitext2_ppl": q["wikitext2_test"]["ppl"],
            "c4val_N": q["c4_validation"]["N"], "c4val_nll": q["c4_validation"]["mean_nll"], "c4val_ppl": q["c4_validation"]["ppl"],
            "hellaswag_n": q["hellaswag"]["n"], "hellaswag_acc": q["hellaswag"]["acc"], "hellaswag_acc_norm": q["hellaswag"]["acc_norm"],
            "hellaswag_first512_acc": q["hellaswag"]["first512"]["acc"], "hellaswag_first512_acc_norm": q["hellaswag"]["first512"]["acc_norm"],
            "compress_s": r["timings_s"].get("calibration_and_compression_s"),
            "prefill512_ms": r["profile"]["prefill"].get("512", next(iter(r["profile"]["prefill"].values())))["median_ms"],
            "prefill2048_ms": r["profile"]["prefill"].get("2048", list(r["profile"]["prefill"].values())[-1])["median_ms"],
            "decode64_ms": r["profile"]["decode"]["median_ms"], "peak_alloc_bytes": r["profile"]["peak_allocated_bytes_overall"],
        }
        if base is not None:
            for k in ("wikitext2", "c4val"):
                key = "wikitext2_test" if k == "wikitext2" else "c4_validation"
                row[f"{k}_ppl_delta_vs_base"] = q[key]["ppl"] - base["quality"][key]["ppl"]
            row["hellaswag_first512_acc_norm_delta_vs_base"] = q["hellaswag"]["first512"]["acc_norm"] - base["quality"]["hellaswag"]["first512"]["acc_norm"]
        rows.append(row)
    write_json(out / "summary_table.json", rows)
    if rows:
        cols = list(rows[0])
        with open(out / "summary_table.csv", "w") as f:
            f.write(",".join(cols) + "\n")
            for r in rows:
                f.write(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")
        short = ["condition", "eligible_nonzero_fraction", "packed_or_sparse_bytes", "wikitext2_ppl", "c4val_ppl", "hellaswag_n", "hellaswag_acc", "hellaswag_acc_norm", "hellaswag_first512_acc_norm", "compress_s"]
        with open(out / "summary_table.md", "w") as f:
            f.write("| " + " | ".join(short) + " |\n|" + "---|" * len(short) + "\n")
            for r in rows:
                f.write("| " + " | ".join(f"{r[c]:.4g}" if isinstance(r.get(c), float) else str(r.get(c)) for c in short) + " |\n")
    return rows


def paired_comparisons(R: Results, out: Path, n_boot: int, seed: int) -> dict:
    res = {}
    pairs = list(COMPARISONS) + [(c, "baseline") for c in R.cond if c != "baseline"]
    for a, b in pairs:
        if a not in R or b not in R:
            continue
        ra, rb = R.cond[a], R.cond[b]
        e = {"a": a, "b": b, "difference": "a - b"}
        for key in ("wikitext2_test", "c4_validation"):
            e[key] = paired_bootstrap_ppl(ra["_arrays"][f"{key}_nll_sum"], ra["_arrays"][f"{key}_count"], rb["_arrays"][f"{key}_nll_sum"],
                                          rb["_arrays"][f"{key}_count"], n_boot=n_boot, seed=seed)
        n = min(len(ra["_arrays"]["hs_correct"]), len(rb["_arrays"]["hs_correct"]))
        e["hellaswag"] = {"n_paired_examples": n,
                          "scope": "full 2000" if n >= 2000 else f"first {n} ids (paired diagnostic subset)",
                          "acc": paired_bootstrap_mean(ra["_arrays"]["hs_correct"][:n], rb["_arrays"]["hs_correct"][:n], n_boot=n_boot, seed=seed),
                          "acc_norm": paired_bootstrap_mean(ra["_arrays"]["hs_correct_norm"][:n], rb["_arrays"]["hs_correct_norm"][:n], n_boot=n_boot, seed=seed)}
        res[f"{a}__minus__{b}"] = e
    write_json(out / "paired_bootstrap.json", res)
    return res


def _heat(ax, M, title, xlabels, ylabel, cmap="viridis", fmt=None, log=False):
    im = ax.imshow(np.log10(np.maximum(M, 1e-12)) if log else M, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="log10" if log else "")


def _grid(vec: np.ndarray, uids: list[str]) -> np.ndarray:
    n_layers = len(uids) // 7
    return np.asarray(vec, float).reshape(n_layers, 7)


def sensitivity_and_allocation_plots(root: Path, out: Path):
    sd = root / "sensitivity"
    if not (sd / "sensitivity_summary.json").exists():
        return
    s = read_json(sd / "sensitivity_summary.json")
    uids = s["uids"]
    lv = list(s["levels"])
    i50, i75 = lv.index(0.5), lv.index(0.75)
    fig, axs = plt.subplots(1, 4, figsize=(17, 5))
    _heat(axs[0], _grid(np.array(s["owl_ratio"]), uids), "OWL outlier ratio (Wanda scores > 5x mean)", PROJ_NAMES, "layer", log=True)
    _heat(axs[1], _grid(np.array(s["local_rel"])[:, i50], uids), "Wanda local rel. output error @ s=0.5", PROJ_NAMES, "layer", log=True)
    _heat(axs[2], _grid(np.array(s["taylor"])[:, i50], uids), "Taylor/Fisher predicted loss increase @ s=0.5", PROJ_NAMES, "layer", log=True)
    _heat(axs[3], _grid(np.array(s["fisher_trace"]), uids), "Fisher trace (HAWQ-style weight)", PROJ_NAMES, "layer", log=True)
    fig.suptitle("Pruning sensitivity probes (calibration data only)")
    fig.tight_layout()
    fig.savefig(out / "sensitivity_pruning.png", dpi=150)
    plt.close(fig)
    qt = sd / "q_tables.json"
    if qt.exists():
        q = read_json(qt)["tables"]
        fig, axs = plt.subplots(1, 4, figsize=(17, 5))
        for ax, b in zip(axs, ("2", "3", "4", "8")):
            _heat(ax, _grid(np.array([q[b][u]["rel_err"] for u in uids]), uids), f"AWQ local rel. error, INT{b}", PROJ_NAMES, "layer", log=True)
        fig.suptitle("Quantization sensitivity: calibration layer-output error per projection")
        fig.tight_layout()
        fig.savefig(out / "sensitivity_quantization.png", dpi=150)
        plt.close(fig)
    # final maps
    maps = []
    for s_ in (50, 75):
        f = root / "selection" / f"p3_allocation_s{s_}.json"
        if f.exists():
            a = read_json(f)
            maps.append((f"P3 sparsity map, global {s_}%", _grid(np.array([a["s"][u] for u in uids]), uids), "magma"))
    for bn in ("int4", "int3"):
        f = root / "conditions" / f"q3_{bn}" / "bit_assignment.json"
        if f.exists():
            a = read_json(f)
            maps.append((f"Q3 bit map, budget = Q2 {bn.upper()}", _grid(np.array([a[u] for u in uids]), uids), "cividis"))
    if maps:
        fig, axs = plt.subplots(1, len(maps), figsize=(4.6 * len(maps), 5), squeeze=False)
        for ax, (t, M, cm) in zip(axs[0], maps):
            _heat(ax, M, t, PROJ_NAMES, "layer", cmap=cm)
        fig.tight_layout()
        fig.savefig(out / "allocation_maps.png", dpi=150)
        plt.close(fig)


def resource_quality_plots(R: Results, out: Path):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    # (a) pruning: quality vs global sparsity
    ax = axs[0]
    for fam, label in (("p1", "P1 magnitude"), ("p2", "P2 Wanda"), ("p3", "P3 non-uniform Wanda")):
        xs, ys = [], []
        for s in (25, 50, 75):
            cid = f"{fam}_s{s}"
            if cid in R:
                xs.append(s / 100)
                ys.append(R.cond[cid]["quality"]["wikitext2_test"]["ppl"])
        if xs:
            ax.plot(xs, ys, "o-", label=label, color=_family_color(fam))
    if "p2_24" in R:
        ax.plot([0.5], [R.cond["p2_24"]["quality"]["wikitext2_test"]["ppl"]], "s", color=COLORS["2:4"], label="2:4 Wanda")
    if "baseline" in R:
        ax.axhline(R.cond["baseline"]["quality"]["wikitext2_test"]["ppl"], color="k", ls=":", label="FP16 baseline")
    ax.set_yscale("log")
    ax.set_xlabel("global eligible-weight sparsity")
    ax.set_ylabel("WikiText-2 test perplexity")
    ax.set_title("Pruning: quality vs sparsity")
    ax.legend(fontsize=8)
    # (b) pruning vs sparse-format bytes, (c) quantization vs packed bytes
    ax = axs[1]
    for cid in PRUNE_ORDER:
        if cid in R and cid != "baseline":
            r = R.cond[cid]
            ax.scatter(bytes_of(r) / 2 ** 20, r["quality"]["wikitext2_test"]["ppl"], color=_family_color(cid))
            ax.annotate(cid, (bytes_of(r) / 2 ** 20, r["quality"]["wikitext2_test"]["ppl"]), fontsize=6)
    ax.set_yscale("log")
    ax.set_xlabel("modelled 16-bit-sparse storage (MiB)  [not a measured kernel]")
    ax.set_title("Pruning: quality vs sparse-format bytes")
    ax = axs[2]
    for cid in QUANT_ORDER:
        if cid in R:
            r = R.cond[cid]
            ax.scatter(bytes_of(r) / 2 ** 20, r["quality"]["wikitext2_test"]["ppl"], color=_family_color(cid))
            ax.annotate(cid, (bytes_of(r) / 2 ** 20, r["quality"]["wikitext2_test"]["ppl"]), fontsize=6)
    ax.set_yscale("log")
    ax.set_xlabel("packed model bytes incl. scales/zero-points/FP16 rest (MiB)")
    ax.set_title("Quantization: quality vs packed bytes")
    fig.tight_layout()
    fig.savefig(out / "quality_vs_resources.png", dpi=150)
    plt.close(fig)


def calibration_vs_heldout(R: Results, out: Path) -> dict:
    """Does a better calibration criterion (dev / calibration NLL, local layer error) predict a better held-out result?"""
    from scipy.stats import spearmanr

    ids = [c for c in R.cond if c != "baseline"]
    crit = {"dev_nll": [R.cond[c]["quality"]["development_nll"]["mean_nll"] for c in ids],
            "calibration_nll": [R.cond[c]["quality"]["calibration_nll"]["mean_nll"] for c in ids],
            "energy_weighted_rel_err": [R.cond[c]["compression"].get("energy_weighted_rel_err", np.nan) for c in ids]}
    held = {"wikitext2_nll": [R.cond[c]["quality"]["wikitext2_test"]["mean_nll"] for c in ids],
            "c4_nll": [R.cond[c]["quality"]["c4_validation"]["mean_nll"] for c in ids]}
    corr = {}
    for ck, cv in crit.items():
        for hk, hv in held.items():
            m = ~np.isnan(np.array(cv, float))
            if m.sum() >= 3:
                rho, p = spearmanr(np.array(cv)[m], np.array(hv)[m])
                corr[f"{ck}__vs__{hk}"] = {"spearman_rho": float(rho), "p": float(p), "n_conditions": int(m.sum())}
    write_json(out / "calibration_vs_heldout.json", corr)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, ck in zip(axs, crit):
        for c, x, y in zip(ids, crit[ck], held["wikitext2_nll"]):
            ax.scatter(x, y, color=_family_color(c))
            ax.annotate(c, (x, y), fontsize=6)
        ax.set_xlabel(ck)
        ax.set_ylabel("WikiText-2 test mean NLL")
        if ck != "dev_nll" or True:
            ax.set_xscale("log" if ck == "energy_weighted_rel_err" else "linear")
    fig.suptitle("Calibration / development criteria vs held-out NLL (one point per condition)")
    fig.tight_layout()
    fig.savefig(out / "calibration_vs_heldout.png", dpi=150)
    plt.close(fig)
    # selection candidates (dev only)
    sel = {}
    for k in ("p3", "q3"):
        f = Path(R.root) / "selection" / f"{k}_selection.json"
        if f.exists():
            sel[k] = read_json(f)
    write_json(out / "selection_summary.json", {k: {"chosen": v["chosen"], "scores_mean_dev_nll": v["scores"], "wall_s": v["selection_wall_s"],
                                                     "uses_test_data": v["uses_test_data"]} for k, v in sel.items()})
    return corr


def wanda_diagnostics(R: Results, out: Path):
    """RQ1: relation between magnitude, activation norms and removal impact across components / sparsity."""
    if "p2_s50" not in R:
        return
    rec = read_json(R.cond["p2_s50"]["_dir"] / "per_unit_records.json")
    if "magnitude_vs_wanda" not in next(iter(rec.values())):
        return
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, s in zip(axs, ("0.25", "0.5", "0.75")):
        for j, name in enumerate(PROJ_NAMES):
            ys = [v["magnitude_vs_wanda"][s]["overlap_of_pruned"] for u, v in rec.items() if u.endswith(name)]
            ax.plot(range(len(ys)), ys, label=name, marker=".")
        ax.set_title(f"overlap of magnitude and Wanda pruned sets, s={s}")
        ax.set_xlabel("layer")
        ax.set_ylim(0, 1)
    axs[0].set_ylabel("|M_mag ∩ M_wanda| / |M_mag|")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "magnitude_vs_wanda_overlap.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for name in PROJ_NAMES:
        ys = [v["act_norm_stats"]["max_over_median"] for u, v in rec.items() if u.endswith(name)]
        ax.plot(ys, label=name, marker=".")
    ax.set_yscale("log")
    ax.set_xlabel("layer")
    ax.set_ylabel("max / median input-channel activation norm")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "activation_outlier_ratio.png", dpi=150)
    plt.close(fig)
    # error ratio magnitude/wanda per projection type (impact of removal)
    tab = {n: {s: float(np.mean([v["magnitude_vs_wanda"][s]["err_magnitude"] / max(v["magnitude_vs_wanda"][s]["err_wanda"], 1e-30)
                                   for u, v in rec.items() if u.endswith(n)])) for s in ("0.25", "0.5", "0.75")} for n in PROJ_NAMES}
    write_json(out / "magnitude_over_wanda_error_ratio.json", tab)


def qualitative_cases(R: Results, cfg: dict, out: Path, n_fixed: int = 6, max_flips: int = 6):
    """Fixed selection = the first ``n_fixed`` manifest ids (decided in advance) plus, post hoc and labelled as such,
    examples where the matched methods disagree."""
    from ..data.manifests import load_hellaswag
    from .evaluation import hs_doc

    try:
        hs = load_hellaswag(cfg["manifests"]["hellaswag"])
    except Exception as e:  # pragma: no cover
        (out / "qualitative_cases.md").write_text(f"HellaSwag manifest unavailable: {e}\n")
        return
    conds = [c for c in ("baseline", "p2_s50", "p3_s50", "q2_int4", "q3_int4", "q2_int3", "q3_int3") if c in R]
    scores = {c: read_jsonl(R.cond[c]["_dir"] / "hellaswag_scores.jsonl") for c in conds}
    n_avail = min(len(v) for v in scores.values())
    sel = [(i, "fixed (first ids of manifest)") for i in range(min(n_fixed, n_avail))]
    flips = []
    for a, b in (("p3_s50", "p2_s50"), ("q3_int4", "q2_int4"), ("q3_int3", "q2_int3")):
        if a in scores and b in scores:
            idx = [i for i in range(n_avail) if scores[a][i]["pred_norm"] != scores[b][i]["pred_norm"]][:max_flips]
            flips += [(i, f"post-hoc: {a} vs {b} disagree (acc_norm choice)") for i in idx]
    lines = ["# Qualitative HellaSwag cases", "", "Same context and candidate endings; per-candidate length-normalised log-likelihood (acc_norm score)."]
    for i, why in sel + flips:
        q, ch, gold = hs_doc(hs.examples[i])
        lines += ["", f"## example index {i} (id {hs.ids[i]}) — {why}", f"**Context:** {q}", f"**Gold:** {gold}", ""]
        lines.append("| candidate | ending | " + " | ".join(conds) + " |")
        lines.append("|---|---|" + "---|" * len(conds))
        for j in range(4):
            cells = []
            for c in conds:
                e = scores[c][i]
                mark = "★" if e["pred_norm"] == j else ""
                cells.append(f"{e['loglik_norm'][j]:.3f}{mark}")
            lines.append(f"| {j}{' (gold)' if j == gold else ''} | {ch[j][:90]} | " + " | ".join(cells) + " |")
    (out / "qualitative_cases.md").write_text("\n".join(lines) + "\n")


def resource_report(R: Results, out: Path):
    rows = []
    for cid, r in R.cond.items():
        p = r["profile"]
        rows.append({"condition": cid, "execution": p["execution"], "reps": p["reps"], "unit": p["unit"],
                     "prefill_ms": {k: v["median_ms"] for k, v in p["prefill"].items()}, "decode_ms_64tok": p["decode"]["median_ms"],
                     "peak_allocated_bytes": p["peak_allocated_bytes_overall"], "logical_bytes": bytes_of(r),
                     "dense_fp16_bytes": r["storage"]["dense_fp16"]["dense_fp16_total_bytes"],
                     "theoretical_compression_vs_dense": r["storage"]["dense_fp16"]["dense_fp16_total_bytes"] / bytes_of(r),
                     "compress_s": r["timings_s"].get("calibration_and_compression_s")})
    write_json(out / "resource_accounting.json", {"note": "All conditions executed as dense FP16 tensors: measured latency/memory differences are not "
                                                         "sparse/low-bit kernel speed-ups; 'logical_bytes' is serialized/packed storage, not GPU allocation.",
                                                  "rows": rows})


def analyze(cfg: dict):
    root = Path(cfg["results_dir"])
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    R = Results(root)
    if not R.cond:
        print("no results found in", root)
        return
    summary_table(R, out)
    paired_comparisons(R, out, cfg["bootstrap"]["n_boot"], cfg["seed"])
    sensitivity_and_allocation_plots(root, out)
    resource_quality_plots(R, out)
    calibration_vs_heldout(R, out)
    wanda_diagnostics(R, out)
    qualitative_cases(R, cfg, out)
    resource_report(R, out)
    print("analysis written to", out)
