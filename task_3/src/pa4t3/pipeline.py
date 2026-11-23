"""Task 3 orchestration (CLI).

    python -m pa4t3.pipeline validate    --data-root ../atml-pa4-datasets --out results
    python -m pa4t3.pipeline distill     --mode off_policy --seed 6304 ...
    python -m pa4t3.pipeline banks | spec_bench | throughput | standalone | analyze
    python -m pa4t3.pipeline all         (validation gate -> distill x4 -> banks -> step 6 -> step 7 -> analysis)
    add --smoke for tiny random Qwen2 models, a byte tokenizer and synthetic data (CPU, no downloads, tiny budgets)

Sweeps refuse to start unless ``<out>/validation.json`` exists with ``passed: true`` (Step 1 gate).
Drafts: ``original`` (unchanged 0.5B), ``off_policy_s6304``, ``off_policy_s6305``, ``on_policy_s6304``, ``on_policy_s6305``; trained
drafts are the adapter merged into a *fresh* FP16 0.5B backbone (merge time recorded and logged).
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from pa4common.bootstrap import paired_bootstrap_mean
from pa4common.paths import data_root
from pa4common.env_meta import collect_environment
from pa4common.results import read_json, write_json
from pa4common.seed import DEFAULT_SEED, set_seed
from . import banks as B
from .bench import analyze_spec, breakeven_requests, run_spec_benchmark, standalone_eval
from .data import Item, load_gsm8k_locked, load_short
from .distill import DistillConfig, QUOTAS, distill, merge_adapter
from .lm import CachedLM
from .lora import inject_lora, load_lora_state
from .validate import run_all

TARGET_ID, DRAFT_ID = "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"
SEEDS = (6304, 6305)
MODES = ("off_policy", "on_policy")
TRAINED = [f"{m}_s{s}" for m in MODES for s in SEEDS]
DRAFTS = ["original"] + TRAINED


class Ctx:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.smoke = a.smoke
        set_seed(DEFAULT_SEED)
        if self.smoke:
            from pa4common.tiny import ByteChatTokenizer, make_tiny_model
            self.device = "cpu"
            self.tok = ByteChatTokenizer()
            self._mk_t = lambda: make_tiny_model(seed=101, layers=3, hidden=128, heads=4, kv_heads=2, dtype=torch.float32)
            self._mk_d = lambda: make_tiny_model(seed=102, layers=2, hidden=64, heads=2, kv_heads=1, dtype=torch.float32)
            self.eos = {2}
            self.math_train, self.short_train, self.math_test, self.short_test = smoke_items()
            self.quotas, self.caps = {"math": 120, "short": 40}, {"math": 24, "short": 8}
            self.max_new, self.standalone_caps, self.n_throughput = 12, {"math": 24, "short": 8}, 4
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tok = AutoTokenizer.from_pretrained(TARGET_ID)
            dt = torch.float16 if self.device == "cuda" else torch.float32
            self._mk_t = lambda: AutoModelForCausalLM.from_pretrained(TARGET_ID, torch_dtype=dt, attn_implementation="sdpa").to(self.device).eval()
            self._mk_d = lambda: AutoModelForCausalLM.from_pretrained(DRAFT_ID, torch_dtype=dt, attn_implementation="sdpa").to(self.device).eval()
            self.eos = {int(self.tok.convert_tokens_to_ids("<|im_end|>")), int(self.tok.convert_tokens_to_ids("<|endoftext|>"))}
            root = Path(a.data_root)
            pl = root / "data" / "public_locked"
            self.math_train = load_gsm8k_locked(pl / "gsm8k_train_locked.jsonl.gz", "train")
            self.math_test = load_gsm8k_locked(pl / "gsm8k_test_locked.jsonl.gz", "test")
            self.short_train = load_short(root / "data" / "short_form" / "short_train.jsonl")
            self.short_test = load_short(root / "data" / "short_form" / "short_test.jsonl")
            self.quotas, self.caps = dict(QUOTAS), {"math": 192, "short": 32}
            self.max_new, self.standalone_caps, self.n_throughput = 64, {"math": 512, "short": 32}, 24
        self._target = None
        self.eval_items = self.math_test + self.short_test            # 60 + 30 familiar + 30 shifted = 120 held-out prompts
        self.drafts_sel = [d for d in (a.drafts.split(",") if a.drafts else DRAFTS)]

    # ------------------------------------------------------------------ models
    def target(self):
        if self._target is None:
            self._target = CachedLM(self._mk_t(), "target")
        return self._target

    def adapter_dir(self, name):
        return self.out / "distill" / name

    def draft_lm(self, name: str, merged: bool = True) -> CachedLM:
        m = self._mk_d()
        if name == "original":
            return CachedLM(m, name)
        state = torch.load(self.adapter_dir(name) / "adapter.pt")
        if merged:
            merged_m, _ = merge_adapter(m, state, DistillConfig(), self.device)
            return CachedLM(merged_m, name)
        inject_lora(m, 8, 16.0)
        load_lora_state(m, state)
        return CachedLM(m.eval(), name + "_unmerged")

    def gate(self):
        f = self.out / "validation.json"
        if not f.exists() or not read_json(f).get("passed"):
            raise SystemExit(f"Step 1 validation has not passed ({f}); run the 'validate' stage first.")


def smoke_items():
    mt = [Item(f"GSM8K-{i:04d}", "math", f"Tom has {i + 2} apples and buys {i + 3} more. How many apples?", str(2 * i + 5), None, "train") for i in range(8)]
    st = [Item(f"SHORT-train-{i:04d}", "short", f"Invoice: purchased {100 + i} crates. How many crates? Give the integer.", str(100 + i), 1, "train") for i in range(6)]
    mte = [Item(f"GSM8K-{i:04d}", "math", f"Ann has {i + 4} pens and loses {i % 3}. How many pens?", str(4 + i - i % 3), None, "test") for i in range(6)]
    ste = [Item(f"SHORT-{'familiar' if i < 3 else 'shifted'}-{i:04d}", "short", f"Report date: March {i + 1}, 2020. Return only the month.", "March", 0,
                "familiar" if i < 3 else "shifted") for i in range(6)]
    return mt, st, mte, ste


# ---------------------------------------------------------------------------------------------- stages
def stage_validate(c: Ctx):
    t = c.target().model
    d0 = c._mk_d()
    tok_d = c.tok
    if not c.smoke:
        from transformers import AutoTokenizer
        tok_d = AutoTokenizer.from_pretrained(DRAFT_ID)
    lo, hi = (10, 250) if c.smoke else (1000, 20000)
    res = run_all(t, d0, c._mk_d, c.tok, tok_d, c.device, c.out, vocab_range=(lo, hi), n_toy=c.a.toy_n)
    print(json.dumps({k: (v["passed"] if isinstance(v, dict) else v) for k, v in res.items()}, indent=1))
    if not res["passed"]:
        raise SystemExit("validation FAILED")


def stage_distill(c: Ctx):
    c.gate()
    modes = [c.a.mode] if c.a.mode else list(MODES)
    seeds = [c.a.seed] if c.a.seed else list(SEEDS)
    for mode in modes:
        for seed in seeds:
            name = f"{mode}_s{seed}"
            d = c.adapter_dir(name)
            if (d / "summary.json").exists():
                print("skip (done):", name)
                continue
            cfg = DistillConfig(seed=seed, quotas=c.quotas, caps=c.caps, amp=c.device == "cuda")
            student = c._mk_d()
            summ = distill(mode, student, c.target(), c.tok, c.math_train, c.short_train, cfg, d, c.eos, c.device)
            merged, secs = merge_adapter(c._mk_d(), torch.load(d / "adapter.pt"), cfg, c.device)
            ids = [i for i in c.tok(c.math_train[0].prompt, add_special_tokens=False)["input_ids"]][:48]
            x = torch.tensor([ids], device=c.device)
            with torch.no_grad():
                diff = float((student(input_ids=x).logits.float() - merged(input_ids=x).logits.float()).abs().max())
            write_json(d / "merge.json", {"adapter_merge_s": secs, "max_abs_logit_diff_merged_vs_unmerged": diff, "fresh_backbone": True})
            print(name, "scored", summ["scored_tokens"], "updates", summ["updates"], "skipped", summ["skipped_or_overflow_updates"], f"merge diff {diff:.4g}")
            del student, merged
            if c.device == "cuda":
                torch.cuda.empty_cache()


def stage_banks(c: Ctx):
    c.gate()
    root = c.out / "banks"
    root.mkdir(exist_ok=True)
    tgt = c.target()
    staff = None if c.smoke else Path(c.a.data_root) / "data" / "prefix_banks" / "prefix_banks_locked.jsonl.gz"
    items = c.eval_items
    recs = []
    if staff is not None and staff.exists():
        recs += B.load_banks(staff)
        print("loaded staff banks:", sorted({r["bank"] for r in recs}))
    else:
        print("staff prefix banks not found: generating teacher / original_draft banks locally with the same sampling rule")
        orig_lm = c.draft_lm("original")
        recs += B.make_bank("teacher", items, c.tok, B.autoregressive_responder(tgt, c.eos, c.device), c.device, cap=c.caps["math"])
        recs += B.make_bank("original_draft", items, c.tok, B.autoregressive_responder(orig_lm, c.eos, c.device), c.device, cap=c.caps["math"])
    orig = c.draft_lm("original")
    f = root / "exact_spec_original_gamma4.jsonl.gz"
    if not f.exists():
        B.write_banks(f, B.make_bank("exact_spec_original_gamma4", items, c.tok, B.speculative_responder(tgt, orig, c.eos, c.device, 4), c.device, cap=c.caps["math"]))
    recs += B.load_banks(f)
    drafts = {"original": orig}
    for name in TRAINED:
        if not (c.adapter_dir(name) / "adapter.pt").exists():
            continue
        lm = c.draft_lm(name)
        drafts[name] = lm
        f = root / f"student_{name}.jsonl.gz"
        if not f.exists():
            B.write_banks(f, B.make_bank(f"student_{name}", items, c.tok, B.autoregressive_responder(lm, c.eos, c.device), c.device, cap=c.caps["math"]))
        recs += B.load_banks(f)
    rows = B.evaluate_alignment(tgt, drafts, c.tok, recs)
    with gzip.open(root / "alignment_rows.jsonl.gz", "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    summ = B.summarize_alignment(rows, "original")
    summ["_note"] = "metrics: forward KL(p_T||q), reverse KL(q||p_T), TV, greedy agreement, expected acceptance 1-TV; temperature 1, full vocabulary"
    write_json(c.out / "alignment_summary.json", summ)


def stage_spec_bench(c: Ctx):
    c.gate()
    tgt = c.target()
    for name in c.drafts_sel:
        if name != "original" and not (c.adapter_dir(name) / "adapter.pt").exists():
            continue
        run_spec_benchmark(tgt, {name: c.draft_lm(name)}, c.tok, c.eval_items, c.eos, c.device, c.out / "spec_runs.jsonl", reps=2, max_new=c.max_new, tag="step6")
    # one unmerged adapter timed separately (merged counterpart on the identical subset / order)
    trained = [n for n in c.drafts_sel if n != "original" and (c.adapter_dir(n) / "adapter.pt").exists()]
    if trained:
        n = trained[0]
        sub = c.eval_items[: min(30, len(c.eval_items))]
        run_spec_benchmark(tgt, {n + "_merged": c.draft_lm(n, True), n + "_unmerged": c.draft_lm(n, False)}, c.tok, sub, c.eos, c.device,
                           c.out / "spec_runs_unmerged.jsonl", gammas=(4,), reps=1, max_new=c.max_new, tag="step6_unmerged")


def stage_throughput(c: Ctx):
    c.gate()
    tgt = c.target()
    items = c.math_test[: c.n_throughput]
    for name in c.drafts_sel:
        if name != "original" and not (c.adapter_dir(name) / "adapter.pt").exists():
            continue
        run_spec_benchmark(tgt, {name: c.draft_lm(name)}, c.tok, items, c.eos, c.device, c.out / "throughput_runs.jsonl", reps=2, max_new=64 if not c.smoke else c.max_new,
                           ignore_eos=True, tag="step7_fixed64")


def stage_standalone(c: Ctx):
    c.gate()
    f = c.out / "standalone.jsonl"
    done = {(json.loads(l)["model"]) for l in open(f)} if f.exists() else set()
    models = {"target": lambda: c.target()}
    for n in c.drafts_sel:
        if n == "original" or (c.adapter_dir(n) / "adapter.pt").exists():
            models[n] = (lambda n=n: c.draft_lm(n))
    for name, mk in models.items():
        if name in done:
            continue
        rows = standalone_eval(mk(), c.tok, c.eval_items, c.eos, c.device, c.standalone_caps)
        with open(f, "a") as fh:
            for r in rows:
                fh.write(json.dumps({"model": name, **r}) + "\n")
        print("standalone", name, "math acc", np.mean([r["correct"] for r in rows if r["domain"] == "math"]))


def _read_jsonl(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def stage_analyze(c: Ctx, n_boot=10000):
    rep = {"env": collect_environment(c.target().model, c.tok), "n_boot": n_boot}
    runs = _read_jsonl(c.out / "spec_runs.jsonl")
    if runs:
        rep["step6_speculative"] = analyze_spec(runs, n_boot)
    un = _read_jsonl(c.out / "spec_runs_unmerged.jsonl")
    if un:
        rep["step6_unmerged_vs_merged"] = analyze_spec(un, n_boot)
    th = _read_jsonl(c.out / "throughput_runs.jsonl")
    if th:
        rep["step7_fixed64_throughput"] = analyze_spec(th, n_boot)
    st = _read_jsonl(c.out / "standalone.jsonl")
    if st:
        tab = {}
        for m in sorted({r["model"] for r in st}):
            rr = [r for r in st if r["model"] == m]
            tab[m] = {}
            for key, sel in (("math", lambda r: r["domain"] == "math"), ("short_familiar", lambda r: r["split"] == "familiar"), ("short_shifted", lambda r: r["split"] == "shifted")):
                s = [r for r in rr if sel(r)]
                if s:
                    tab[m][key] = {"n": len(s), "accuracy": float(np.mean([r["correct"] for r in s])), "format_ok": float(np.mean([r["format_ok"] for r in s])),
                                   "finished_within_cap": float(np.mean([r["finished"] for r in s])), "mean_tokens": float(np.mean([r["n_tokens"] for r in s]))}
        rep["step7_standalone"] = {"table": tab, "paired_bootstrap_vs_original": {}}
        if "original" in tab:
            for m in tab:
                if m in ("original", "target"):
                    continue
                for key, sel in (("math", lambda r: r["domain"] == "math"), ("short", lambda r: r["domain"] == "short")):
                    a = {r["id"]: r["correct"] for r in st if r["model"] == m and sel(r)}
                    b = {r["id"]: r["correct"] for r in st if r["model"] == "original" and sel(r)}
                    ids = sorted(set(a) & set(b))
                    if ids:
                        rep["step7_standalone"]["paired_bootstrap_vs_original"][f"{m}__{key}"] = paired_bootstrap_mean([a[i] for i in ids], [b[i] for i in ids], n_boot=n_boot)
    # deployment break-even: one-time cost (distillation + merge) / per-request saving
    if "step6_speculative" in rep:
        be = {}
        s6 = rep["step6_speculative"]
        for name in TRAINED:
            sf = c.adapter_dir(name) / "summary.json"
            if not sf.exists() or name not in s6:
                continue
            summ, mj = read_json(sf), read_json(c.adapter_dir(name) / "merge.json")
            be[name] = {}
            for g, e in s6[name]["gammas"].items():
                vs_target = s6[name]["target_e2e_s_per_request"] - e["e2e_s_per_request"]
                vs_orig = (s6["original"]["gammas"][g]["e2e_s_per_request"] - e["e2e_s_per_request"]) if "original" in s6 else None
                be[name][f"gamma{g}"] = {"vs_target_only": breakeven_requests(summ, mj["adapter_merge_s"], vs_target),
                                         "vs_original_draft": breakeven_requests(summ, mj["adapter_merge_s"], vs_orig) if vs_orig is not None else None}
        rep["deployment_break_even"] = be
    al = c.out / "alignment_summary.json"
    if al.exists():
        rep["step5_alignment_summary_file"] = str(al)
    rep["distillation"] = {n: read_json(c.adapter_dir(n) / "summary.json") for n in TRAINED if (c.adapter_dir(n) / "summary.json").exists()}
    write_json(c.out / "task3_report.json", rep)
    print("wrote", c.out / "task3_report.json")


STAGES = {"validate": stage_validate, "distill": stage_distill, "banks": stage_banks, "spec_bench": stage_spec_bench,
          "throughput": stage_throughput, "standalone": stage_standalone, "analyze": stage_analyze}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=[*STAGES, "all"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--data-root", default=data_root())
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--toy-n", type=int, default=100_000, help="sequences for the chi-square toy (spec: 100,000)")
    ap.add_argument("--mode", choices=MODES)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--drafts", help="comma-separated subset of: " + ",".join(DRAFTS))
    a = ap.parse_args(argv)
    c = Ctx(a)
    for s in (["validate", "distill", "banks", "spec_bench", "throughput", "standalone", "analyze"] if a.stage == "all" else [a.stage]):
        print(f"=== stage: {s}", flush=True)
        STAGES[s](c)


if __name__ == "__main__":
    main()
