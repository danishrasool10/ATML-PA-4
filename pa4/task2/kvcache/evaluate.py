"""Part C driver: C1 reference validation, per-prompt evaluation of the 5 compressed-cache methods (C2-C5), analysis.

    python -m pa4.task2.kvcache.evaluate --cases ../atml-pa4-datasets/data/kv_retrieval/kv_retrieval_cases.jsonl.gz \
        --out results/task2_partC [--limit N] [--smoke]

Per prompt (one pass over a shared FP16 prefill): reference + 5 methods, each with
  teacher-forced pass over the gold code (answer NLL, post-prefill KL, mean KL over the answer positions),
  free-running greedy decoding (exact match of the deterministic 5-digit extractor),
  realized packed bytes, evidence retention, K/V reconstruction error (quantizing methods),
  scoring / compress / suffix+decode times and peak allocated GPU memory.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ...common.bootstrap import paired_bootstrap_mean
from ...common.env_meta import collect_environment
from ...common.results import read_jsonl, write_json
from ...common.seed import DEFAULT_SEED, set_seed
from ...common.timing import Stopwatch
from .data import Prompt, build_prompt, case_cell, extract_code, load_cases
from .engine import KVEngine
from .policies import METHODS, NEEDS_SCORES, _select, compress

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW = 12
N_BOOT = 10000                     # >= 2,000 required


def _kl(ref_logits, logits):
    lp, lq = torch.log_softmax(ref_logits, -1), torch.log_softmax(logits, -1)
    return (lp.exp() * (lp - lq)).sum(-1)


def _peak_reset(device):
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak(device):
    return torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else None


def eos_ids_of(tok, model) -> set[int]:
    ids = set()
    e = getattr(model.generation_config, "eos_token_id", None)
    for x in ([e] if isinstance(e, int) else (e or [])):
        ids.add(int(x))
    if getattr(tok, "eos_token_id", None) is not None:
        ids.add(int(tok.eos_token_id))
    return ids


@torch.no_grad()
def evaluate_prompt(engine: KVEngine, tok, pr: Prompt, eos: set[int], methods=("fp16", *METHODS)) -> list[dict]:
    dev = engine.device
    prefix = torch.tensor([pr.prefix_ids], device=dev)
    suffix = torch.tensor([pr.suffix_ids], device=dev)
    ans = torch.tensor([tok.encode(pr.code, add_special_tokens=False)], device=dev)
    L = pr.n_prefix
    dims = (engine.n_layers, engine.Hkv, engine.d)
    with Stopwatch(dev) as sw:
        _, ref_cache = engine.prefill(prefix)
    t_prefill = sw.elapsed_s
    scores, t_score = None, 0.0
    if any(m in NEEDS_SCORES for m in methods):
        with Stopwatch(dev) as sw:
            scores = engine.score_suffix(suffix, ref_cache, L)
        t_score = sw.elapsed_s
    ref_tf = None
    recs = []
    ev = set(pr.evidence_tokens)
    for m in methods:
        _peak_reset(dev)
        base = ref_cache.clone()
        with Stopwatch(dev) as sw:
            c = compress(m, base, scores, dims)
        t_comp = sw.elapsed_s
        tf = engine.teacher_forced(suffix, ans, c.cache.clone(), L)
        if m == "fp16":
            ref_tf = tf
        with Stopwatch(dev) as sw:
            gen, _ = engine.run_suffix_and_generate(suffix, c.cache, L, MAX_NEW, eos)
        t_dec = sw.elapsed_s
        text = tok.decode([g for g in gen if g not in eos])
        pred = extract_code(text)
        n_ans = ans.shape[1]
        lp = torch.log_softmax(tf[:n_ans], -1)
        nll = float(-lp.gather(1, ans[0][:, None]).sum())
        kl = _kl(ref_tf[:n_ans], tf[:n_ans])
        recs.append({
            "id": pr.id, "method": m, "split": pr.split, "target_tokens": pr.cell[0], "position": pr.cell[1], "distractors": pr.cell[2],
            "n_prefix_tokens": L, "n_suffix_tokens": len(pr.suffix_ids), "n_evidence_tokens": len(ev),
            "answer": text, "pred": pred, "gold": pr.code, "correct": int(pred == pr.code),
            "nll_gold": nll, "kl_post_prefill": float(kl[0]), "kl_tf_mean": float(kl.mean()),
            "packed_bytes": c.packed_bytes, "budget_bytes": c.budget_bytes, "fp16_bytes": int(4 * dims[0] * dims[1] * dims[2] * L),
            "evidence_retained_frac": (len(ev & set(c.kept_any)) / len(ev)) if ev and c.kept_any is not None else None,
            "evidence_fp16_frac": (len(ev & set(c.kept_fp16)) / len(ev)) if ev and c.kept_fp16 is not None else None,
            "k_rel_err": c.k_err, "v_rel_err": c.v_err, "info": c.info,
            "t_prefill_s": t_prefill, "t_score_s": t_score if m in NEEDS_SCORES else 0.0, "t_compress_s": t_comp, "t_suffix_decode_s": t_dec,
            "peak_alloc_bytes": _peak(dev),
        })
    return recs


# ------------------------------------------------------------------------------------------ C1
@torch.no_grad()
def validate_reference(engine: KVEngine, tok, prompts: list[Prompt], eos: set[int], n_gen: int = 6) -> dict:
    """C1: engine vs the unmodified HF model. Compares (a) last-position logits of a direct full forward, an engine
    one-shot forward and split prefill + suffix; (b) greedy generation of the engine reference vs ``generate``;
    (c) 100% retention (identity selection) and reversed physical cache order, which must reproduce the logits because
    keys carry their absolute rotary positions."""
    dev = engine.device
    rows = []
    for pr in prompts[:n_gen]:
        ids = torch.tensor([pr.ids], device=dev)
        pre, suf = ids[:, : pr.n_prefix], ids[:, pr.n_prefix:]
        hf = engine.model(ids).logits[0, -1].float()
        c0 = engine.empty_cache(1)
        pos = torch.arange(ids.shape[1], device=dev)[None]
        one = engine.forward(ids, pos, c0, update=False)[0, -1].float()
        _, cache = engine.prefill(pre)
        split = engine.forward(suf, torch.arange(pr.n_prefix, ids.shape[1], device=dev)[None], cache.clone(), update=True)[0, -1].float()
        keep_all = _select(cache, torch.arange(pr.n_prefix, device=dev))
        full = engine.forward(suf, torch.arange(pr.n_prefix, ids.shape[1], device=dev)[None], keep_all, update=True)[0, -1].float()
        rev = _select(cache, torch.arange(pr.n_prefix - 1, -1, -1, device=dev))
        rev_l = engine.forward(suf, torch.arange(pr.n_prefix, ids.shape[1], device=dev)[None], rev, update=True)[0, -1].float()
        gen_hf = engine.model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False, eos_token_id=list(eos) or None,
                                       pad_token_id=next(iter(eos)) if eos else 0)[0, ids.shape[1]:].tolist()
        gen_en, _ = engine.run_suffix_and_generate(suf, cache.clone(), pr.n_prefix, MAX_NEW, eos)
        n = min(len(gen_hf), len(gen_en))
        rows.append({"id": pr.id,
                     "maxabs_hf_vs_oneshot": float((hf - one).abs().max()), "maxabs_hf_vs_split": float((hf - split).abs().max()),
                     "maxabs_split_vs_keepall": float((split - full).abs().max()), "maxabs_split_vs_reversed": float((split - rev_l).abs().max()),
                     "top1_agree": bool(hf.argmax() == split.argmax() == full.argmax() == rev_l.argmax()),
                     "generation_identical": gen_hf[:n] == gen_en[:n] and abs(len(gen_hf) - len(gen_en)) <= 1})
    return {"n": len(rows), "rows": rows,
            "all_top1_agree": all(r["top1_agree"] for r in rows), "all_generations_identical": all(r["generation_identical"] for r in rows),
            "max_maxabs_hf_vs_split": max(r["maxabs_hf_vs_split"] for r in rows) if rows else None}


# ------------------------------------------------------------------------------------------ C5 analysis
def _col(recs, method, key, subset=None):
    by = {r["id"]: r for r in recs if r["method"] == method and (subset is None or r["split"] in subset)}
    return by


def analyze(recs: list[dict], n_boot: int = N_BOOT) -> dict:
    out = {"n_boot": n_boot, "subsets": {}}
    ids_all = sorted({r["id"] for r in recs})
    for sname, sub in (("all", None), ("test_only", {"test"})):
        by = {m: _col(recs, m, "", sub) for m in ["fp16", *METHODS] if any(r["method"] == m for r in recs)}
        if not by.get("fp16"):
            continue                                   # e.g. no final-test prompts in a tiny smoke subset
        ids = sorted(by["fp16"]) if "fp16" in by else ids_all
        table, boots = {}, {}
        ref_ok = np.array([by["fp16"][i]["correct"] for i in ids], float)
        for m, d in by.items():
            g = lambda k: np.array([d[i][k] for i in ids], float)
            row = {"n": len(ids), "exact_match": float(g("correct").mean()),
                   "exact_match_given_fp16_correct": float(g("correct")[ref_ok == 1].mean()) if ref_ok.sum() else None,
                   "nll_gold": float(g("nll_gold").mean()), "kl_post_prefill": float(g("kl_post_prefill").mean()),
                   "kl_tf_mean": float(g("kl_tf_mean").mean()), "packed_bytes_mean": float(g("packed_bytes").mean()),
                   "budget_bytes_mean": float(g("budget_bytes").mean()), "fp16_bytes_mean": float(g("fp16_bytes").mean()),
                   "t_prefill_s": float(g("t_prefill_s").mean()), "t_score_s": float(g("t_score_s").mean()),
                   "t_compress_s": float(g("t_compress_s").mean()), "t_suffix_decode_s": float(g("t_suffix_decode_s").mean())}
            for k in ("evidence_retained_frac", "evidence_fp16_frac", "k_rel_err", "v_rel_err", "peak_alloc_bytes"):
                v = [d[i][k] for i in ids if d[i][k] is not None]
                row[k] = float(np.mean(v)) if v else None
            table[m] = row
        cells = defaultdict(lambda: defaultdict(list))
        for m, d in by.items():
            for i in ids:
                r = d[i]
                cells[m][f"{r['target_tokens']}|{r['position']}|{r['distractors']}"].append(r["correct"])
        table_cells = {m: {c: float(np.mean(v)) for c, v in sorted(cs.items())} for m, cs in cells.items()}

        def diff(a, b, key):
            x = np.array([by[a][i][key] for i in ids], float)
            y = np.array([by[b][i][key] for i in ids], float)
            return paired_bootstrap_mean(x, y, n_boot=n_boot)

        pairs = [(m, "fp16") for m in METHODS if m in by] + [("question_aware", "sink_recent"), ("int4_channel", "int4_token"),
                                                             ("hybrid", "int4_channel"), ("hybrid", "question_aware")]
        for a, b in pairs:
            if a in by and b in by:
                boots[f"{a}__minus__{b}"] = {k: diff(a, b, k) for k in ("correct", "nll_gold", "kl_post_prefill")}
                if ref_ok.sum():
                    sel = np.array([by[a][i]["correct"] for i in ids], float)[ref_ok == 1]
                    sel_b = np.array([by[b][i]["correct"] for i in ids], float)[ref_ok == 1]
                    boots[f"{a}__minus__{b}"]["correct_given_fp16_correct"] = paired_bootstrap_mean(sel, sel_b, n_boot=n_boot)
        out["subsets"][sname] = {"n": len(ids), "methods": table, "per_cell_exact_match": table_cells, "paired_bootstrap": boots}
    return out


def write_markdown(res: dict, path: Path):
    with open(path, "w") as f:
        for sname, s in res["subsets"].items():
            f.write(f"### {sname} (n={s['n']})\n\n| method | EM | EM|fp16✓ | NLL(gold) | KL post-prefill | KL tf | evid. kept | evid. FP16 | MB (realized) | MB (budget) | K err | V err |\n|" + "---|" * 12 + "\n")
            for m, r in s["methods"].items():
                fm = lambda v, p=3: "-" if v is None else f"{v:.{p}f}"
                f.write(f"| {m} | {r['exact_match']:.3f} | {fm(r['exact_match_given_fp16_correct'])} | {r['nll_gold']:.3f} | {r['kl_post_prefill']:.4f} | "
                        f"{r['kl_tf_mean']:.4f} | {fm(r['evidence_retained_frac'])} | {fm(r['evidence_fp16_frac'])} | {r['packed_bytes_mean']/2**20:.2f} | "
                        f"{r['budget_bytes_mean']/2**20:.2f} | {fm(r['k_rel_err'])} | {fm(r['v_rel_err'])} |\n")
            f.write("\n")


def run(cases, tok, model, out: Path, limit: int | None = None, n_boot: int = N_BOOT, validate_n: int = 6, resume: bool = True):
    out.mkdir(parents=True, exist_ok=True)
    set_seed(DEFAULT_SEED)
    engine = KVEngine(model)
    eos = eos_ids_of(tok, model)
    prompts = [build_prompt(r, tok) for r in (cases[:limit] if limit else cases)]
    write_json(out / "prompt_audit.json", [{"id": p.id, "n_prefix": p.n_prefix, "n_suffix": len(p.suffix_ids), "n_total": len(p.ids),
                                            "n_evidence_tokens": len(p.evidence_tokens), "split": p.split} for p in prompts])
    f = out / "per_example.jsonl"
    done = {(r["id"], r["method"]) for r in read_jsonl(f)} if resume and f.exists() else set()
    if validate_n and not (out / "c1_validation.json").exists():
        write_json(out / "c1_validation.json", validate_reference(engine, tok, prompts, eos, validate_n))
    t0 = time.time()
    for k, pr in enumerate(prompts):
        if all((pr.id, m) in done for m in ("fp16", *METHODS)):
            continue
        recs = evaluate_prompt(engine, tok, pr, eos)
        with open(f, "a") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        print(f"[{k + 1}/{len(prompts)}] {pr.id} L={pr.n_prefix} {' '.join(str(r['correct']) for r in recs)}  ({time.time() - t0:.0f}s)", flush=True)
    recs = read_jsonl(f)
    res = analyze(recs, n_boot)
    res["env"] = collect_environment(model, tok)
    res["note"] = ("Matched budget M_4(L) at pre-question compression; shared retention across layers/heads, 2 B per retained index; "
                   "quantization applied when 64 aligned slots are full; hybrid keeps new tokens in FP16.")
    write_json(out / "partC_summary.json", res)
    write_markdown(res, out / "partC_summary.md")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", default="results/task2_partC")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--validate-n", type=int, default=6)
    ap.add_argument("--smoke", action="store_true", help="tiny random Qwen2 + byte tokenizer (CPU, no downloads)")
    a = ap.parse_args()
    cases = load_cases(a.cases)
    if a.smoke:
        from ...data.tiny import ByteChatTokenizer, make_tiny_model
        tok, model = ByteChatTokenizer(), make_tiny_model(layers=2, hidden=128, heads=2, kv_heads=1)
        dev = "cpu"
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, attn_implementation="sdpa").to(dev).eval()
    run(cases, tok, model, Path(a.out), a.limit, a.n_boot, a.validate_n)


if __name__ == "__main__":
    main()
