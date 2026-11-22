"""Steps 2, 6, 7: speculative-decoding benchmark, analytical speed-up model, throughput diagnostic, standalone quality.

Protocol (Step 6): for each draft and each of the 120 held-out prompts (60 GSM8K test + 30 familiar + 30 shifted short-form),
2 paired repetitions, temperature 1, top_p 1, 64-new-token EOS-aware cap, a *block* = {target-only, spec gamma=2, spec gamma=4}
run back to back on the same prompt in counterbalanced order (forward order for even (prompt + rep), reversed for odd), so every
speculative run has an adjacent target-only timing. Timers are CUDA-synchronised: draft proposal passes, single target
verification pass, acceptance/residual sampling, cache rollback/advancement, plus prefill (host overhead = remainder).

Analytical model (decode only, per draft x gamma):   S_hat = (1 + sum_{j=1..gamma} alpha^j) / (gamma c + v_gamma)
  c       = (draft pass time) / (target-only 1-token decode step time)
  v_gamma = (target verification pass time, gamma+1 tokens) / (target-only 1-token decode step time)
  alpha   = MLE of the per-token acceptance probability (accepted / (accepted + rejection events))
  break-even alpha: the alpha at which S_hat = 1.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..common.bootstrap import paired_bootstrap_mean
from .data import Item, build_chat_ids, score_math, score_short
from .lm import CachedLM
from .specdec import autoregressive_generate, speculative_generate

GAMMAS = (2, 4)


def _gen(device, seed, i, rep, kind):
    return torch.Generator(device=device).manual_seed(seed * 1_000_003 + i * 101 + rep * 7 + kind)


def run_spec_benchmark(target: CachedLM, drafts: dict[str, CachedLM], tok, items: list[Item], eos_ids, device, out: Path, *, gammas=GAMMAS,
                       reps: int = 2, max_new: int = 64, ignore_eos: bool = False, seed: int = 6304, tag: str = "step6", warmup: int = 2) -> list[dict]:
    """Appends one JSONL record per run to ``out`` (resumable). Returns all records."""
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    recs = []
    if out.exists():
        for l in open(out):
            r = json.loads(l)
            recs.append(r)
            done.add((r["draft"], r["id"], r["rep"], r["kind"], r["gamma"]))
    eos = frozenset() if ignore_eos else frozenset(eos_ids)
    for it in items[:warmup]:                                                       # warm up every path (CUDA kernels, allocator)
        p = build_chat_ids(tok, it.prompt)
        for name, d in drafts.items():
            autoregressive_generate(target, p, 8, _gen(device, seed, 0, 99, 0), eos, 1.0, device, timed=False)
            speculative_generate(target, d, p, max(gammas), 8, _gen(device, seed, 0, 99, 1), eos, 1.0, device=device, timed=False)
    t0 = time.time()
    with open(out, "a") as f:
        for name, d in drafts.items():
            for i, it in enumerate(items):
                prompt = build_chat_ids(tok, it.prompt)
                for rep in range(reps):
                    runs = [("target", 0)] + [("spec", g) for g in gammas]
                    if (i + rep) % 2 == 1:
                        runs = runs[::-1]
                    for pos, (kind, g) in enumerate(runs):
                        key = (name, it.id, rep, kind, g)
                        if key in done:
                            continue
                        gen = _gen(device, seed, i, rep, g)
                        if kind == "target":
                            toks, st = autoregressive_generate(target, prompt, max_new, gen, eos, 1.0, device, ignore_eos=ignore_eos)
                            rec = {**st}
                        else:
                            toks, st = speculative_generate(target, d, prompt, g, max_new, gen, eos, 1.0, device=device)
                            rec = {k: v for k, v in st.__dict__.items() if k != "round_lengths"}
                            rec["alpha_mle"] = st.alpha_mle
                            rec["tokens_per_round"] = float(np.mean(st.round_lengths)) if st.round_lengths else 0.0
                            rec["n_tokens"] = len(toks)
                        rec.update({"tag": tag, "draft": name, "id": it.id, "domain": it.domain, "split": it.split, "rep": rep, "kind": kind,
                                    "gamma": g, "order_in_block": pos, "block_reversed": (i + rep) % 2 == 1, "n_prompt": len(prompt)})
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                        recs.append(rec)
            print(f"[{tag}] draft={name} done ({time.time() - t0:.0f}s)", flush=True)
    return recs


def breakeven_alpha(gamma: int, c: float, v: float) -> float | None:
    f = lambda a: 1 + sum(a ** j for j in range(1, gamma + 1)) - (gamma * c + v)
    if f(0.0) >= 0:
        return 0.0
    if f(1.0) < 0:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


def analyze_spec(recs: list[dict], n_boot: int = 10000, seed: int = 6304) -> dict:
    res = {}
    by = defaultdict(list)
    for r in recs:
        by[(r["draft"], r["gamma"], r["kind"])].append(r)
    drafts = sorted({r["draft"] for r in recs})
    for d in drafts:
        tgt = [r for r in recs if r["draft"] == d and r["kind"] == "target"]
        step_t = sum(r["t_decode_s"] for r in tgt) / max(sum(r["decode_passes"] for r in tgt), 1)      # target 1-token decode step (s)
        res[d] = {"target_decode_step_ms": step_t * 1e3, "target_e2e_s_per_request": float(np.mean([r["t_total_s"] for r in tgt])),
                  "target_tokens_per_request": float(np.mean([r["n_tokens"] for r in tgt])), "gammas": {}}
        for g in sorted({r["gamma"] for r in recs if r["kind"] == "spec"}):
            sp = [r for r in recs if r["draft"] == d and r["kind"] == "spec" and r["gamma"] == g]
            if not sp:
                continue
            A, R = sum(r["accepted"] for r in sp), sum(r["rejections"] for r in sp)
            alpha = A / max(A + R, 1)
            step_d = sum(r["t_draft_s"] for r in sp) / max(sum(r["draft_passes"] for r in sp), 1)
            step_v = sum(r["t_verify_s"] for r in sp) / max(sum(r["target_passes"] for r in sp), 1)
            c, v = step_d / step_t, step_v / step_t
            S_hat = (1 + sum(alpha ** j for j in range(1, g + 1))) / (g * c + v)
            # paired blocks: spec run vs the target-only run of the same (prompt, rep) block
            tmap = {(r["id"], r["rep"]): r for r in tgt}
            pairs = [(r, tmap[(r["id"], r["rep"])]) for r in sp if (r["id"], r["rep"]) in tmap]
            dec_s = np.array([(r["t_total_s"] - r["t_prefill_s"]) / max(r["n_tokens"], 1) for r, _ in pairs])
            dec_t = np.array([t["t_decode_s"] / max(t["n_tokens"], 1) for _, t in pairs])
            e2e_s = np.array([r["t_total_s"] / max(r["n_tokens"], 1) for r, _ in pairs])
            e2e_t = np.array([t["t_total_s"] / max(t["n_tokens"], 1) for _, t in pairs])
            timers = {k: float(sum(r[k] for r in sp)) for k in ("t_prefill_s", "t_draft_s", "t_verify_s", "t_accept_s", "t_cache_s", "t_total_s")}
            timers["t_host_other_s"] = timers["t_total_s"] - sum(timers[k] for k in ("t_prefill_s", "t_draft_s", "t_verify_s", "t_accept_s", "t_cache_s"))
            res[d]["gammas"][str(g)] = {
                "n_runs": len(sp), "alpha_mle": alpha, "alpha_naive_accepted_over_proposed": sum(r["accepted"] for r in sp) / max(sum(r["proposed"] for r in sp), 1),
                "tokens_per_round": float(np.mean([r["tokens_per_round"] for r in sp])), "c": c, "v_gamma": v, "S_hat_analytic": S_hat,
                "S_measured_decode": float(np.sum(dec_t) / np.sum(dec_s)), "S_measured_e2e": float(np.sum(e2e_t) / np.sum(e2e_s)),
                "S_measured_decode_bootstrap_ratio_of_means": _ratio_ci(dec_t, dec_s, n_boot, seed),
                "break_even_alpha": breakeven_alpha(g, c, v), "timers_total_s": timers,
                "e2e_s_per_request": float(np.mean([r["t_total_s"] for r in sp])), "tokens_per_request": float(np.mean([r["n_tokens"] for r in sp])),
                "paired_speedup_per_block_e2e_mean": float(np.mean(e2e_t / e2e_s))}
    return res


def _ratio_ci(num, den, n_boot, seed, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(num)
    idx = rng.integers(0, n, size=(n_boot, n))
    r = num[idx].sum(1) / den[idx].sum(1)
    return {"estimate": float(num.sum() / den.sum()), "ci_low": float(np.percentile(r, 100 * alpha / 2)), "ci_high": float(np.percentile(r, 100 * (1 - alpha / 2))),
            "n": int(n), "n_boot": n_boot, "resampling": "paired blocks (sum of per-token decode time ratios)"}


def standalone_eval(lm: CachedLM, tok, items: list[Item], eos_ids, device, caps=None, seed: int = 6304, greedy: bool = True) -> list[dict]:
    """Greedy standalone generation (512 tokens math / 32 tokens short-form), scored with the deterministic extractors."""
    caps = caps or {"math": 512, "short": 32}
    out = []
    for i, it in enumerate(items):
        prompt = build_chat_ids(tok, it.prompt)
        gen = torch.Generator(device=device).manual_seed(seed + i)
        toks, st = autoregressive_generate(lm, prompt, caps[it.domain], gen, frozenset(eos_ids), 0.0 if greedy else 1.0, device, timed=False)
        finished = bool(toks) and toks[-1] in eos_ids
        text = tok.decode([t for t in toks if t not in eos_ids], skip_special_tokens=True)
        sc = score_math(text, it.answer) if it.domain == "math" else score_short(text, it.answer, it.family)
        out.append({"id": it.id, "domain": it.domain, "split": it.split, "n_tokens": len(toks), "finished": int(finished), "text": text, **sc})
    return out


def breakeven_requests(distill_summary: dict, merge_s: float, saving_per_request_s: float) -> dict:
    t = distill_summary["times_s"]
    cost = sum(t.values()) + merge_s
    return {"one_time_cost_s": cost, "cost_breakdown_s": {**t, "adapter_merge_s": merge_s}, "saving_per_request_s": saving_per_request_s,
            "break_even_requests": (cost / saving_per_request_s) if saving_per_request_s > 0 else None}
