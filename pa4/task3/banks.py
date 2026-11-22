"""Step 5: prefix-bank distribution alignment.

A bank record is (prompt, response prefix of ``position - 1`` tokens); the evaluated context is h = chat(prompt) + prefix and
the distributions are the next-token laws p_T(. | h) (target) and q(. | h) (draft), both at temperature 1 over the full vocabulary.
Metrics per record: forward KL(p||q), reverse KL(q||p), TV, greedy agreement (argmax p == argmax q) and expected speculative
acceptance 1 - TV. Records of one response share a prefix, so one forward pass per (bank, prompt) yields all of its positions.

Banks: staff ``teacher`` and ``original_draft`` banks (``prefix_banks_locked.jsonl.gz``, built by ``setup_data.py --prefix-banks``)
plus banks generated here with the same position-sampling rule: ``exact_spec_original_gamma4`` (responses produced by exact
speculative sampling, gamma = 4, original draft) and one ``student_<tag>`` bank per trained student (its own policy prefixes).
"""
from __future__ import annotations

import gzip
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..common.bootstrap import paired_bootstrap_mean
from .data import Item, build_chat_ids
from .lm import CachedLM
from .specdec import autoregressive_generate, speculative_generate

STRATA = [(1, 16, 8), (17, 32, 8), (33, 64, 16), (65, 192, 16)]
STRATUM_NAMES = ["[1,16]", "[17,32]", "[33,64]", "[65,192]"]


def stratum_of(pos: int) -> str:
    for (lo, hi, _), n in zip(STRATA, STRATUM_NAMES):
        if lo <= pos <= hi:
            return n
    raise ValueError(pos)


def load_banks(path) -> list[dict]:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_banks(path, recs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "wt", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")


def sample_positions(pid: str, bank: str, seq_len: int, seed: int = 6304) -> list[int]:
    rng = random.Random(f"PA4-banks-v1:{seed}:{pid}:{bank}")
    out = []
    for lo, hi, n in STRATA:
        cand = list(range(lo, min(hi, seq_len) + 1))
        out.extend(sorted(rng.sample(cand, min(n, len(cand)))))
    return out


def make_bank(bank: str, items: list[Item], tok, responder, device, seed: int = 6304, cap: int = 192) -> list[dict]:
    """``responder(prompt_ids, cap, gen) -> response tokens`` (a frozen sampler); one response per prompt."""
    recs = []
    for j, it in enumerate(items):
        gen = torch.Generator(device=device).manual_seed(seed + 10007 * j + 1)
        resp = responder(build_chat_ids(tok, it.prompt), cap, gen)
        for pos in sample_positions(it.id, bank, len(resp), seed):
            recs.append({"prompt_id": it.id, "domain": it.domain, "bank": bank, "position": pos, "prompt": it.prompt,
                         "response_prefix_token_ids": resp[: pos - 1], "source_length": len(resp), "source_generation_seed": seed + 10007 * j + 1})
    return recs


def autoregressive_responder(lm: CachedLM, eos_ids, device):
    return lambda prompt, cap, gen: autoregressive_generate(lm, prompt, cap, gen, eos_ids, 1.0, device, timed=False)[0]


def speculative_responder(target: CachedLM, draft: CachedLM, eos_ids, device, gamma: int = 4):
    return lambda prompt, cap, gen: speculative_generate(target, draft, prompt, gamma, cap, gen, eos_ids, 1.0, device=device, timed=False)[0]


@torch.no_grad()
def _rows(lm: CachedLM, ids: list[int], first_row: int) -> torch.Tensor:
    """Logits for positions first_row .. len(ids)-1 (row i = distribution after ids[i])."""
    n = len(ids) - first_row
    t = torch.tensor([ids], device=lm.device)
    pos = torch.arange(len(ids), device=lm.device)[None]
    return lm.engine.forward(t, pos, None, update=False, logits_last=n)[0].float()


def divergences(tl: torch.Tensor, ql: torch.Tensor) -> dict[str, torch.Tensor]:
    lp, lq = torch.log_softmax(tl, -1), torch.log_softmax(ql, -1)
    p, q = lp.exp(), lq.exp()
    tv = 0.5 * (p - q).abs().sum(-1)
    return {"kl_fwd": (p * (lp - lq)).sum(-1), "kl_rev": (q * (lq - lp)).sum(-1), "tv": tv,
            "greedy_agree": (tl.argmax(-1) == ql.argmax(-1)).float(), "accept": 1 - tv}


def evaluate_alignment(teacher: CachedLM, drafts: dict[str, CachedLM], tok, recs: list[dict]) -> list[dict]:
    """One row per (bank record, draft)."""
    groups = defaultdict(list)
    for r in recs:
        groups[(r["bank"], r["prompt_id"])].append(r)
    out = []
    for (bank, pid), rs in groups.items():
        rs = sorted(rs, key=lambda r: r["position"])
        prompt = build_chat_ids(tok, rs[0]["prompt"])
        longest = rs[-1]["response_prefix_token_ids"]
        for r in rs:                                                       # prefixes of one response must be nested
            assert longest[: len(r["response_prefix_token_ids"])] == r["response_prefix_token_ids"], (bank, pid)
        ids = prompt + list(longest)
        first = len(prompt) - 1
        tl = _rows(teacher, ids, first)
        for name, lm in drafts.items():
            ql = _rows(lm, ids, first)
            d = divergences(tl, ql)
            for r in rs:
                k = len(r["response_prefix_token_ids"])                    # row index of the distribution at this position
                out.append({"bank": bank, "prompt_id": pid, "domain": r["domain"], "position": r["position"], "stratum": stratum_of(r["position"]),
                            "draft": name, **{m: float(v[k]) for m, v in d.items()}})
            del ql, d
        del tl
    return out


METRICS = ["kl_fwd", "kl_rev", "tv", "greedy_agree", "accept"]


def summarize_alignment(rows: list[dict], baseline: str = "original", n_boot: int = 10000) -> dict:
    """Per (bank, draft): mean over records overall and per stratum; paired prompt-cluster bootstrap of (draft - baseline)."""
    res = {}
    banks = sorted({r["bank"] for r in rows})
    drafts = sorted({r["draft"] for r in rows})
    for b in banks:
        res[b] = {}
        for d in drafts:
            rr = [r for r in rows if r["bank"] == b and r["draft"] == d]
            if not rr:
                continue
            ent = {"n_records": len(rr), "overall": {m: float(np.mean([r[m] for r in rr])) for m in METRICS}, "by_stratum": {}}
            for s in STRATUM_NAMES:
                sr = [r for r in rr if r["stratum"] == s]
                if sr:
                    ent["by_stratum"][s] = {"n": len(sr), **{m: float(np.mean([r[m] for r in sr])) for m in METRICS}}
            if d != baseline:
                base = {(r["prompt_id"], r["position"]): r for r in rows if r["bank"] == b and r["draft"] == baseline}
                pids = sorted({r["prompt_id"] for r in rr})
                ent["vs_baseline_paired_bootstrap"] = {}
                for m in ("kl_fwd", "accept"):
                    a = [np.mean([r[m] for r in rr if r["prompt_id"] == p]) for p in pids]
                    c = [np.mean([base[(r["prompt_id"], r["position"])][m] for r in rr if r["prompt_id"] == p]) for p in pids]
                    ent["vs_baseline_paired_bootstrap"][m] = paired_bootstrap_mean(a, c, n_boot=n_boot) | {"resampling_unit": "prompt (records of a prompt averaged)"}
            res[b][d] = ent
    return res
