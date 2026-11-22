"""Task 3 data: GSM8K (locked layout), short-form JSONL, chat prompts, answer scoring, seeded round-robin quotas.

Prompt format (identical for target, student, banks and every benchmark): chat template with the *default* Qwen system
prompt, one user message holding the raw question / instruction, ``add_generation_prompt=True``, non-thinking (Qwen2.5 has no
thinking mode). Generation stops at ``<|im_end|>`` (Qwen2.5-Instruct generation_config EOS set), which is part of the scored
response; ``<|endoftext|>`` also terminates.
"""
from __future__ import annotations

import gzip
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


@dataclass
class Item:
    id: str
    domain: str               # 'math' | 'short'
    prompt: str
    answer: str               # gold final answer (string)
    family: int | None = None
    split: str | None = None  # short-form: train | familiar | shifted ; math: train | test


def _records(path):
    path = Path(path)
    op = gzip.open if path.name.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def gsm8k_gold(answer_field: str) -> str:
    return answer_field.split("####")[-1].strip().replace(",", "")


def load_gsm8k_locked(path, split: str) -> list[Item]:
    """``{order, source_row_index, example: {question, answer}}`` (staff locked layout), manifest order."""
    recs = sorted(_records(path), key=lambda r: r.get("order", 0))
    return [Item(f"GSM8K-{r.get('order', i):04d}", "math", r["example"]["question"], gsm8k_gold(r["example"]["answer"]), None, split)
            for i, r in enumerate(recs)]


def load_short(path) -> list[Item]:
    return [Item(r["id"], "short", r["input"], str(r["answer"]), r.get("family"), r.get("split")) for r in _records(path)]


def build_chat_ids(tok, prompt: str) -> list[int]:
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return list(tok(text, add_special_tokens=False)["input_ids"])


# ---------------------------------------------------------------------------------------------- scoring
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _norm_num(s: str) -> str | None:
    s = s.replace(",", "").rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    return str(int(v)) if v == int(v) else str(v)


def extract_math_answer(text: str) -> str | None:
    """\\boxed{...} if present, else '#### n', else the last number in the response."""
    m = re.findall(r"\\boxed\{([^{}]*)\}", text)
    cand = m[-1] if m else None
    if cand is None:
        h = re.findall(r"####\s*(-?[\d,\.]+)", text)
        cand = h[-1] if h else None
    if cand is None:
        nums = _NUM.findall(text)
        cand = nums[-1] if nums else None
    if cand is None:
        return None
    nums = _NUM.findall(cand)
    return _norm_num(nums[-1]) if nums else None


def score_math(text: str, gold: str) -> dict:
    pred = extract_math_answer(text)
    return {"pred": pred, "correct": int(pred is not None and pred == _norm_num(gold)), "format_ok": int(pred is not None)}


def short_format_ok(text: str, family: int) -> bool:
    """Format = the entire response is exactly the requested kind of answer (month name / integer / 2-letter code)."""
    t = text.strip().rstrip(".")
    if family == 0:
        return t in MONTHS
    if family == 1:
        return re.fullmatch(r"\d+", t) is not None
    if family == 2:
        return re.fullmatch(r"[A-Z]{2}", t) is not None
    return False


def score_short(text: str, gold: str, family: int) -> dict:
    t = text.strip().rstrip(".")
    return {"pred": t, "correct": int(t == gold), "format_ok": int(short_format_ok(text, family))}


# ---------------------------------------------------------------------------------------------- quotas
def round_robin(items: list[Item], seed: int, domain: str):
    """Endless iterator over a fixed seeded permutation of ``items`` (re-used after exhaustion; permutation reseeded per
    pass from (seed, domain, pass) so that reuse is not an identical order)."""
    rnd = 0
    while True:
        order = list(range(len(items)))
        random.Random(f"PA4-T3:{seed}:{domain}:{rnd}").shuffle(order)
        for i in order:
            yield items[i]
        rnd += 1


class QuotaSchedule:
    """Chooses the category (math/short) of the next training sample as the one with the largest remaining quota fraction
    (ties: math), so both quotas are exhausted together; same prompt order for every method given the seed."""

    def __init__(self, math_items, short_items, seed, quotas=None, caps=None):
        self.quota = dict(quotas or {"math": 30_000, "short": 10_000})
        self.total = dict(self.quota)
        self.caps = dict(caps or {"math": 192, "short": 32})
        self.its = {"math": round_robin(math_items, seed, "math"), "short": round_robin(short_items, seed, "short")}

    def exhausted(self) -> bool:
        return all(v <= 0 for v in self.quota.values())

    def next(self):
        dom = max(self.quota, key=lambda d: (self.quota[d] / self.total[d], d == "math"))
        if self.quota[dom] <= 0:
            dom = next(d for d in self.quota if self.quota[d] > 0)
        return next(self.its[dom]), self.caps[dom], self.quota[dom]

    def consume(self, domain: str, n: int):
        self.quota[domain] -= n
