"""Staff retrieval cases for Part C: loading, chat-template prompt construction, token-level splits and scoring.

Record fields (``kv_retrieval_cases.jsonl.gz``): id, user_text, code, position, distractors, target_tokens, seed,
evidence_text, evidence_char_start, evidence_char_end (Python string offsets into ``user_text``).

Prompt = chat template applied to ``[{"role": "user", "content": user_text}]`` with the generation prompt. The prompt is
cut at the first token of the ``QUESTION_BEGIN`` marker:
  * ``prefix``  = everything before it  (processed and cached *before* the question is known: this is the cache that
                  gets compressed, and whose size defines L in the budget formula),
  * ``suffix``  = ``QUESTION_BEGIN ... <|im_start|>assistant\\n`` (the question / observation suffix).
Gold evidence offsets are used ONLY to measure evidence retention; they are never given to a retention policy.
"""
from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path

MARKER = "QUESTION_BEGIN"
CODE_RE = re.compile(r"(?<!\d)\d{5}(?!\d)")


def load_cases(path) -> list[dict]:
    path = Path(path)
    op = gzip.open if path.name.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_code(text: str) -> str | None:
    """Deterministic extractor: the first standalone run of exactly five digits in the decoded answer."""
    m = CODE_RE.search(text)
    return m.group(0) if m else None


def case_cell(rec: dict) -> tuple:
    return (rec["target_tokens"], rec["position"], rec["distractors"])


def case_split(rec: dict) -> str:
    """Case 0 of every (context length, location, distractors) cell is the development case (18 of 180); the other
    nine per cell are final-test cases. Nothing is tuned on either; dev is used for pipeline debugging only."""
    return "dev" if rec["id"].rsplit("-", 1)[-1] == "00" else "test"


@dataclass
class Prompt:
    id: str
    ids: list[int]                 # full prompt token ids
    n_prefix: int                  # L: tokens before the question marker
    evidence_tokens: list[int]     # indices of prefix tokens overlapping the gold evidence span (evaluation only)
    code: str
    cell: tuple
    split: str

    @property
    def prefix_ids(self):
        return self.ids[: self.n_prefix]

    @property
    def suffix_ids(self):
        return self.ids[self.n_prefix:]


def build_prompt(rec: dict, tokenizer) -> Prompt:
    user = rec["user_text"]
    text = tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    off = text.index(user)                                   # where the user content starts inside the rendered template
    marker = off + user.index(MARKER)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    n_prefix = next(i for i, (s, e) in enumerate(offs) if e > marker)       # first token reaching into the marker
    ev_s, ev_e = off + rec["evidence_char_start"], off + rec["evidence_char_end"]
    ev = [i for i, (s, e) in enumerate(offs) if s < ev_e and e > ev_s and i < n_prefix]
    return Prompt(rec["id"], ids, n_prefix, ev, rec["code"], case_cell(rec), case_split(rec))
