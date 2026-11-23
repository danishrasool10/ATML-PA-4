"""Loaders for the data produced by ``atml-pa4-datasets`` (``python scripts/setup_data.py --public``).

Real layouts (verified against the dataset repo's ``scripts/setup_data.py`` / ``docs/DATA_FORMATS.md``; the
``data/public_locked/*.jsonl.gz`` files themselves exist only after a student runs ``--public`` with Hugging Face access):

* ``c4_calibration_locked.jsonl.gz`` (128) / ``c4_development_locked.jsonl.gz`` (32):
      {"source_row", "token_offset", "token_ids": [512 ints], "raw_text_sha256"}
* ``wikitext2_test_windows_locked.jsonl.gz`` / ``c4_validation_windows_locked.jsonl.gz`` (50,000 scored tokens each):
      {"source", "source_row", "token_offset", "token_ids": [<=512], "score_target_offsets": [1, n+1], "scored_tokens": n}
  one record = one independently evaluated window (the first token is context only; the last window of a
  quota may be truncated so scored_tokens == len(token_ids) - 1).
* ``hellaswag_val_locked.jsonl.gz`` (2,000, manifest order, first 512 = diagnostic subset):
      {"order", "source_row_index", "example": {Rowan/hellaswag row: ind, activity_label, ctx_a, ctx_b, endings, label, ...}}

The older flat layouts used by the synthetic test fixtures (``input_ids`` / ``text`` / flat HellaSwag rows) are accepted too.
JSON and JSONL, optionally gzip-compressed, are supported.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pa4common.results import sha256_file

LIST_KEYS = ("items", "records", "sequences", "windows", "examples", "ids", "data")


class ManifestError(ValueError):
    pass


def load_raw_manifest(path: str | Path) -> tuple[list[Any], dict[str, Any]]:
    """Return (records, meta) from a JSON / JSONL manifest."""
    path = Path(path)
    if not path.exists():
        raise ManifestError(f"manifest not found: {path}")
    name = path.name
    opener = (lambda p: gzip.open(p, "rt", encoding="utf-8")) if name.endswith(".gz") else (lambda p: open(p, encoding="utf-8"))
    if name.endswith(".jsonl") or name.endswith(".jsonl.gz"):
        with opener(path) as f:
            return [json.loads(line) for line in f if line.strip()], {}
    with opener(path) as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj, {}
    if isinstance(obj, dict):
        for k in LIST_KEYS:
            if k in obj and isinstance(obj[k], list):
                return obj[k], {kk: vv for kk, vv in obj.items() if kk != k}
    raise ManifestError(f"cannot find a record list in {path} (keys: {list(obj)[:10] if isinstance(obj, dict) else type(obj)})")


@dataclass
class TokenSeqSet:
    name: str
    ids: list[list[int]]
    record_ids: list[Any]
    meta: dict[str, Any] = field(default_factory=dict)
    sha256: str | None = None

    def __len__(self):
        return len(self.ids)


@dataclass
class PplWindowSet:
    name: str
    windows: list[list[int]]
    doc_ids: list[Any]
    meta: dict[str, Any] = field(default_factory=dict)
    sha256: str | None = None

    @property
    def n_scored_tokens(self) -> int:
        return sum(len(w) - 1 for w in self.windows)

    def __len__(self):
        return len(self.windows)


@dataclass
class HellaSwagSet:
    name: str
    examples: list[dict[str, Any]]
    ids: list[Any]
    meta: dict[str, Any] = field(default_factory=dict)
    sha256: str | None = None

    def __len__(self):
        return len(self.examples)


def _ids_of(rec: dict, key_options=("input_ids", "token_ids", "tokens")):
    for k in key_options:
        if k in rec:
            return [int(t) for t in rec[k]]
    return None


def load_token_sequences(path, tokenizer=None, *, name: str, expected_n: int | None = None, seq_len: int = 512) -> TokenSeqSet:
    records, meta = load_raw_manifest(path)
    ids, rids = [], []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ManifestError(f"{name}: record {i} is not an object: {type(rec)}")
        toks = _ids_of(rec)
        if toks is None:
            if "text" not in rec or tokenizer is None:
                raise ManifestError(f"{name}: record {i} has neither token ids nor text (+tokenizer): {list(rec)}")
            start = int(rec.get("token_start", rec.get("start_token", 0)))
            toks = tokenizer.encode(rec["text"], add_special_tokens=False)[start : start + seq_len]
        if len(toks) != seq_len:
            raise ManifestError(f"{name}: record {i} has {len(toks)} tokens, expected {seq_len}")
        ids.append(toks)
        rids.append(rec.get("id", rec.get("seq_id", rec.get("source_row", i))))
    if expected_n is not None and len(ids) != expected_n:
        raise ManifestError(f"{name}: expected {expected_n} sequences, found {len(ids)}")
    return TokenSeqSet(name, ids, rids, meta, sha256_file(path))


def chunk_document(tokens: list[int], window_len: int = 512) -> list[list[int]]:
    """Non-overlapping prediction windows; a trailing window needs >= 2 tokens to score anything."""
    out = [tokens[i : i + window_len] for i in range(0, len(tokens), window_len)]
    return [w for w in out if len(w) >= 2]


def load_ppl_windows(path, tokenizer=None, *, name: str, window_len: int = 512, expected_scored: int | None = None) -> PplWindowSet:
    records, meta = load_raw_manifest(path)
    windows, dids = [], []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ManifestError(f"{name}: record {i} is not an object: {type(rec)}")
        doc_id = rec.get("doc_id", rec.get("id", f'{rec["source_row"]}:{rec.get("token_offset", 0)}' if "source_row" in rec else i))
        toks = _ids_of(rec)
        if toks is not None and len(toks) <= window_len:
            if "scored_tokens" in rec and int(rec["scored_tokens"]) != len(toks) - 1:
                raise ManifestError(f"{name}: record {i} scored_tokens {rec['scored_tokens']} != len(token_ids)-1")
            if len(toks) >= 2:
                windows.append(toks)
                dids.append(doc_id)
            continue
        if toks is None:
            if "text" not in rec or tokenizer is None:
                raise ManifestError(f"{name}: record {i} has neither token ids nor text (+tokenizer): {list(rec)}")
            toks = tokenizer.encode(rec["text"], add_special_tokens=False)  # each document tokenized separately
        for w in chunk_document(toks, window_len):
            windows.append(w)
            dids.append(doc_id)
    ws = PplWindowSet(name, windows, dids, meta, sha256_file(path))
    declared = meta.get("n_scored_tokens", expected_scored)
    if declared is not None and ws.n_scored_tokens != int(declared):
        raise ManifestError(f"{name}: scored tokens {ws.n_scored_tokens} != expected {declared}")
    return ws


def load_hellaswag(path, *, name: str = "hellaswag", expected_n: int | None = None, hf_resolver=None) -> HellaSwagSet:
    records, meta = load_raw_manifest(path)
    if records and isinstance(records[0], dict) and "example" in records[0]:      # locked layout: {"order", "example": {...}}
        records = sorted(records, key=lambda r: r.get("order", 0))
        records = [dict(r["example"], order=r.get("order"), source_row_index=r.get("source_row_index")) for r in records]
    if records and not isinstance(records[0], dict):
        # list of ids: resolve through the HF dataset (validation split), preserving manifest order
        resolver = hf_resolver or _resolve_hellaswag_from_hf
        examples = resolver([int(r) for r in records])
    else:
        examples = records
    ids = [e.get("ind", e.get("id", e.get("source_row_index", i))) for i, e in enumerate(examples)]
    for i, e in enumerate(examples):
        for k in ("activity_label", "ctx_a", "ctx_b", "endings", "label"):
            if k not in e:
                raise ManifestError(f"{name}: example {i} lacks field {k!r}")
        if len(e["endings"]) != 4:
            raise ManifestError(f"{name}: example {i} has {len(e['endings'])} endings")
    if expected_n is not None and len(examples) != expected_n:
        raise ManifestError(f"{name}: expected {expected_n} examples, found {len(examples)}")
    return HellaSwagSet(name, examples, ids, meta, sha256_file(path))


def _resolve_hellaswag_from_hf(inds: list[int]) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split="validation")
    by_ind = {int(r["ind"]): r for r in ds}
    missing = [i for i in inds if i not in by_ind]
    if missing:
        raise ManifestError(f"HellaSwag ids not in validation split: {missing[:5]}")
    return [dict(by_ind[i]) for i in inds]
