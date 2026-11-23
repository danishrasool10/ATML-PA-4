"""Task 2 Part B: multi-query associative recall (MQAR) data access, invariants and metrics.

Uses the staff-supplied generator (vendored unchanged in ``pa4/data/mqar_generator.py``) and the frozen splits of
``atml-pa4-datasets``: ``data/mqar/mqar_test.jsonl.gz`` (9,216 = 1,024 per K x regime) and
``data/mqar/mqar_development.jsonl.gz`` (2,304 = 256 per K x regime). Both files are exactly
``make_example(split, K, regime, index)``; if a file is missing the identical sequences are regenerated and when it is
present it is verified against the generator.  Training sequences: ``make_example('train', K, regime, index)`` with
index ranges disjoint across seeds.

Sequence: T=256, keys 64..127 and values 128..191 at positions 0..2K-1 (k1 v1 ... kK vK), filler 2..63, K queries (the
key token is repeated) in disjoint later slots; the supervised target (the paired value) is labelled *at the query
position* (query-position classification, not next-token LM). ``distance = query position - key position``.
Regimes in the supplied generator: near 1<=d<=63, far 128<=d<=250, mixed 1<=d<=250 (uniform over feasible slots).
Loss: CE over the 64 value-ID logits (128..191) at the K query positions (ASSUMPTION for "classification logits":
the alternative restricted to the K stored values is available via ``loss_over="stored_k"``); accuracy is reported
both 64-way (chance 1/64) and restricted to the K stored values (chance 1/K).
"""
from __future__ import annotations

import gzip
import json
import os
import queue
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pa4common.paths import data_root
from .mqar_generator import T as T_DEFAULT
from .mqar_generator import make_example, validate

VOCAB = 256
KEY_LO, KEY_HI = 64, 128
VAL_LO, VAL_HI = 128, 192
BINS = ((0, 32), (32, 64), (64, 128), (128, 256))
REGIMES = ("near", "far", "mixed")
KS = (4, 8, 16)
N_TEST, N_DEV = 1024, 256
DEFAULT_DATA_ROOT = data_root()
_CACHE: dict = {}


def examples_to_batch(exs: list[dict]) -> dict:
    K, regime = exs[0]["K"], exs[0]["regime"]
    tok = np.array([e["tokens"] for e in exs], dtype=np.int64)
    qpos = np.array([e["query_positions"] for e in exs], dtype=np.int64)
    src = np.array([e["source_positions"] for e in exs], dtype=np.int64)
    ans = np.array([e["answer_ids"] for e in exs], dtype=np.int64)
    dist = np.array([e["distances"] for e in exs], dtype=np.int64)
    vals = tok[:, 1:2 * K:2].copy()
    return {"tokens": tok, "qpos": qpos, "src": src, "answer": ans, "dist": dist, "values": vals, "K": K, "regime": regime,
            "T": tok.shape[1], "ids": [e["id"] for e in exs]}


def check_invariants(batch: dict) -> None:
    """Staff invariants (vendored ``validate``) + extra checks on a batch."""
    K = batch["K"]
    for b in range(batch["tokens"].shape[0]):
        ex = {"tokens": batch["tokens"][b].tolist(), "K": K, "regime": batch["regime"], "query_positions": batch["qpos"][b].tolist(),
              "answer_ids": batch["answer"][b].tolist(), "source_positions": batch["src"][b].tolist(), "distances": batch["dist"][b].tolist()}
        validate(ex)
        t = batch["tokens"][b]
        assert len(set(t[0:2 * K:2])) == K and t[0:2 * K:2].min() >= KEY_LO and t[0:2 * K:2].max() < KEY_HI
        assert len(set(t[1:2 * K:2])) == K and t[1:2 * K:2].min() >= VAL_LO and t[1:2 * K:2].max() < VAL_HI
        assert np.array_equal(np.sort(t[batch["qpos"][b]]), np.sort(t[0:2 * K:2])), "each key queried exactly once"
    if batch["regime"] == "far":
        assert (batch["dist"] % 4 != 0).any() and len(np.unique(batch["dist"])) > 10, "far offsets must not be a fixed grid"


def _frozen(split: str, data_root) -> dict:
    """(K, regime) -> list of records, from the frozen gz file (verified against the generator) or regenerated."""
    split = "development" if split in ("dev", "development") else split
    key = (split, str(data_root))
    if key in _CACHE:
        return _CACHE[key]
    f = Path(data_root) / "data" / "mqar" / ("mqar_test.jsonl.gz" if split == "test" else "mqar_development.jsonl.gz")
    groups: dict = {}
    n = N_TEST if split == "test" else N_DEV
    if f.exists():
        with gzip.open(f, "rt", encoding="utf8") as fh:
            for line in fh:
                r = json.loads(line)
                groups.setdefault((r["K"], r["regime"]), []).append(r)
        for (K, regime), rs in groups.items():
            assert len(rs) == n, f"{f}: {K}/{regime} has {len(rs)} records, expected {n}"
            for r in (rs[0], rs[len(rs) // 2], rs[-1]):          # spot-check the frozen file against the generator
                assert make_example(split, K, regime, int(r["id"].split("-")[-1])) == r
    else:
        import warnings
        warnings.warn(f"frozen MQAR file {f} not found: regenerating the {split} split with the (deterministic, identical) staff generator")
        for K in KS:
            for regime in REGIMES:
                groups[(K, regime)] = [make_example(split, K, regime, i) for i in range(n)]
    _CACHE[key] = groups
    return groups


def make_eval_set(K: int, regime: str, split: str = "test", data_root=DEFAULT_DATA_ROOT, n: int | None = None) -> dict:
    rs = _frozen(split, data_root)[(K, regime)]
    return examples_to_batch(rs[:n] if n else rs)


def train_batch(K: int, regime: str, seed: int, step: int, B: int = 32) -> dict:
    base = (seed - 6304) * 10_000_000 + step * B                # disjoint index ranges per seed (runs use << 10M examples)
    return examples_to_batch([make_example("train", K, regime, base + b) for b in range(B)])


class PrefetchStream:
    """Generates training batches on a background thread so CPU data generation overlaps GPU compute."""

    def __init__(self, K, regime, seed, n_steps, B=32, depth=8):
        self.q = queue.Queue(depth)
        self.t = threading.Thread(target=self._run, args=(K, regime, seed, n_steps, B), daemon=True)
        self.t.start()

    def _run(self, K, regime, seed, n_steps, B):
        for s in range(n_steps):
            self.q.put(train_batch(K, regime, seed, s, B))

    def next(self):
        return self.q.get()


def to_torch(batch: dict, device="cpu") -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            out[k] = torch.as_tensor(v, device=device)
        else:
            out[k] = v
    out["qpos"] = out.get("qpos")
    return out


def query_logits(logits: torch.Tensor, qpos: torch.Tensor) -> torch.Tensor:
    """(N,T,V) -> (N,K,64): value-ID logits at the query positions."""
    idx = qpos[..., None].expand(-1, -1, logits.shape[-1])
    return torch.gather(logits, 1, idx)[..., VAL_LO:VAL_HI]


def _stored_mask(values: torch.Tensor) -> torch.Tensor:
    """(N,K) stored value IDs -> (N,1,64) boolean mask of the stored candidates."""
    m = torch.zeros(values.shape[0], 64, dtype=torch.bool, device=values.device)
    m.scatter_(1, values - VAL_LO, True)
    return m[:, None, :]


def loss_fn(logits, batch, loss_over: str = "value64") -> torch.Tensor:
    ql = query_logits(logits, batch["qpos"])
    target = batch["answer"] - VAL_LO
    if loss_over == "stored_k":
        ql = ql.masked_fill(~_stored_mask(batch["values"]), float("-inf"))
    return F.cross_entropy(ql.reshape(-1, 64).float(), target.reshape(-1))


@torch.no_grad()
def evaluate(model, batch: dict, device="cpu", chunk: int = 256, **model_kw) -> dict:
    """Per-query records and aggregate metrics on one evaluation batch (a fixed (K, regime) set)."""
    model.eval()
    b = to_torch(batch, device)
    N, K = b["tokens"].shape[0], batch["K"]
    c64, cK, ce = [], [], []
    for s in range(0, N, chunk):
        sl = slice(s, s + chunk)
        logits = model(b["tokens"][sl], **model_kw)
        sub = {k: (v[sl] if torch.is_tensor(v) else v) for k, v in b.items()}
        ql = query_logits(logits, sub["qpos"]).float()
        tgt = sub["answer"] - VAL_LO
        c64.append(ql.argmax(-1) == tgt)
        cK.append(ql.masked_fill(~_stored_mask(sub["values"]), float("-inf")).argmax(-1) == tgt)
        ce.append(F.cross_entropy(ql.reshape(-1, 64), tgt.reshape(-1), reduction="none").view(-1, K))
    c64, cK, ce = torch.cat(c64).cpu().numpy(), torch.cat(cK).cpu().numpy(), torch.cat(ce).cpu().numpy()
    d = batch["dist"]
    out = {"n_sequences": N, "K": K, "regime": batch["regime"], "acc64": float(c64.mean()), "accK": float(cK.mean()),
           "ce": float(ce.mean()), "chance64": 1 / 64, "chanceK": 1 / K, "bins": {}}
    for lo, hi in BINS:
        m = (d >= lo) & (d < hi)
        if m.any():
            out["bins"][f"[{lo},{hi})"] = {"n_queries": int(m.sum()), "acc64": float(c64[m].mean()), "accK": float(cK[m].mean()), "ce": float(ce[m].mean())}
    out["_per_query"] = {"correct64": c64, "correctK": cK, "dist": d}
    return out
