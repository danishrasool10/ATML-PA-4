"""Machine-readable result serialisation (JSON / JSONL / NPZ) used by every task."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def jsonable(obj: Any) -> Any:
    """Recursively convert tensors / arrays / dataclasses / paths to JSON-safe objects."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if obj != obj:
            return "NaN"
        if obj in (float("inf"), float("-inf")):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return jsonable(float(obj))
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, torch.Tensor):
        return jsonable(obj.detach().cpu().tolist())
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


def write_json(path: str | Path, obj: Any, indent: int | None = 2) -> Path:
    """Atomic JSON write (write to temp file, then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(jsonable(obj), f, indent=indent)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def append_jsonl(path: str | Path, record: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(jsonable(record)) + "\n")


def write_jsonl(path: str | Path, records: Iterable[Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(jsonable(r)) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[Any]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(jsonable(obj), sort_keys=True).encode()).hexdigest()
