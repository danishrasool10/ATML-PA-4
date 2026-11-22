"""Capture environment / hardware / checkpoint-revision metadata for every result file."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

KEY_PACKAGES = ["torch", "transformers", "huggingface_hub", "numpy", "scipy", "pandas", "matplotlib", "datasets", "safetensors", "tokenizers"]


def _git(*args: str, cwd: str | Path | None = None) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def package_versions() -> dict[str, str | None]:
    from importlib import metadata

    out = {}
    for p in KEY_PACKAGES:
        try:
            out[p] = metadata.version(p)
        except Exception:
            out[p] = None
    return out


def cuda_info() -> dict[str, Any]:
    info: dict[str, Any] = {"cuda_available": torch.cuda.is_available(), "torch_cuda_version": torch.version.cuda}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            device_name=props.name,
            compute_capability=f"{props.major}.{props.minor}",
            total_memory_bytes=props.total_memory,
            device_count=torch.cuda.device_count(),
            cudnn_version=torch.backends.cudnn.version(),
        )
    return info


def tokenizer_fingerprint(tokenizer) -> dict[str, Any]:
    out: dict[str, Any] = {"class": type(tokenizer).__name__}
    for attr in ("name_or_path", "vocab_size"):
        out[attr] = getattr(tokenizer, attr, None)
    try:
        out["len"] = len(tokenizer)
        vocab = tokenizer.get_vocab()
        out["vocab_sha256"] = hashlib.sha256(json.dumps(sorted(vocab.items())).encode()).hexdigest()
    except Exception:
        pass
    return out


def model_fingerprint(model) -> dict[str, Any]:
    cfg = model.config
    out = {
        "name_or_path": getattr(cfg, "_name_or_path", None),
        "checkpoint_revision": getattr(cfg, "_commit_hash", None),
        "model_type": cfg.model_type,
        "dtype": str(next(model.parameters()).dtype),
        "attention_backend": getattr(cfg, "_attn_implementation", None),
        "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", None),
        "num_hidden_layers": cfg.num_hidden_layers,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "device": str(next(model.parameters()).device),
    }
    return out


def collect_environment(model=None, tokenizer=None, extra: dict | None = None, repo_root: str | Path | None = None) -> dict[str, Any]:
    env: dict[str, Any] = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
        "packages": package_versions(),
        "cuda": cuda_info(),
        "git_commit": _git("rev-parse", "HEAD", cwd=repo_root),
        "git_dirty": bool(_git("status", "--porcelain", cwd=repo_root)),
    }
    if model is not None:
        env["model"] = model_fingerprint(model)
    if tokenizer is not None:
        env["tokenizer"] = tokenizer_fingerprint(tokenizer)
    if extra:
        env.update(extra)
    return env
