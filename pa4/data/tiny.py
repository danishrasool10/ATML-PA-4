"""Tiny offline stand-ins (byte tokenizer, random Qwen2, synthetic manifests).

Used by the unit tests and ``--smoke`` mode so the whole pipeline can be exercised on CPU with no
network access. Nothing here is used for reported results.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch

BYTE_OFFSET = 3  # ids 0..2 reserved
TINY_VOCAB = 288


class ByteTokenizer:
    """Minimal tokenizer: UTF-8 bytes shifted by 3. Mimics ``encode(text, add_special_tokens=False)``."""

    name_or_path = "tiny-byte-tokenizer"
    vocab_size = 256 + BYTE_OFFSET

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [b + BYTE_OFFSET for b in text.encode("utf-8")]

    def __len__(self):
        return self.vocab_size

    def get_vocab(self):
        return {str(i): i for i in range(self.vocab_size)}


def tiny_config(layers: int = 3, hidden: int = 128, inter: int = 256, heads: int = 4, kv_heads: int = 2, tie: bool = True):
    from transformers import Qwen2Config

    return Qwen2Config(
        vocab_size=TINY_VOCAB,
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        max_position_embeddings=4096,
        tie_word_embeddings=tie,
        use_sliding_window=False,
        rope_theta=10000.0,
    )


def make_tiny_model(seed: int = 6304, dtype=torch.float32, **cfg_kwargs):
    from transformers import Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = tiny_config(**cfg_kwargs)
    cfg._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(cfg).to(dtype).eval()
    # Give the (randomly initialised) weights a little heavy-tailed structure + non-trivial biases so
    # pruning/quantization metrics are not degenerate.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 2 and "embed" not in name:
                p.mul_(torch.exp(0.5 * torch.randn_like(p)))
            if name.endswith("bias"):
                p.normal_(0, 0.05)
            if "layernorm" in name:
                p.copy_(1.0 + 0.3 * torch.rand_like(p))
    model.requires_grad_(False)
    return model


_WORDS = "the a of to and in is it you that he was for on are with as his they be at one have this from or had by hot word but what some we can out other were all there when up use your how said an each she which do their time if will way about many then them write would like so these her long make thing see him two has look more day could go come did number sound no most people my over know water than call first who may down side been now find".split()


def _sentence(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(n))


def make_synthetic_manifests(out_dir: str | Path, *, seed: int = 6304, seq_len: int = 64, n_calib: int = 8, n_dev: int = 4,
                             n_windows: int = 10, n_hs: int = 24) -> dict[str, str]:
    """Write manifests in the layouts documented in ``manifests.py`` and return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    tok = ByteTokenizer()

    def toks(n):
        return [rng.randrange(BYTE_OFFSET, TINY_VOCAB) for _ in range(n)]

    def text_toks(n):
        t = tok.encode(_sentence(rng, 4 * n))[:n]
        return t

    paths = {}
    paths["calibration"] = out / "calib.json"
    json.dump({"seed": seed, "items": [{"id": i, "input_ids": text_toks(seq_len)} for i in range(n_calib)]}, open(paths["calibration"], "w"))
    paths["development"] = out / "dev.json"
    json.dump({"seed": seed, "items": [{"id": i, "input_ids": text_toks(seq_len)} for i in range(n_dev)]}, open(paths["development"], "w"))
    for key in ("wikitext2_test", "c4_validation"):
        wins = [{"doc_id": f"{key}-{i}", "input_ids": text_toks(seq_len if i % 4 else seq_len // 2)} for i in range(n_windows)]
        paths[key] = out / f"{key}.json"
        json.dump({"windows": wins, "n_scored_tokens": sum(len(w["input_ids"]) - 1 for w in wins)}, open(paths[key], "w"))
    hs = []
    for i in range(n_hs):
        hs.append({
            "ind": 1000 + i,
            "activity_label": rng.choice(["Cooking", "Sports", "Travel"]),
            "ctx_a": _sentence(rng, 6) + ".",
            "ctx_b": _sentence(rng, 3),
            "endings": [_sentence(rng, 4 + j) + (" [title] x" if j == 1 else "") for j in range(4)],
            "label": str(rng.randrange(4)),
        })
    paths["hellaswag"] = out / "hellaswag.json"
    json.dump({"items": hs}, open(paths["hellaswag"], "w"))
    return {k: str(v) for k, v in paths.items()}
