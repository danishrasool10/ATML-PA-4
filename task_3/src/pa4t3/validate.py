"""Automated assertions that must pass before any distillation / speculative-decoding sweep (Step 1).

  1. chi-square (Pearson) on the 3-symbol Markov toy: exact sampler not rejected, buggy residual rejected (alpha = 0.01),
  2. KV-cache rollback equivalence: after arbitrary feed / truncate / re-feed histories and after real speculative rounds the
     cached next-token logits equal a fresh full forward of the committed sequence (positions are absolute),
  3. split-prefill equivalence: prefill in several chunks == one-shot forward,
  4. draft == target => every proposal accepted (acceptance probability exactly 1),
  5. merged vs unmerged LoRA logits agree (FP32 tight; FP16 up to rounding) and merging is applied to a fresh backbone,
  6. target / draft tokenizer and chat-template agreement.
``run_all`` writes ``validation.json``; sweeps check ``passed``.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from pa4common.results import write_json
from .lm import CachedLM
from .lora import inject_lora, lora_state, merge_into
from .specdec import speculative_generate
from .toy import run_toy_validation


def _tol(dtype):
    return (2e-3, 1e-4) if dtype == torch.float32 else (0.15, 5e-2)       # (logit max-abs tolerance, reserved)


@torch.no_grad()
def check_rollback_and_split_prefill(model, vocab_lo: int, vocab_hi: int, device, seed: int = 6304, n_trials: int = 6) -> dict:
    lm = CachedLM(model)
    g = torch.Generator().manual_seed(seed)
    tol = _tol(next(model.parameters()).dtype)[0]
    worst_roll, worst_split = 0.0, 0.0
    for t in range(n_trials):
        ids = torch.randint(vocab_lo, vocab_hi, (40,), generator=g).tolist()
        ref = lm.full_logits(ids)
        # (3) split prefill: random chunking
        lm.reset()
        cuts = sorted(torch.randint(1, len(ids), (3,), generator=g).tolist())
        rows, prev = [], 0
        for c in cuts + [len(ids)]:
            if c > prev:
                rows.append(lm.feed(ids[prev:c]))
                prev = c
        worst_split = max(worst_split, float((torch.cat(rows) - ref).abs().max()))
        # (2) rollback: feed junk, truncate to a random point, re-feed the true continuation
        lm.reset()
        lm.feed(ids[:25])
        lm.feed(torch.randint(vocab_lo, vocab_hi, (7,), generator=g).tolist())
        k = int(torch.randint(5, 25, (1,), generator=g))
        lm.truncate(k)
        assert len(lm) == k and lm.cache.n == k
        rows = lm.feed(ids[k:])
        worst_roll = max(worst_roll, float((rows - ref[k:]).abs().max()))
        assert lm.tokens == ids
    return {"max_abs_split_prefill": worst_split, "max_abs_rollback": worst_roll, "tol": tol, "passed": worst_split < tol and worst_roll < tol}


@torch.no_grad()
def check_speculative_cache(target_model, draft_model, prompt: list[int], device, gammas=(2, 4), max_new: int = 24, seed: int = 6304) -> dict:
    tgt, drf = CachedLM(target_model, "target"), CachedLM(draft_model, "draft")
    tol = _tol(next(target_model.parameters()).dtype)[0]
    out = {"cases": [], "passed": True}
    for g in gammas:
        gen = torch.Generator(device=device).manual_seed(seed + g)
        toks, st = speculative_generate(tgt, drf, prompt, g, max_new, gen, device=device, check_cache=True, timed=False)
        seq = prompt + toks
        # caches hold exactly the committed sequence (the output may be shorter than the last round's commit: max_new cap)
        m_t, m_d = min(len(tgt), len(seq)), min(len(drf), len(seq))
        ok_tokens = tgt.tokens[:m_t] == seq[:m_t] and drf.tokens[:m_d] == seq[:m_d] and len(drf) <= len(tgt) + 1
        tgt.truncate(min(len(tgt), len(seq) - 1))                               # exact rollback to 'all but the newest token'
        # cached next-token logits == fresh forward of the committed sequence
        fresh_t = tgt.full_logits(seq)
        cached = tgt.feed(seq[len(tgt):])
        err = float((cached - fresh_t[-cached.shape[0]:]).abs().max())
        out["cases"].append({"gamma": g, "n_new": len(toks), "rounds": st.rounds, "alpha_mle": st.alpha_mle, "cache_tokens_ok": ok_tokens, "max_abs_vs_fresh": err})
        out["passed"] &= bool(ok_tokens and err < tol)
    return out


@torch.no_grad()
def check_identical_models_accept_all(model, prompt, device, gamma=4, max_new=20, seed=6304) -> dict:
    a, b = CachedLM(model, "t"), CachedLM(model, "d")
    gen = torch.Generator(device=device).manual_seed(seed)
    _, st = speculative_generate(a, b, prompt, gamma, max_new, gen, device=device, timed=False)
    return {"accepted": st.accepted, "proposed": st.proposed, "passed": st.accepted == st.proposed and st.rejections == 0}


@torch.no_grad()
def check_merge(base_factory, device, vocab_lo, vocab_hi, r=8, alpha=16.0, seed=6304) -> dict:
    """``base_factory()`` returns a *fresh* adapter-free model. Random non-zero B so the check is not vacuous."""
    torch.manual_seed(seed)
    unmerged = base_factory()
    dt = next(unmerged.parameters()).dtype
    inject_lora(unmerged, r, alpha)
    for n, p in unmerged.named_parameters():
        if n.endswith("lora_B"):
            p.data.normal_(0, 0.02)
    state = lora_state(unmerged)
    merged = merge_into(base_factory(), state, r, alpha)
    base_only = base_factory()
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(vocab_lo, vocab_hi, (1, 48), generator=g).to(device)
    lu, lm_, lb = unmerged(input_ids=ids).logits.float(), merged(input_ids=ids).logits.float(), base_only(input_ids=ids).logits.float()
    diff = float((lu - lm_).abs().max())
    effect = float((lu - lb).abs().max())
    tol = 1e-3 if dt == torch.float32 else 0.15
    return {"max_abs_merged_vs_unmerged": diff, "max_abs_adapter_effect": effect, "top1_agree": bool((lu.argmax(-1) == lm_.argmax(-1)).float().mean() > 0.97),
            "tol": tol, "passed": diff < tol and effect > 1e-3}


def check_tokenizers(tok_t, tok_d, samples=("Example 123", "Hello there", "A complicated example with 𝜋", "Janet has 3 apples. How many?")) -> dict:
    same_enc = all(tok_t(s, add_special_tokens=False)["input_ids"] == tok_d(s, add_special_tokens=False)["input_ids"] for s in samples)
    msg = [{"role": "user", "content": "What is 2+2?"}]
    same_tpl = tok_t.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) == tok_d.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return {"same_encoding": same_enc, "same_chat_template": same_tpl, "passed": bool(same_enc and same_tpl)}


def run_all(target_model, draft_model, base_factory_draft, tok_t, tok_d, device, out_dir: Path, vocab_range=(10, 250), prompt=None, n_toy: int = 100_000) -> dict:
    res = {"toy_chi_square": run_toy_validation(n_toy)}
    lo, hi = vocab_range
    prompt = prompt or [5, 17, 29, 41, 53, 65, 77, 89]
    res["rollback_split_prefill_target"] = check_rollback_and_split_prefill(target_model, lo, hi, device)
    res["rollback_split_prefill_draft"] = check_rollback_and_split_prefill(draft_model, lo, hi, device)
    res["speculative_cache"] = check_speculative_cache(target_model, draft_model, prompt, device)
    res["identical_models_accept_all"] = check_identical_models_accept_all(target_model, prompt, device)
    res["merge"] = check_merge(base_factory_draft, device, lo, hi)
    res["tokenizers"] = check_tokenizers(tok_t, tok_d)
    res["passed"] = all(v["passed"] for v in res.values())
    write_json(Path(out_dir) / "validation.json", res)
    return res
