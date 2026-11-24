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


def _maxabs(a: torch.Tensor, b: torch.Tensor) -> float:
    """max |a-b|; +inf if either side contains NaN/inf (python's ``max`` would silently swallow a NaN)."""
    d = (a.float() - b.float()).abs()
    return float(d.max()) if bool(torch.isfinite(d).all()) else float("inf")


def _trial(lm: CachedLM, ids: list[int], junk: list[int], cuts: list[int], k: int) -> tuple[float, float, torch.Tensor]:
    """One split-prefill + one rollback trial; returns (split error, rollback error, fresh reference logits)."""
    ref = lm.full_logits(ids)
    lm.reset()                                                              # (3) split prefill: random chunking
    rows, prev = [], 0
    for c in cuts + [len(ids)]:
        if c > prev:
            rows.append(lm.feed(ids[prev:c]))
            prev = c
    e_split = _maxabs(torch.cat(rows), ref)
    lm.reset()                                                              # (2) rollback: feed junk, truncate, re-feed truth
    lm.feed(ids[:25])
    lm.feed(junk)
    lm.truncate(k)
    assert len(lm) == k and lm.cache.n == k
    rows = lm.feed(ids[k:])
    e_roll = _maxabs(rows, ref[k:])
    assert lm.tokens == ids
    return e_split, e_roll, ref


@torch.no_grad()
def check_rollback_and_split_prefill(model, vocab_lo: int, vocab_hi: int, device, seed: int = 6304, n_trials: int = 6) -> dict:
    """KV-cache rollback / split-prefill equivalence with a *two-level* criterion.

    * Logic (hard gate): on an FP32 copy of the model the cached logits must equal a fresh forward to ``2e-3`` -- any
      real indexing / position / rollback bug shows up as O(1) logit errors there.
    * FP16 (the dtype the experiments run in): chunked vs one-shot attention are different kernels, so they round
      differently. A fixed absolute logit bound is not meaningful (Qwen logits are O(10-30) and random rare-token inputs
      excite outlier channels), so the deployed-precision error is compared with the model's own FP16 rounding floor
      ``|fresh FP16 forward - FP32 forward|`` on the same inputs: pass iff max error <= max(0.15, 3 x floor) *and* finite.
    """
    dt = next(model.parameters()).dtype
    lm = CachedLM(model)
    g = torch.Generator().manual_seed(seed)
    trials = []
    for _ in range(n_trials):
        ids = torch.randint(vocab_lo, vocab_hi, (40,), generator=g).tolist()
        cuts = sorted(torch.randint(1, len(ids), (3,), generator=g).tolist())
        junk = torch.randint(vocab_lo, vocab_hi, (7,), generator=g).tolist()
        k = int(torch.randint(5, 25, (1,), generator=g))
        trials.append((ids, junk, cuts, k))
    out = {"dtype": str(dt)}
    if dt == torch.float32:
        rows = [_trial(lm, *t)[:2] for t in trials]
        tol = _tol(dt)[0]
        ws, wr = max(r[0] for r in rows), max(r[1] for r in rows)
        out.update(max_abs_split_prefill=ws, max_abs_rollback=wr, tol=tol, passed=bool(ws < tol and wr < tol))
        return out
    # --- FP16 path: deployed-precision errors vs. rounding floor, plus the exact logic check on an FP32 copy
    rows16 = [_trial(lm, *t) for t in trials]
    ws16, wr16 = max(r[0] for r in rows16), max(r[1] for r in rows16)
    m32 = copy.deepcopy(model).float()
    lm32 = CachedLM(m32)
    rows32, floor = [], 0.0
    for t, r16 in zip(trials, rows16):
        r32 = _trial(lm32, *t)
        rows32.append(r32[:2])
        floor = max(floor, _maxabs(r16[2], r32[2]))
    del lm32, m32
    if str(device).startswith("cuda") or getattr(device, "type", "") == "cuda":
        torch.cuda.empty_cache()
    ws32, wr32 = max(r[0] for r in rows32), max(r[1] for r in rows32)
    tol32, tol16 = _tol(torch.float32)[0], max(_tol(dt)[0], 3.0 * floor)
    logic_ok = ws32 < tol32 and wr32 < tol32
    fp16_ok = ws16 < tol16 and wr16 < tol16
    out.update(max_abs_split_prefill=ws16, max_abs_rollback=wr16, tol=tol16, fp16_rounding_floor=floor,
               fp32_logic_max_abs_split_prefill=ws32, fp32_logic_max_abs_rollback=wr32, fp32_logic_tol=tol32,
               logic_passed=bool(logic_ok), fp16_passed=bool(fp16_ok), passed=bool(logic_ok and fp16_ok))
    return out


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
