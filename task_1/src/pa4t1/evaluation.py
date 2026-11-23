"""Perplexity and zero-shot HellaSwag evaluation (shared, unchanged across all conditions)."""
from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from .constants import TASK_VERSIONS


# ----------------------------------------------------------------------------- perplexity

@torch.no_grad()
def eval_windows(model, windows: list[list[int]], batch_size: int = 4, device=None) -> dict:
    """Nonoverlapping, document-isolated perplexity.

    Every window is an independently evaluated sequence (no state carried between windows). The
    first token of each window is conditioning only and is *excluded* from N. Windows of equal
    length are batched together; right-padding is never needed (so no padding mask either) and the
    result does not depend on the batch size beyond floating-point noise.
    Returns per-window summed NLL / scored-token counts (needed for the paired bootstrap).
    """
    device = device or next(model.parameters()).device
    n = len(windows)
    nll = np.zeros(n, dtype=np.float64)
    cnt = np.zeros(n, dtype=np.int64)
    by_len = defaultdict(list)
    for i, w in enumerate(windows):
        assert len(w) >= 2
        by_len[len(w)].append(i)
    for L, idxs in by_len.items():
        for s in range(0, len(idxs), batch_size):
            chunk = idxs[s : s + batch_size]
            ids = torch.tensor([windows[i] for i in chunk], dtype=torch.long, device=device)
            logits = model(input_ids=ids, use_cache=False).logits
            lp = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="none")
            lp = lp.view(len(chunk), L - 1).double().sum(1).cpu().numpy()
            for j, i in enumerate(chunk):
                nll[i] = lp[j]
                cnt[i] = L - 1
    N = int(cnt.sum())
    mean_nll = float(nll.sum() / N)
    return {"N": N, "n_windows": n, "mean_nll": mean_nll, "ppl": math.exp(mean_nll),
            "window_nll_sum": nll, "window_token_count": cnt}


# ----------------------------------------------------------------------------- HellaSwag

def hs_preprocess(text: str) -> str:
    """lm-evaluation-harness hellaswag preprocessing (task version 1.0)."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def hs_doc(ex: dict) -> tuple[str, list[str], int]:
    ctx = ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
    query = hs_preprocess(ex["activity_label"] + ": " + ctx)
    choices = [hs_preprocess(e) for e in ex["endings"]]
    return query, choices, int(ex["label"])


def encode_pair(tokenizer, context: str, continuation: str) -> tuple[list[int], list[int]]:
    """lm-eval ``_encode_pair``: trailing context whitespace moves to the continuation; the
    continuation tokens are the suffix of the whole-string encoding past the context encoding."""
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole = tokenizer.encode(context + continuation, add_special_tokens=False)
    ctx = tokenizer.encode(context, add_special_tokens=False)
    cont = whole[len(ctx):]
    assert len(cont) > 0 and whole[: len(ctx)] == ctx, "context/continuation tokenization boundary mismatch"
    return ctx, cont


@torch.no_grad()
def score_continuations(model, requests: list[tuple[list[int], list[int]]], batch_size: int = 8, device=None) -> np.ndarray:
    """Sum of conditional log-probabilities of each continuation given its context."""
    device = device or next(model.parameters()).device
    order = sorted(range(len(requests)), key=lambda i: -(len(requests[i][0]) + len(requests[i][1])))
    out = np.zeros(len(requests), dtype=np.float64)
    for s in range(0, len(order), batch_size):
        chunk = order[s : s + batch_size]
        seqs = [requests[i][0] + requests[i][1] for i in chunk]
        L = max(len(x) for x in seqs)
        ids = torch.zeros(len(chunk), L, dtype=torch.long, device=device)
        for j, x in enumerate(seqs):  # right padding is harmless under causal masking
            ids[j, : len(x)] = torch.tensor(x, device=device)
        logp = F.log_softmax(model(input_ids=ids, use_cache=False).logits.float(), dim=-1)
        for j, i in enumerate(chunk):
            ctx, cont = requests[i]
            a = len(ctx)
            tgt = torch.tensor(cont, device=device)
            out[i] = logp[j, a - 1 : a - 1 + len(cont)].gather(1, tgt[:, None]).sum().item()
    return out


def eval_hellaswag(model, tokenizer, examples: list[dict], ids: list, batch_size: int = 8, device=None) -> dict:
    """Zero-shot 4-way scoring. ``acc`` uses raw log-likelihood, ``acc_norm`` divides by the
    *character* length of the ending (lm-eval convention), never by token count."""
    requests, lens, golds = [], [], []
    for ex in examples:
        query, choices, gold = hs_doc(ex)
        golds.append(gold)
        for ch in choices:
            requests.append(encode_pair(tokenizer, query, " " + ch))
            lens.append(float(len(ch)))
    ll = score_continuations(model, requests, batch_size, device).reshape(len(examples), 4)
    clen = np.array(lens).reshape(len(examples), 4)
    golds = np.array(golds)
    pred = ll.argmax(1)
    pred_norm = (ll / clen).argmax(1)
    correct = (pred == golds).astype(np.float64)
    correct_norm = (pred_norm == golds).astype(np.float64)
    per_example = [
        {"id": ids[i], "loglik": ll[i].tolist(), "char_len": clen[i].tolist(), "loglik_norm": (ll[i] / clen[i]).tolist(),
         "pred": int(pred[i]), "pred_norm": int(pred_norm[i]), "label": int(golds[i])}
        for i in range(len(examples))
    ]
    return {"n": len(examples), "acc": float(correct.mean()), "acc_norm": float(correct_norm.mean()),
            "correct": correct, "correct_norm": correct_norm, "per_example": per_example,
            "task": TASK_VERSIONS["hellaswag_task"],
            "prompt_format": "query = preprocess(activity_label + ': ' + ctx_a + ' ' + ctx_b.capitalize()); "
                             "continuation = ' ' + preprocess(ending); preprocess = lm-eval hellaswag (strip, ' [title]'->'. ', "
                             "drop [..], collapse double spaces); no BOS; zero-shot",
            "normalization": "loglik / len(ending in characters)"}


def subset_metrics(hs: dict, n: int = 512) -> dict:
    return {"n": n, "acc": float(hs["correct"][:n].mean()), "acc_norm": float(hs["correct_norm"][:n].mean())}
