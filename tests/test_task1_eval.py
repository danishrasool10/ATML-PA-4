"""Perplexity token counts / isolation, HellaSwag scoring conventions, bootstrap."""
import math

import numpy as np
import torch
import torch.nn.functional as F

from pa4.common.bootstrap import paired_bootstrap_mean, paired_bootstrap_ppl
from pa4.data.manifests import chunk_document
from pa4.task1.evaluation import (encode_pair, eval_hellaswag, eval_windows, hs_doc, hs_preprocess, score_continuations)


def _ref_nll(model, window):
    ids = torch.tensor([window])
    logits = model(input_ids=ids).logits[0, :-1].double()
    return F.cross_entropy(logits, ids[0, 1:], reduction="sum").item()


def test_chunking_is_nonoverlapping_and_excludes_first_token():
    toks = list(range(1, 1300))
    wins = chunk_document(toks, 512)
    assert [len(w) for w in wins] == [512, 512, 275]
    assert sum(len(w) for w in wins) == len(toks)           # every token is in exactly one window
    assert sum(len(w) - 1 for w in wins) == len(toks) - 3   # N excludes first token of each window


def test_ppl_token_count_matches_manifest_and_reference(model):
    g = torch.Generator().manual_seed(0)
    wins = [torch.randint(3, 280, (L,), generator=g).tolist() for L in (64, 64, 64, 40, 64, 17)]
    r = eval_windows(model, wins, batch_size=4)
    assert r["N"] == sum(len(w) - 1 for w in wins) == int(r["window_token_count"].sum())
    ref = [_ref_nll(model, w) for w in wins]
    assert np.allclose(r["window_nll_sum"], ref, rtol=1e-6)
    assert math.isclose(r["mean_nll"], sum(ref) / r["N"], rel_tol=1e-6)
    assert math.isclose(r["ppl"], math.exp(r["mean_nll"]), rel_tol=1e-9)


def test_ppl_batch_size_invariance_and_window_isolation(model):
    g = torch.Generator().manual_seed(3)
    wins = [torch.randint(3, 280, (64,), generator=g).tolist() for _ in range(6)]
    a = eval_windows(model, wins, batch_size=1)
    b = eval_windows(model, wins, batch_size=6)
    assert np.allclose(a["window_nll_sum"], b["window_nll_sum"], rtol=1e-5)
    # changing another window must not change this window's score (no state carried across documents)
    wins2 = [wins[0]] + [torch.randint(3, 280, (64,), generator=g).tolist() for _ in range(5)]
    c = eval_windows(model, wins2, batch_size=3)
    assert np.isclose(c["window_nll_sum"][0], a["window_nll_sum"][0], rtol=1e-5)


class _FakeTok:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 250 + 3 for c in text]


def test_hellaswag_preprocess_and_doc():
    assert hs_preprocess("  A man [title] runs  fast [step] now. ") == "A man. runs fast now."
    ex = {"activity_label": "Cooking", "ctx_a": "He cuts.", "ctx_b": "then he stirs", "endings": ["a [title] b", "c", "d", "e"], "label": "2"}
    q, ch, gold = hs_doc(ex)
    assert q == "Cooking: He cuts. Then he stirs" and gold == 2 and ch[0] == "a. b"


def test_encode_pair_boundary_and_trailing_space():
    tok = _FakeTok()
    ctx, cont = encode_pair(tok, "hello ", "world")
    assert tok.encode("hello world") == ctx + cont
    assert tok.encode("hello") == ctx and len(cont) == len(" world")


def test_hellaswag_acc_and_acc_norm_use_character_length(model, tok):
    exs = []
    for i in range(10):
        exs.append({"ind": i, "activity_label": "A", "ctx_a": "the cat sat on", "ctx_b": "a mat and",
                    "endings": ["x", "the long long long ending here", "mid ending", "tiny"], "label": str(i % 4)})
    ids = [e["ind"] for e in exs]
    r = eval_hellaswag(model, tok, exs, ids, batch_size=3)
    for pe in r["per_example"]:
        ll, cl = np.array(pe["loglik"]), np.array(pe["char_len"])
        assert pe["pred"] == int(ll.argmax()) and pe["pred_norm"] == int((ll / cl).argmax())
        assert cl.tolist() == [1.0, 30.0, 10.0, 4.0]    # characters of the ending, NOT tokenizer tokens
    # batch-size invariance of the log-likelihood scores
    r2 = eval_hellaswag(model, tok, exs, ids, batch_size=40)
    assert np.allclose([p["loglik"] for p in r["per_example"]], [p["loglik"] for p in r2["per_example"]], rtol=1e-5)
    assert len(r["per_example"][0]["loglik"]) == 4
    # subset of the first 5 equals the scores computed on those 5 alone (manifest order)
    r5 = eval_hellaswag(model, tok, exs[:5], ids[:5])
    assert r["correct"][:5].tolist() == r5["correct"].tolist()


def test_score_continuations_matches_manual(model):
    ctx, cont = [5, 6, 7, 8], [9, 10, 11]
    s = score_continuations(model, [(ctx, cont)])[0]
    ids = torch.tensor([ctx + cont])
    lp = F.log_softmax(model(input_ids=ids).logits[0].double(), -1)
    manual = sum(lp[len(ctx) - 1 + i, t].item() for i, t in enumerate(cont))
    assert math.isclose(s, manual, rel_tol=1e-5)


def test_paired_bootstrap_mean_detects_difference_and_null():
    rng = np.random.default_rng(0)
    a = (rng.random(2000) < 0.6).astype(float)
    b = a.copy()
    flip = rng.random(2000) < 0.15
    b[flip] = 1 - b[flip]
    r = paired_bootstrap_mean(a, b, n_boot=2000)
    assert r["ci_low"] <= r["estimate"] <= r["ci_high"] and r["n"] == 2000
    null = paired_bootstrap_mean(a, a, n_boot=500)
    assert null["estimate"] == 0 and null["ci_low"] == 0 == null["ci_high"]


def test_paired_bootstrap_ppl_pools_tokens_then_exponentiates():
    rng = np.random.default_rng(1)
    cnt = rng.integers(100, 511, size=60)
    nll_a = cnt * (3.0 + 0.2 * rng.standard_normal(60))
    nll_b = nll_a + cnt * 0.05
    r = paired_bootstrap_ppl(nll_a, cnt, nll_b, cnt, n_boot=1500)
    assert math.isclose(r["mean_nll_a"], nll_a.sum() / cnt.sum()) and math.isclose(r["ppl_a"], math.exp(nll_a.sum() / cnt.sum()))
    assert r["delta_nll"]["estimate"] < 0 and r["delta_nll"]["ci_high"] < 0   # A is consistently better by 0.05 nats/token
    assert r["delta_ppl"]["excludes_zero"]
    same = paired_bootstrap_ppl(nll_a, cnt, nll_a, cnt, n_boot=300)
    assert same["delta_ppl"]["estimate"] == 0
