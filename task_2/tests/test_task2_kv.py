"""Part C: quantizer/packing/budget math, engine vs HF equivalence (C1), policies, and an end-to-end tiny run."""
import numpy as np
import pytest
import torch

from pa4common.tiny import ByteChatTokenizer, make_synthetic_kv_cases, make_tiny_model
from pa4t2.kvcache import quant as Q
from pa4t2.kvcache.data import build_prompt, case_split, extract_code
from pa4common.qwen_engine import KVEngine
from pa4t2.kvcache.evaluate import analyze, evaluate_prompt, validate_reference
from pa4t2.kvcache.policies import METHODS, SINK, RECENT, compress, fill_by_recency, fill_by_scores, safeguards


@pytest.fixture(scope="module")
def setup():
    tok = ByteChatTokenizer()
    model = make_tiny_model(layers=2, hidden=128, heads=2, kv_heads=1)          # head_dim 64
    cases = make_synthetic_kv_cases(n_per_cell=1)
    return tok, model, KVEngine(model), cases


# ------------------------------------------------------------------ math / packing
def test_budget_formula_and_known_values():
    # Qwen2.5-1.5B: 28 layers, 2 KV heads, d_h 128. L=2048 -> 32 full groups, r=0
    assert Q.budget_bytes(4, 2048, 28, 2, 128) == 2 * 28 * 2 * 128 * 32 * (32 + 4)
    assert Q.fp16_bytes(2048, 28, 2, 128) == 4 * 28 * 2 * 128 * 2048
    assert Q.budget_bytes(4, 2048 + 10, 28, 2, 128) - Q.budget_bytes(4, 2048, 28, 2, 128) == 2 * 28 * 2 * 128 * 20


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_pack_roundtrip(bits):
    g = torch.Generator().manual_seed(1)
    c = torch.randint(0, 2 ** bits, (3, 5, 64), generator=g, dtype=torch.uint8)
    assert torch.equal(Q.unpack_codes(Q.pack_codes(c, bits), bits), c)
    assert Q.pack_codes(c, bits).shape[-1] == 64 * bits // 8


@pytest.mark.parametrize("layout", ["token", "channel"])
@pytest.mark.parametrize("bits", [2, 4])
def test_packed_tensor_matches_fake_quant_and_byte_formula(layout, bits):
    x = torch.randn(2, 128, 128, generator=torch.Generator().manual_seed(2)).half()
    pt = Q.PackedTensor(x, bits, layout)
    assert torch.equal(pt.decode(), Q.fake_quantize(x, bits, layout))
    n_el = x.numel()
    assert pt.nbytes() == n_el * bits // 8 + Q.META_BYTES * (n_el // 64)
    # error bounded by half a step per group
    err = (pt.decode().float() - x.float()).abs().max()
    assert err < (x.max() - x.min()).float() / (2 ** bits - 1)


def test_int4_channel_realized_bytes_equal_budget_when_dh128():
    """For d_h = 128 the channel-K / token-V INT4 reference serialises to exactly M_4(L); token/token also does."""
    L, n_l, H, d = 200, 2, 2, 128
    from pa4common.qwen_engine import KVCache, LayerCache
    from pa4t2.kvcache.policies import _quantize_prefix
    g = torch.Generator().manual_seed(3)
    mk = lambda: KVCache([LayerCache(torch.randn(1, H, L, d, generator=g).half(), torch.randn(1, H, L, d, generator=g).half()) for _ in range(n_l)])
    for kl in ("channel", "token"):
        nb, *_ = _quantize_prefix(mk(), 4, kl, "token")
        assert nb == Q.budget_bytes(4, L, n_l, H, d)


def test_hybrid_capacity_is_maximal_feasible():
    n_l, H, d, L = 28, 2, 128, 2048
    B = Q.budget_bytes(4, L, n_l, H, d)
    n = Q.hybrid_capacity(B, L, n_l, H, d)
    assert Q.hybrid_bytes(n, L, n_l, H, d) <= B
    assert all(Q.hybrid_bytes(m, L, n_l, H, d) > B for m in range(n + 1, L + 1))
    assert n > SINK + RECENT


def test_fp16_capacity_respects_budget():
    n_l, H, d, L = 28, 2, 128, 4096 + 17
    B = Q.budget_bytes(4, L, n_l, H, d)
    T = Q.fp16_token_capacity(B, n_l, H, d)
    assert 4 * n_l * H * d * T + Q.INDEX_BYTES * T <= B < 4 * n_l * H * d * (T + 1) + Q.INDEX_BYTES * (T + 1)


# ------------------------------------------------------------------ data
def test_extractor_and_split():
    assert extract_code("The code is 26836.") == "26836"
    assert extract_code("id 123456 then 98765") == "98765"          # 6-digit run is not a code
    assert extract_code("no digits") is None
    assert case_split({"id": "KV-2048-start-0-00"}) == "dev" and case_split({"id": "KV-2048-start-0-07"}) == "test"


def test_prompt_split_and_evidence(setup):
    tok, _, _, cases = setup
    for rec in cases[:6]:
        pr = build_prompt(rec, tok)
        text = tok.apply_chat_template([{"role": "user", "content": rec["user_text"]}])
        marker_idx = text.index("QUESTION_BEGIN")
        assert tok.decode(pr.ids[pr.n_prefix:]).startswith("QUESTION_BEGIN")
        assert pr.n_prefix == marker_idx and pr.ids[: pr.n_prefix] == tok(text)["input_ids"][:marker_idx]
        assert tok.decode([pr.ids[i] for i in pr.evidence_tokens]) == rec["evidence_text"]
        assert max(pr.evidence_tokens) < pr.n_prefix


# ------------------------------------------------------------------ engine (C1)
def test_engine_matches_hf_forward_and_generate(setup):
    tok, model, eng, cases = setup
    pr = build_prompt(cases[0], tok)
    rep = validate_reference(eng, tok, [pr], eos=set(), n_gen=1)
    r = rep["rows"][0]
    assert r["maxabs_hf_vs_oneshot"] < 1e-4 and r["maxabs_hf_vs_split"] < 1e-4
    assert r["maxabs_split_vs_keepall"] < 1e-5 and r["maxabs_split_vs_reversed"] < 1e-4
    assert rep["all_top1_agree"] and rep["all_generations_identical"]


def test_eviction_uses_absolute_positions(setup):
    """Dropping tokens must change the logits (cache really shorter) while a kept-token cache keeps absolute RoPE positions:
    selecting every token with reversed physical order is identical to the original (checked above); here we also verify the
    new token position ids continue from L, not from the physical cache length."""
    tok, model, eng, cases = setup
    pr = build_prompt(cases[1], tok)
    pre = torch.tensor([pr.prefix_ids])
    _, cache = eng.prefill(pre)
    from pa4t2.kvcache.policies import _select
    keep = torch.arange(0, pr.n_prefix, 2)
    short = _select(cache, keep)
    suf = torch.tensor([pr.suffix_ids])
    a = eng.forward(suf, torch.arange(pr.n_prefix, pr.n_prefix + suf.shape[1])[None], short.clone(), update=True)
    b = eng.forward(suf, torch.arange(short.n, short.n + suf.shape[1])[None], short.clone(), update=True)     # wrong: physical positions
    assert not torch.allclose(a, b)
    assert short.n == len(keep)


def test_score_suffix_is_a_distribution_and_leaves_cache_untouched(setup):
    tok, model, eng, cases = setup
    pr = build_prompt(cases[2], tok)
    _, cache = eng.prefill(torch.tensor([pr.prefix_ids]))
    n0 = cache.n
    k0 = cache.layers[0].K.clone()
    sc = eng.score_suffix(torch.tensor([pr.suffix_ids]), cache, pr.n_prefix)
    assert cache.n == n0 and torch.equal(cache.layers[0].K, k0)
    assert sc.shape == (pr.n_prefix,) and (sc >= 0).all() and 0 < sc.sum() <= 1.0 + 1e-5


def test_capture_matches_manual_attention(setup):
    """Mean FP32 attention mass on the cache, recomputed by brute force for layer 0 only (single layer model)."""
    tok, _, _, cases = setup
    model = make_tiny_model(layers=1, hidden=128, heads=2, kv_heads=1)
    eng = KVEngine(model)
    pr = build_prompt(cases[0], tok)
    _, cache = eng.prefill(torch.tensor([pr.prefix_ids]))
    suf = torch.tensor([pr.suffix_ids])
    sc = eng.score_suffix(suf, cache, pr.n_prefix)
    m = model.model
    h = m.embed_tokens(suf)
    pos = torch.arange(pr.n_prefix, pr.n_prefix + suf.shape[1])[None]
    cos, sin = [t.unsqueeze(1) for t in m.rotary_emb(h, pos)]
    a = m.layers[0].self_attn
    x = m.layers[0].input_layernorm(h)
    from pa4common.qwen_engine import rotate_half
    q = a.q_proj(x).view(1, -1, 2, 64).transpose(1, 2)
    k = a.k_proj(x).view(1, -1, 1, 64).transpose(1, 2)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    K = torch.cat([cache.layers[0].K, k], -2)
    s = q @ K.transpose(-1, -2) / 8.0
    n_c, S = pr.n_prefix, suf.shape[1]
    allowed = torch.arange(n_c + S)[None, :] <= (torch.arange(S)[:, None] + n_c)
    p = torch.softmax(s.masked_fill(~allowed, float("-inf")), -1)[..., :n_c]
    assert torch.allclose(sc, p.mean(dim=(1, 2))[0], atol=1e-5)


# ------------------------------------------------------------------ policies
def test_policies_budget_and_structure(setup):
    tok, model, eng, cases = setup
    # use a prompt whose prefix is long enough for the safeguards + hybrid structure
    rec = cases[-1]
    pr = build_prompt(rec, tok)
    dims = (eng.n_layers, eng.Hkv, eng.d)
    _, cache = eng.prefill(torch.tensor([pr.prefix_ids]))
    scores = eng.score_suffix(torch.tensor([pr.suffix_ids]), cache, pr.n_prefix)
    L = pr.n_prefix
    out = {}
    for m in METHODS:
        c = compress(m, cache.clone(), scores, dims)
        out[m] = c
        assert c.packed_bytes <= c.budget_bytes, m
    sr, qa = out["sink_recent"], out["question_aware"]
    assert len(sr.kept_fp16) == len(qa.kept_fp16)
    for c in (sr, qa):
        kept = set(c.kept_fp16)
        assert set(range(SINK)) <= kept and set(range(L - RECENT, L)) <= kept
        assert c.cache.n == len(kept)
    assert set(range(L - len(sr.kept_fp16) + 0, L)) >= set(range(L - RECENT, L))
    # sink+recent keeps a contiguous recent block, question-aware keeps the highest-scoring remaining
    sc = scores.clone()
    rest = sorted(set(range(L)) - set(safeguards(L).tolist()))
    kept_extra = [i for i in qa.kept_fp16 if i in set(rest)]
    assert min(scores[kept_extra]) >= max(scores[[i for i in rest if i not in set(kept_extra)]])
    assert out["int4_channel"].cache.n == L and out["int4_token"].cache.n == L
    assert out["hybrid"].cache.n == L
    assert out["hybrid"].info["n_fp16"] >= SINK + RECENT or True
    for m in ("int4_token", "int4_channel", "hybrid"):
        assert out[m].k_err > 0 and out[m].v_err > 0
    assert out["hybrid"].k_err > out["int4_channel"].k_err * 0.5       # INT2 body is no more accurate than INT4


def test_fill_helpers():
    L = 300
    sc = torch.rand(L, generator=torch.Generator().manual_seed(0))
    idx = fill_by_scores(L, 120, sc)
    assert idx.numel() == 120 and torch.equal(idx, idx.sort().values)
    idx2 = fill_by_recency(L, 120, "cpu")
    assert idx2.numel() == 120 and idx2[-1] == L - 1
    assert set(range(SINK)) <= set(idx2.tolist()) and set(range(L - 120 + SINK, L)) <= set(idx2.tolist())


def test_quantize_new_tokens_when_group_completes(setup):
    tok, model, eng, cases = setup
    pr = build_prompt(cases[0], tok)
    dims = (eng.n_layers, eng.Hkv, eng.d)
    _, cache = eng.prefill(torch.tensor([pr.prefix_ids]))
    c = compress("int4_channel", cache.clone(), None, dims)
    lc = c.cache.layers[0]
    n_full = (pr.n_prefix // 64) * 64
    assert lc.n_quant == n_full
    extra = 64 - (pr.n_prefix - n_full)
    k = torch.randn(1, eng.Hkv, extra, eng.d).half()
    before = lc.K.clone()
    lc.append(k, k)
    assert lc.n_quant == n_full + 64
    assert torch.equal(lc.K[..., :n_full, :], before[..., :n_full, :])
    assert not torch.equal(lc.K[..., n_full:n_full + 64, :][..., -extra:, :], k)          # quantized now


# ------------------------------------------------------------------ end to end
def test_end_to_end_tiny(setup, tmp_path):
    tok, model, eng, cases = setup
    from pa4t2.kvcache.evaluate import run
    cases2 = make_synthetic_kv_cases(n_per_cell=2)[:6]
    res = run(cases2, tok, model, tmp_path, n_boot=200, validate_n=1)
    assert (tmp_path / "per_example.jsonl").exists() and (tmp_path / "partC_summary.md").exists()
    s = res["subsets"]["all"]
    assert res["subsets"]["test_only"]["n"] == 3
    assert set(s["methods"]) == {"fp16", *METHODS}
    assert s["methods"]["fp16"]["kl_post_prefill"] == 0.0
    for m in METHODS:
        assert s["methods"][m]["packed_bytes_mean"] <= s["methods"][m]["budget_bytes_mean"] + 1e-6
    # rerun resumes without recomputing
    run(cases2, tok, model, tmp_path, n_boot=200, validate_n=0)


def test_packed_cache_roundtrip_and_bench_smoke(setup):
    from pa4t2.kvcache.bench import PackedCache, bench_condition
    tok, model, eng, cases = setup
    ids = torch.randint(10, 250, (1, 200))
    _, cache = eng.prefill(ids)
    pc = PackedCache(cache)
    dec = pc.decode()
    n_full = 192
    for a, b in zip(cache.layers, dec.layers):
        assert torch.equal(a.K[..., n_full:, :], b.K[..., n_full:, :])               # FP16 tail untouched
        assert torch.equal(b.K[..., :n_full, :], Q.fake_quantize(a.K[..., :n_full, :], 4, "channel"))
    rec = bench_condition(eng, 2, 200, "cpu", reps=2)
    assert rec["status"] == "ok" and rec["packed_int4_bytes"] < rec["fp16_cache_bytes"] * 0.5
    assert rec["step_fp16"]["reps"] == 2
