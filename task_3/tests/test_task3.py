"""Task 3: toy chi-square, cache rollback, merged-vs-unmerged, distillation loop and benchmarks on tiny random Qwen2 models."""
import numpy as np
import pytest
import torch

from pa4common.tiny import ByteChatTokenizer, make_tiny_model
from pa4t3 import toy
from pa4t3.banks import STRATUM_NAMES, divergences, evaluate_alignment, load_banks, make_bank, sample_positions, summarize_alignment, write_banks
from pa4t3.bench import analyze_spec, breakeven_alpha, breakeven_requests, run_spec_benchmark, standalone_eval
from pa4t3.data import Item, QuotaSchedule, extract_math_answer, score_math, score_short, short_format_ok
from pa4t3.distill import DistillConfig, distill, forward_kl_sum, merge_adapter
from pa4t3.lm import CachedLM, TableLM
from pa4t3.lora import LoRALinear, inject_lora, lora_parameters, lora_state, merge_into
from pa4t3.specdec import autoregressive_generate, residual_distribution, speculative_generate
from pa4t3.validate import (check_identical_models_accept_all, check_merge, check_rollback_and_split_prefill, check_speculative_cache,
                                check_tokenizers, run_all)


@pytest.fixture(scope="module")
def models():
    t = make_tiny_model(seed=1, layers=3, hidden=128, heads=4, kv_heads=2, dtype=torch.float32)
    d = make_tiny_model(seed=2, layers=2, hidden=64, heads=2, kv_heads=1, dtype=torch.float32)
    return t, d


# ------------------------------------------------------------------------------------------------- chi-square toy
def test_exact_law_is_a_distribution():
    law = toy.exact_law()
    assert abs(law.sum() - 1) < 1e-12 and law.min() * 100_000 >= 10


def test_residual_distribution():
    p = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    q = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
    r = residual_distribution(p, q)
    assert torch.allclose(r, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
    assert torch.allclose(residual_distribution(p, p), p)                           # no positive mass -> p (never used: always accepted)


def test_chi_square_correct_not_rejected_and_buggy_rejected_full_100k():
    res = toy.run_toy_validation(100_000)
    assert res["correct"]["n"] == 100_000 and res["correct"]["dof"] == 26
    assert res["correct"]["p_value"] >= 0.01, res
    assert res["buggy_residual_is_p"]["p_value"] < 0.01, res
    assert res["passed"]


@pytest.mark.parametrize("gamma", [1, 3])
def test_chi_square_other_gammas(gamma):
    c = toy.sample_counts(30_000, gamma=gamma, seed=11)
    assert toy.pearson_test(c)["p_value"] > 0.001


# ------------------------------------------------------------------------------------------------- cache / merge
def test_rollback_and_split_prefill(models):
    t, d = models
    for m in (t, d):
        r = check_rollback_and_split_prefill(m, 10, 250, "cpu")
        assert r["passed"], r


def test_rollback_check_fp16_two_level_and_detects_bug():
    from pa4common.tiny import make_tiny_model
    m = make_tiny_model(seed=101, layers=3, hidden=128, heads=4, kv_heads=2, dtype=torch.float16)
    r = check_rollback_and_split_prefill(m, 10, 250, "cpu")
    assert r["passed"] and r["logic_passed"] and r["fp16_passed"], r
    assert r["fp32_logic_max_abs_rollback"] < 2e-3
    # a genuine rollback bug (truncate keeps one stale token) must be caught by the FP32 logic gate
    from pa4t3 import lm as lm_mod
    orig = lm_mod.CachedLM.truncate
    lm_mod.CachedLM.truncate = lambda self, n: orig(self, min(n + 1, len(self.tokens)))
    try:
        with pytest.raises(AssertionError):
            check_rollback_and_split_prefill(m, 10, 250, "cpu")        # (length assertion fires first)
    finally:
        lm_mod.CachedLM.truncate = orig


def test_speculative_cache_consistency_and_identical_models(models):
    t, d = models
    r = check_speculative_cache(t, d, [5, 17, 29, 41, 53], "cpu")
    assert r["passed"], r
    assert check_identical_models_accept_all(t, [5, 17, 29, 41, 53], "cpu")["passed"]


def test_merge_matches_unmerged_and_uses_fresh_backbone():
    def fresh():
        return make_tiny_model(seed=2, layers=2, hidden=64, heads=2, kv_heads=1, dtype=torch.float32)
    r = check_merge(fresh, "cpu", 10, 250)
    assert r["passed"] and r["max_abs_adapter_effect"] > 1e-3, r


def test_lora_structure():
    m = make_tiny_model(seed=3, layers=2, hidden=64, heads=2, kv_heads=1, dtype=torch.float32)
    n_base = sum(p.numel() for p in m.parameters())
    mods = inject_lora(m, 8, 16.0)
    assert len(mods) == 2 * 4
    ps = lora_parameters(m)
    assert all(p.dtype == torch.float32 and p.requires_grad for p in ps)
    assert not any(p.requires_grad for n, p in m.named_parameters() if "lora_" not in n)
    # zero-initialised B: the adapted model is initially identical to the base
    m0 = make_tiny_model(seed=3, layers=2, hidden=64, heads=2, kv_heads=1, dtype=torch.float32)
    ids = torch.randint(10, 250, (1, 12))
    assert torch.allclose(m(input_ids=ids).logits, m0(input_ids=ids).logits, atol=1e-6)
    # q:(64->64) k,v:(64->32) o:(64->64); r=8 -> 8*(in+out) per module
    assert sum(p.numel() for p in ps) == 2 * 8 * ((64 + 64) + (64 + 32) + (64 + 32) + (64 + 64))
    assert all(isinstance(mod, LoRALinear) and mod.scaling == 2.0 for mod in mods.values())


def test_tokenizer_check():
    tk = ByteChatTokenizer()
    assert check_tokenizers(tk, tk)["passed"]


# ------------------------------------------------------------------------------------------------- spec decoding
def test_speculative_output_distribution_matches_target_on_real_model_tiny(models):
    """First-token law of speculative sampling (draft != target) equals the target's next-token distribution."""
    t, d = models
    tgt, drf = CachedLM(t), CachedLM(d)
    prompt = [5, 17, 29, 41]
    p0 = torch.softmax(tgt.full_logits(prompt)[-1].double(), -1)
    top = torch.topk(p0, 8).indices.tolist()
    n = 2500
    counts = np.zeros(len(p0))
    gen = torch.Generator().manual_seed(7)
    for _ in range(n):
        out, _ = speculative_generate(tgt, drf, prompt, 3, 1, gen, timed=False)
        counts[out[0]] += 1
    # compare on the top-8 tokens + 'other' bucket (Pearson, loose threshold; chi2 df = 8)
    exp = np.append(p0[top].numpy() * n, (1 - p0[top].sum().item()) * n)
    obs = np.append(counts[top], n - counts[top].sum())
    chi2 = ((obs - exp) ** 2 / np.maximum(exp, 1e-9)).sum()
    from scipy import stats
    assert stats.chi2.sf(chi2, len(exp) - 1) > 1e-3


def test_autoregressive_and_speculative_eos_and_caps(models):
    t, d = models
    tgt, drf = CachedLM(t), CachedLM(d)
    gen = torch.Generator().manual_seed(1)
    toks, st = speculative_generate(tgt, drf, [5, 6, 7], 4, 10, gen, eos_ids=frozenset(), timed=False)
    assert len(toks) == 10
    eos = int(toks[3])
    toks2, _ = speculative_generate(tgt, drf, [5, 6, 7], 4, 50, torch.Generator().manual_seed(1), eos_ids=frozenset({eos}), timed=False)
    assert toks2[-1] == eos and eos not in toks2[:-1]
    ar, info = autoregressive_generate(tgt, [5, 6, 7], 12, torch.Generator().manual_seed(1), timed=False)
    assert len(ar) == 12 and info["n_tokens"] == 12
    g1, _ = autoregressive_generate(tgt, [5, 6, 7], 6, torch.Generator().manual_seed(1), temperature=0, timed=False)
    g2, _ = autoregressive_generate(tgt, [5, 6, 7], 6, torch.Generator().manual_seed(2), temperature=0, timed=False)
    assert g1 == g2


def test_breakeven_alpha_monotone_and_none():
    a = breakeven_alpha(4, 0.3, 1.2)
    assert 0 < a < 1 and abs(1 + sum(a ** j for j in range(1, 5)) - (4 * 0.3 + 1.2)) < 1e-9
    assert breakeven_alpha(2, 5.0, 5.0) is None
    assert breakeven_alpha(2, 0.0, 0.5) == 0.0


# ------------------------------------------------------------------------------------------------- data helpers
def test_scoring_and_quota():
    assert extract_math_answer("so the answer is \\boxed{1,234}.") == "1234"
    assert extract_math_answer("... #### 18") == "18"
    assert extract_math_answer("we get 3 then 7.50 apples") == "7.5"
    assert score_math("The answer is 42.", "42")["correct"] == 1 and score_math("none", "42")["correct"] == 0
    assert short_format_ok("March", 0) and not short_format_ok("It is March", 0)
    assert score_short("AU", "AU", 2)["correct"] == 1 and score_short("au", "AU", 2)["format_ok"] == 0
    m = [Item(f"m{i}", "math", f"q{i}", "1") for i in range(5)]
    s = [Item(f"s{i}", "short", f"q{i}", "x", 0) for i in range(3)]
    sc = QuotaSchedule(m, s, 6304, {"math": 30, "short": 10}, {"math": 10, "short": 4})
    seen = []
    while not sc.exhausted():
        it, cap, left = sc.next()
        seen.append(it.domain)
        sc.consume(it.domain, min(cap, left))
    assert sc.quota == {"math": 0, "short": 0} and seen[0] == "math"
    sc2 = QuotaSchedule(m, s, 6304, {"math": 30, "short": 10}, {"math": 10, "short": 4})
    a, b = QuotaSchedule(m, s, 6304), QuotaSchedule(m, s, 6304)
    assert [a.next()[0].id for _ in range(1)] == [b.next()[0].id for _ in range(1)]


# ------------------------------------------------------------------------------------------------- banks
def test_divergences_basic():
    tl = torch.tensor([[2.0, 0.0, -1.0]])
    d = divergences(tl, tl)
    assert d["kl_fwd"].abs().max() < 1e-6 and d["tv"].abs().max() < 1e-6 and d["accept"].item() == pytest.approx(1.0, abs=1e-6)
    ql = torch.tensor([[-1.0, 0.0, 2.0]])
    d = divergences(tl, ql)
    p, q = torch.softmax(tl, -1), torch.softmax(ql, -1)
    assert d["tv"].item() == pytest.approx(0.5 * (p - q).abs().sum().item(), abs=1e-6) and d["greedy_agree"].item() == 0.0
    assert d["kl_fwd"].item() == pytest.approx((p * (p / q).log()).sum().item(), abs=1e-5)


def test_positions_and_bank_alignment_end_to_end(models, tmp_path):
    t, d = models
    tok = ByteChatTokenizer()
    pos = sample_positions("X", "b", 200)
    assert len(pos) == 48 and all(1 <= p <= 192 for p in pos)
    assert sample_positions("X", "b", 20) == sample_positions("X", "b", 20) and max(sample_positions("X", "b", 20)) <= 20
    items = [Item("a", "math", "What is two plus two?", "4"), Item("b", "short", "Name a month.", "May", 0, "familiar")]
    tgt, drf = CachedLM(t), CachedLM(d)
    recs = make_bank("teacher", items, tok, lambda p, cap, g: autoregressive_generate(tgt, p, 40, g, timed=False)[0], "cpu")
    write_banks(tmp_path / "b.jsonl.gz", recs)
    recs = load_banks(tmp_path / "b.jsonl.gz")
    rows = evaluate_alignment(tgt, {"original": drf, "self": CachedLM(t)}, tok, recs)
    assert len(rows) == 2 * len(recs)
    selfrows = [r for r in rows if r["draft"] == "self"]
    assert max(r["kl_fwd"] for r in selfrows) < 1e-4 and min(r["accept"] for r in selfrows) > 0.9999
    assert np.mean([r["accept"] for r in rows if r["draft"] == "original"]) < 0.99
    summ = summarize_alignment(rows, "original", n_boot=200)
    assert "vs_baseline_paired_bootstrap" in summ["teacher"]["self"]


# ------------------------------------------------------------------------------------------------- distillation + bench
@pytest.mark.parametrize("mode", ["off_policy", "on_policy"])
def test_distillation_loop_quota_updates_and_learning(mode, tmp_path):
    tok = ByteChatTokenizer()
    teacher_model = make_tiny_model(seed=11, layers=3, hidden=128, heads=4, kv_heads=2, dtype=torch.float32)
    student = make_tiny_model(seed=12, layers=2, hidden=64, heads=2, kv_heads=1, dtype=torch.float32)
    m = [Item(f"m{i}", "math", f"Compute {i} plus {i}.", "0") for i in range(6)]
    s = [Item(f"s{i}", "short", f"Say word {i}.", "x", 0) for i in range(4)]
    cfg = DistillConfig(seed=6304, quotas={"math": 160, "short": 48}, caps={"math": 24, "short": 8}, update_tokens=64, lr=3e-3, amp=False)
    sm = distill(mode, student, CachedLM(teacher_model), tok, m, s, cfg, tmp_path / mode, frozenset(), "cpu", log=lambda *a: None)
    assert sm["scored_tokens"] == 208 and sm["scored_tokens_by_domain"] == {"math": 160, "short": 48}
    assert sm["updates"] >= 208 // 64 and sm["skipped_or_overflow_updates"] == 0
    assert all(sm["times_s"][k] >= 0 for k in sm["times_s"])
    if mode == "off_policy":
        assert sm["times_s"]["teacher_generation_s"] > 0 and sm["times_s"]["student_rollout_s"] == 0
    else:
        assert sm["times_s"]["student_rollout_s"] > 0 and sm["times_s"]["teacher_generation_s"] == 0
    assert sm["trainable_params"] > 0 and (tmp_path / mode / "adapter.pt").exists()


def test_distillation_reduces_kl_and_merge_equivalence(tmp_path):
    """Off-policy training on a fixed pool drives the token-KL down; the merged student reproduces the trained logits."""
    tok = ByteChatTokenizer()
    teacher_model = make_tiny_model(seed=21, layers=3, hidden=128, heads=4, kv_heads=2, dtype=torch.float32)
    base = lambda: make_tiny_model(seed=22, layers=2, hidden=64, heads=2, kv_heads=1, dtype=torch.float32)
    student = base()
    m = [Item(f"m{i}", "math", f"Compute {i} plus {i}.", "0") for i in range(4)]
    s = [Item(f"s{i}", "short", f"Say word {i}.", "x", 0) for i in range(3)]
    cfg = DistillConfig(seed=6304, quotas={"math": 600, "short": 200}, caps={"math": 30, "short": 10}, update_tokens=32, lr=1e-2, amp=False)
    sm = distill("off_policy", student, CachedLM(teacher_model), tok, m, s, cfg, tmp_path / "x", frozenset(), "cpu", log=lambda *a: None)
    first, last = np.mean(sm["first_updates_kl"]), np.mean(sm["last_updates_kl"])
    assert last < first
    state = torch.load(tmp_path / "x" / "adapter.pt")
    merged, secs = merge_adapter(base(), state, cfg, "cpu")
    assert secs >= 0
    ids = torch.randint(10, 250, (1, 30))
    assert (student(input_ids=ids).logits - merged(input_ids=ids).logits).abs().max() < 1e-3


def test_spec_benchmark_records_and_analysis(models, tmp_path):
    t, d = models
    tok = ByteChatTokenizer()
    items = [Item(f"i{k}", "math" if k % 2 == 0 else "short", f"Question number {k}?", "1", 0, "familiar") for k in range(4)]
    tgt = CachedLM(t)
    drafts = {"orig": CachedLM(d), "self": CachedLM(t)}
    recs = run_spec_benchmark(tgt, drafts, tok, items, {99}, "cpu", tmp_path / "r.jsonl", gammas=(2, 4), reps=2, max_new=12, warmup=1)
    assert len(recs) == 2 * 4 * 2 * 3
    # counterbalancing: both block orders occur
    assert {r["block_reversed"] for r in recs} == {True, False}
    res = analyze_spec(recs, n_boot=200)
    for name in drafts:
        for g in ("2", "4"):
            e = res[name]["gammas"][g]
            assert e["c"] > 0 and e["v_gamma"] > 0 and e["S_hat_analytic"] > 0 and e["S_measured_decode"] > 0
            assert set(e["timers_total_s"]) >= {"t_draft_s", "t_verify_s", "t_accept_s", "t_cache_s", "t_prefill_s"}
    assert res["self"]["gammas"]["4"]["alpha_mle"] == 1.0
    # resumable: rerun adds nothing
    n = len(recs)
    recs2 = run_spec_benchmark(tgt, drafts, tok, items, {99}, "cpu", tmp_path / "r.jsonl", gammas=(2, 4), reps=2, max_new=12, warmup=1)
    assert len(recs2) == n
    ev = standalone_eval(tgt, tok, items, {99}, "cpu", caps={"math": 8, "short": 4})
    assert len(ev) == 4 and all("correct" in r for r in ev)
    be = breakeven_requests({"times_s": {"a": 10.0, "b": 5.0}}, 1.0, 0.5)
    assert be["one_time_cost_s"] == 16.0 and be["break_even_requests"] == 32.0
    assert breakeven_requests({"times_s": {"a": 1.0}}, 0.0, -0.1)["break_even_requests"] is None



def test_pipeline_smoke_end_to_end(tmp_path):
    from pa4t3.pipeline import main
    out = str(tmp_path / "t3")
    main(["all", "--smoke", "--out", out, "--toy-n", "20000"])
    rep = json.load(open(tmp_path / "t3" / "task3_report.json"))
    assert rep["step6_speculative"]["original"]["gammas"]["4"]["alpha_mle"] >= 0
    assert set(rep["distillation"]) == {"off_policy_s6304", "off_policy_s6305", "on_policy_s6304", "on_policy_s6305"}
    assert all(v["skipped_or_overflow_updates"] == 0 and v["scored_tokens"] == 160 for v in rep["distillation"].values())
    assert "deployment_break_even" in rep and "step7_standalone" in rep and "step7_fixed64_throughput" in rep
    assert (tmp_path / "t3" / "alignment_summary.json").exists()
    # the gate really gates
    import pytest
    (tmp_path / "t3" / "validation.json").write_text(json.dumps({"passed": False}))
    with pytest.raises(SystemExit):
        main(["spec_bench", "--smoke", "--out", out])


import json  # noqa: E402


def test_generate_generator_compat_shim_is_deterministic_and_matches_global_seed():
    from pa4t3 import data_compat
    data_compat.install()
    from transformers import GenerationMixin
    assert GenerationMixin.generate._pa4_generator_shim
    m = make_tiny_model(seed=101, layers=2, hidden=64, heads=2, kv_heads=1)
    ids = torch.tensor([[5, 17, 29]])
    kw = dict(max_new_tokens=12, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, pad_token_id=2)

    def run(seed):
        g = torch.Generator().manual_seed(seed)
        return m.generate(ids, generator=g, **kw)[0].tolist(), g.get_state()

    a, ga = run(7)
    b, _ = run(7)
    c, _ = run(8)
    assert a == b and a != c                                    # seeded per call, deterministic
    torch.manual_seed(7)
    assert a == GenerationMixin.generate._pa4_orig(m, ids, **kw)[0].tolist()   # == plain global-seeded sampling
    g = torch.Generator().manual_seed(7)
    assert not torch.equal(g.get_state(), ga)                   # generator state advanced (stream continues across calls)
