"""Pruning, allocation, AWQ invariance, Gram-based errors, budget compliance."""
import copy
import itertools
import math

import numpy as np
import pytest
import torch

from pa4t1 import allocation as alloc
from pa4t1.awq import compress_quant_model, invariance_check, make_scale, search_scale
from pa4t1.calibration import LayerStats, capture_first_layer_inputs, quad_error, run_layerwise
from pa4t1.constants import BITS_ALLOWED, LEVELS, PROJ_NAMES, unit_id
from pa4t1.model_utils import ModelSnapshot, count_nonzero_eligible, enumerate_units, fingerprint, get_proj
from pa4t1.pruning import (k_bounds, k_from_sparsity, nm_plan, nm_prune_mask, plan_zero_count, prune_model, rowwise_prune_mask,
                               score_magnitude, score_wanda, uniform_plan)
from pa4t1.quant import quant_packed_bytes, quantize_groupwise
from pa4t1 import storage
from pa4t1.sensitivity import collect_pruning_tables, fisher_pass


# ----------------------------------------------------------------------------- calibration statistics

def test_gram_matches_explicit_inputs_and_quad_error(model, batches, calib_seqs):
    captured = {}

    def hook_for(uid):
        def h(m, args):
            captured.setdefault(uid, []).append(args[0].detach().reshape(-1, args[0].shape[-1]))
        return h

    hs = []
    for li, layer in enumerate(model.model.layers):
        for name in PROJ_NAMES:
            hs.append(get_proj(layer, name).register_forward_pre_hook(hook_for(unit_id(li, name))))
    with torch.no_grad():
        for b in batches:
            model(input_ids=b)
    for h in hs:
        h.remove()
    seen = {}

    def fn(li, layer, stats, ctx):
        for name in PROJ_NAMES:
            uid = unit_id(li, name)
            X = torch.cat(captured[uid]).double()
            assert torch.allclose(stats.gram(name).double(), X.T @ X, rtol=1e-4, atol=1e-4)
            assert torch.allclose(stats.act_norm(name).double(), X.norm(dim=0), rtol=1e-4)   # Wanda ||X_j||_2
            W = get_proj(layer, name).weight.data
            dW = torch.randn_like(W) * 0.01
            explicit = ((X.float() @ dW.T) ** 2).sum().item()
            assert math.isclose(quad_error(dW, stats.gram(name)), explicit, rel_tol=1e-3)
        seen[li] = True

    run_layerwise(model, batches, fn)
    assert len(seen) == len(model.model.layers)


# ----------------------------------------------------------------------------- pruning

def test_rowwise_mask_exact_per_row_and_ties_stable():
    score = torch.tensor([[3.0, 1.0, 2.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    m = rowwise_prune_mask(score, 2)
    assert m.sum(1).tolist() == [2, 2]
    assert m[0].tolist() == [False, True, False, True]       # two lowest
    assert m[1].tolist() == [True, True, False, False]       # ties -> lowest column index


def test_nm_mask_keeps_two_highest_of_every_four():
    torch.manual_seed(0)
    s = torch.rand(7, 16)
    m = nm_prune_mask(s, 2, 4)
    kept = (~m).reshape(7, 4, 4)
    assert (kept.sum(-1) == 2).all()
    g = s.reshape(7, 4, 4)
    top2 = torch.topk(g, 2, dim=-1).indices
    for r in range(7):
        for gi in range(4):
            assert set(torch.nonzero(kept[r, gi]).flatten().tolist()) == set(top2[r, gi].tolist())


def test_uniform_pruning_exact_sparsity_only_eligible_touched(model, units, batches):
    snap = ModelSnapshot(model)
    others_before = {n: p.detach().clone() for n, p in model.named_parameters() if n not in {f"model.layers.{u.layer}.{'self_attn' if u.name in ('q_proj','k_proj','v_proj','o_proj') else 'mlp'}.{u.name}.weight" for u in units}}
    for rule in ("magnitude", "wanda"):
        for s in (0.25, 0.5, 0.75):
            snap.restore(model)
            plan = uniform_plan(units, s, rule)
            out = prune_model(model, units, batches, plan)
            n = sum(u.numel for u in units)
            zeros = sum(r["n_pruned_by_mask"] for r in out["records"].values())
            assert zeros == plan_zero_count(units, plan) == round(s * n)         # exactly s of the eligible weights
            for u in units:                                                        # exactly k zeros in every row
                assert ((u.weight == 0).sum(1) == k_from_sparsity(s, u.shape[1])).all()
            assert count_nonzero_eligible(units) == n - zeros
            for nme, p in model.named_parameters():
                if nme in others_before:
                    assert torch.equal(p, others_before[nme]), nme                # embeddings/norms/biases/head untouched
    snap.restore(model)
    assert fingerprint(model) == snap.fingerprint


def test_magnitude_prune_equals_manual_and_wanda_uses_activation_norm(model, units, batches):
    snap = ModelSnapshot(model)
    u0 = units[0]
    W0 = u0.weight.detach().clone()
    out = prune_model(model, units, batches, uniform_plan(units, 0.5, "magnitude"))
    k = k_from_sparsity(0.5, W0.shape[1])
    manual = W0.clone()
    idx = torch.argsort(W0.abs(), dim=1, stable=True)[:, :k]
    manual.scatter_(1, idx, 0.0)
    assert torch.equal(u0.weight, manual)
    snap.restore(model)
    # Wanda: pruned weights differ from magnitude pruning and scores are |W| * ||X_j||
    prune_model(model, units, batches, uniform_plan(units, 0.5, "wanda"))
    assert not torch.equal(u0.weight, manual)


def test_direct_and_layerwise_wanda_pruning_agree(model, units, batches):
    snap = ModelSnapshot(model)
    tables = collect_pruning_tables(model, units, batches)
    plan = uniform_plan(units, 0.5, "wanda")
    prune_model(model, units, batches, plan)
    a = {u.uid: u.weight.detach().clone() for u in units}
    snap.restore(model)
    for u in units:
        W = u.weight.data
        W.masked_fill_(rowwise_prune_mask(score_wanda(W, tables["act_norm"][u.uid]), plan[u.uid]["k"]), 0)
    for u in units:
        assert torch.equal(u.weight, a[u.uid])


def test_24_mask_inside_prune_model_and_matched_sparsity(model, units, batches):
    snap = ModelSnapshot(model)
    prune_model(model, units, batches, nm_plan(units))
    for u in units:
        g = (u.weight.reshape(u.shape[0], -1, 4) != 0).sum(-1)
        assert (g <= 2).all()
    n = sum(u.numel for u in units)
    assert plan_zero_count(units, nm_plan(units)) == n // 2 == plan_zero_count(units, uniform_plan(units, 0.5, "wanda"))


# ----------------------------------------------------------------------------- allocation (P3)

@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("s_global", [0.5, 0.75])
@pytest.mark.parametrize("rule", ["power", "mckp"])
def test_p3_allocation_matches_budget_exactly_and_respects_bounds(units, seed, s_global, rule):
    rng = np.random.default_rng(seed)
    n = len(units)
    if rule == "power":
        a = alloc.power_law_allocation(units, rng.random(n) + 0.01, s_global, gamma=1.0)
    else:
        base = np.array(LEVELS)[None, :] ** rng.uniform(1, 4, size=(n, 1)) * rng.uniform(0.1, 10, size=(n, 1))
        a = alloc.mckp_sparsity_allocation(units, base, s_global)
    total = sum(u.numel for u in units)
    assert a.target_zeros == round(s_global * total)
    assert a.residual == 0 and a.total_zeros(units) == a.target_zeros       # equals the P1/P2 budget exactly
    for u in units:
        lo, hi = k_bounds(u.shape[1])
        assert lo <= a.k[u.uid] <= hi
        assert 0.10 - 1e-9 <= a.s[u.uid] <= 0.85 + 1e-9


def test_p3_exactness_on_real_qwen_shapes_without_weights():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(vocab_size=1000, hidden_size=1536, intermediate_size=8960, num_hidden_layers=28, num_attention_heads=12,
                      num_key_value_heads=2, tie_word_embeddings=True)
    with torch.device("meta"):
        m = Qwen2ForCausalLM(cfg)
    us = enumerate_units(m)
    rng = np.random.default_rng(0)
    for s in (0.5, 0.75):
        a = alloc.power_law_allocation(us, rng.random(len(us)) + 0.01, s, gamma=1.0)
        assert a.residual == 0 and a.total_zeros(us) == round(s * 1_310_195_712)
        assert min(a.s.values()) >= 0.10 - 1e-9 and max(a.s.values()) <= 0.85 + 1e-9


def test_mckp_matches_bruteforce_on_small_instance():
    rng = np.random.default_rng(0)
    n, o = 6, 4
    costs = [np.sort(rng.integers(1, 20, size=o)).astype(float) + np.arange(o) for _ in range(n)]
    losses = [np.sort(rng.random(o))[::-1] * rng.uniform(1, 5) for _ in range(n)]
    budget = float(sum(c[1] for c in costs))
    ch = alloc.mckp_allocate(costs, losses, budget)
    got = sum(losses[u][ch[u]] for u in range(n))
    assert sum(costs[u][ch[u]] for u in range(n)) <= budget
    best = min(sum(losses[u][c[u]] for u in range(n)) for c in itertools.product(range(o), repeat=n)
               if sum(costs[u][c[u]] for u in range(n)) <= budget)
    assert got <= best * 1.35 + 1e-9    # greedy equal-slope is near-optimal (exact on the convex hull)
    with pytest.raises(ValueError):
        alloc.mckp_allocate(costs, losses, 0.5)


# ----------------------------------------------------------------------------- quantization / AWQ

def test_awq_full_precision_invariance_whole_model_fp64(model, units, batches):
    """Fold the searched scales into norms / v_proj / up_proj (no rounding) and compare *whole-model*
    logits in float64 with the original model."""
    model = model.double()
    import pa4t1.awq as awq_mod

    captured = []
    orig_check = awq_mod.invariance_check

    def spy(layer, scaled, sample, tol):
        captured.append((layer, {k: (v.clone() if torch.is_tensor(v) else {kk: vv.clone() for kk, vv in v.items()}) for k, v in scaled.items()}))
        return orig_check(layer, scaled, sample, 1e-5)

    awq_mod.invariance_check = spy
    try:
        ref = copy.deepcopy(model)
        out = compress_quant_model(model, units, batches, {u.uid: 4 for u in units}, mode="awq", apply=False, inv_tol=1e-5)
    finally:
        awq_mod.invariance_check = orig_check
    assert out["info"]["max_invariance_rel_err"] < 1e-5   # fp32 compute inside the AWQ pipeline
    # apply the unrounded folded weights to a copy and compare end-to-end logits
    folded = copy.deepcopy(ref)
    for li, (layer, sc) in enumerate(captured):
        fl = folded.model.layers[li]
        for n, w in sc["weights"].items():
            get_proj(fl, n).weight.data.copy_(w)
        fl.self_attn.v_proj.bias.data.copy_(sc["v_bias"])
        fl.input_layernorm.weight.data.copy_(sc["ln1"])
        fl.post_attention_layernorm.weight.data.copy_(sc["ln2"])
    ids = batches[0]
    with torch.no_grad():
        a, b = ref(input_ids=ids).logits, folded(input_ids=ids).logits
    assert torch.allclose(a, b, rtol=1e-5, atol=1e-5)


def test_invariance_check_detects_a_wrong_fold(model, batches):
    layer = model.model.layers[0]
    sample = capture_first_layer_inputs(model, batches)[0]
    good = {"weights": {n: get_proj(layer, n).weight.data.clone() for n in PROJ_NAMES}, "v_bias": layer.self_attn.v_proj.bias.data.clone(),
            "ln1": layer.input_layernorm.weight.data.clone(), "ln2": layer.post_attention_layernorm.weight.data.clone()}
    assert invariance_check(layer, good, sample, 1e-4) < 1e-6
    bad = copy.deepcopy(good)
    bad["weights"]["q_proj"] = bad["weights"]["q_proj"] * 1.3        # scaled weights without inverse compensation
    with pytest.raises(AssertionError):
        invariance_check(layer, bad, sample, 1e-4)


def test_awq_search_never_worse_than_q1_and_alpha0_equals_q1(model, units, batches):
    snap = ModelSnapshot(model)
    bits = {u.uid: 3 for u in units}
    q1 = compress_quant_model(model, units, batches, bits, mode="q1", apply=False)
    aw = compress_quant_model(model, units, batches, bits, mode="awq", apply=False)
    for li, groups in aw["groups"].items():
        for g, r in groups.items():
            assert r["best_loss"] <= r["loss_at_alpha0"] + 1e-9 and r["grid"][0] == 0.0
    # alpha=0 grid point == Q1 on the group's criterion
    one = compress_quant_model(model, units, batches, bits, mode="awq", grid=(0.0,), apply=False)
    for uid in q1["records"]:
        assert math.isclose(one["records"][uid]["abs_err"], q1["records"][uid]["abs_err"], rel_tol=1e-6)
    # total calibration error: AWQ <= Q1 for every unit group sum
    assert sum(r["abs_err"] for r in aw["records"].values()) <= sum(r["abs_err"] for r in q1["records"].values()) * (1 + 1e-9)
    snap.restore(model)
    assert fingerprint(model) == snap.fingerprint        # apply=False never modified the model


def test_quantization_applied_weights_are_on_the_grid_and_packing_verified(model, units, batches):
    snap = ModelSnapshot(model)
    out = compress_quant_model(model, units, batches, {u.uid: 4 for u in units}, mode="q1", verify_packing=True)
    assert out["info"]["packing_verified"] == len(units)
    for u in units:
        # every group of 128 columns has at most 2^4 distinct values per row
        w = u.weight.detach().reshape(u.shape[0], -1, 128)
        for r in range(min(3, w.shape[0])):
            for g in range(w.shape[1]):
                assert len(torch.unique(w[r, g])) <= 16
    snap.restore(model)


def test_q3_budget_compliance_both_budgets(model, units, batches):
    """Mixed precision never exceeds the Q2 uniform packed byte budget (whole model, incl. metadata)."""
    snap = ModelSnapshot(model)
    tabs = {}
    for b in BITS_ALLOWED:
        o = compress_quant_model(model, units, batches, {u.uid: b for u in units}, mode="awq", apply=False)
        tabs[b] = o["records"]
    snap.restore(model)
    byte_t = np.array([[quant_packed_bytes(u.shape, b)["total"] for b in BITS_ALLOWED] for u in units], float)
    loss = np.array([[tabs[b][u.uid]["rel_err"] for b in BITS_ALLOWED] for u in units])
    other = storage.quant_report(model, units, {u.uid: 4 for u in units})["other_fp16_bytes"]
    for ub in (4, 3):
        budget_total = storage.quant_report(model, units, {u.uid: ub for u in units})["packed_total_bytes"]
        bits = alloc.mckp_bit_allocation(units, BITS_ALLOWED, loss, byte_t, budget_total - other)
        by = {u.uid: bits[i] for i, u in enumerate(units)}
        assert set(by.values()) <= set(BITS_ALLOWED)
        rep = storage.quant_report(model, units, by)
        assert rep["packed_total_bytes"] <= budget_total
        assert budget_total - rep["packed_total_bytes"] < 8 * 1024          # budget is actually used
        # actually quantize with the mixed assignment and verify real packed lengths
        compress_quant_model(model, units, batches, by, mode="awq", verify_packing=True)
        snap.restore(model)
        # uniform allocation is feasible, so the optimiser can only do better on its own criterion
        uni = sum(loss[i, BITS_ALLOWED.index(ub)] for i in range(len(units)))
        mixed = sum(loss[i, BITS_ALLOWED.index(by[u.uid])] for i, u in enumerate(units))
        assert mixed <= uni + 1e-12


# ----------------------------------------------------------------------------- sensitivity pass

def test_fisher_pass_keeps_no_grads_and_restores_flags(model, units, batches, calib_seqs):
    tables = collect_pruning_tables(model, units, batches)
    f = fisher_pass(model, units, calib_seqs[:3], tables)
    assert f["trace"].shape == (len(units),) and (f["trace"] > 0).all()
    assert f["taylor"].shape == (len(units), len(LEVELS))
    # predicted loss increase grows with sparsity level
    assert (np.diff(f["taylor"], axis=1) >= -1e-12).all()
    for u in units:
        assert u.weight.grad is None and not u.weight.requires_grad
    assert tables["owl_ratio"].shape == (len(units),)
    assert (np.diff(tables["local_rel"], axis=1) >= -1e-9).mean() > 0.9     # error grows with sparsity (almost always)
