import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pa4.task2.attention_ops import dense_causal_attention, detect_sdpa_backend, sdpa_attention, tiled_online_softmax_attention
from pa4.task2.bench_attention import causality_check, edge_cases, make_qkv, tile_size_check
from pa4.task2.mqar import BINS, REGIMES, check_invariants, evaluate, make_eval_set, train_batch
from pa4.task2.mqar_models import (MQARConfig, MQARTransformer, accounting, delta_rule_attention, dilated_attention_gather, head_masks,
                                   linear_attention_cumsum, linear_attention_parallel, linear_attention_recurrent, window_attention_chunked)
from pa4.task2.mqar_train import PROTOCOL, lr_at

DATA_ROOT = os.environ.get("PA4_DATA_ROOT", str(Path(__file__).resolve().parents[2] / "atml-pa4-datasets"))


def ref_attention(q, k, v):
    return F.scaled_dot_product_attention(q.double(), k.double(), v.double(), is_causal=True)


@pytest.mark.parametrize("L", [37, 128, 300])
def test_three_operators_agree_with_fp64_reference(L):
    q, k, v = make_qkv(L, torch.float32, "cpu", requires_grad=False)
    ref = ref_attention(q, k, v)
    for op in (dense_causal_attention, lambda a, b, c: tiled_online_softmax_attention(a, b, c, 64, 32), sdpa_attention):
        assert (op(q, k, v).double() - ref).abs().max() < 2e-5


@pytest.mark.parametrize("dtype", ["fp32", "fp16"])
def test_tile_sizes_incl_ragged_and_fp16(dtype):
    r = tile_size_check(133, dtype, "cpu", tiles=(16, 50, 64, 200))
    tol = 3e-2 if dtype == "fp16" else 1e-5
    assert all(v["max_abs"] < tol and not v["has_nan"] for v in r["tiles"].values())


@pytest.mark.parametrize("name", ["dense", "tiled", "sdpa"])
def test_causality_by_future_perturbation(name):
    r = causality_check(name, 96, "fp32", "cpu")
    assert r["causal"] and r["perturbation_has_effect"]


def test_gradients_match_dense_fp32_reference():
    q, k, v = make_qkv(96, torch.float32, "cpu")
    go = torch.randn_like(q)
    grads = {}
    for n, op in (("dense", dense_causal_attention), ("tiled", lambda a, b, c: tiled_online_softmax_attention(a, b, c, 32, 32)), ("sdpa", sdpa_attention)):
        q2, k2, v2 = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
        op(q2, k2, v2).backward(go)
        grads[n] = (q2.grad, k2.grad, v2.grad)
    for n in ("tiled", "sdpa"):
        for a, b in zip(grads[n], grads["dense"]):
            assert (a - b).abs().max() < 1e-4


def test_edge_cases_fp32_all_masked_rows_and_extreme_scores():
    e = edge_cases("cpu")["fp32"]
    for n in ("dense", "tiled"):
        assert e["all_masked_rows"][n]["nan_rows"] == 0 and e["all_masked_rows"][n]["masked_rows_output_abs_max"] == 0.0
        assert e["scale_16.0"][n]["max_abs"] < 1e-3 and not e["scale_16.0"][n]["has_nan"]


def test_tiled_fp16_does_not_overflow_where_dense_fp16_may():
    q, k, v = make_qkv(128, torch.float16, "cpu", requires_grad=False, scale=64.0)
    # scores ~ 3e4 are ill-conditioned even in fp32 (fp64-reference error ~5e-2), so compare the fp16 run with the
    # same operator on fp32 copies of the identical inputs: the only difference is the fp16 output/PV rounding.
    out = tiled_online_softmax_attention(q, k, v)
    ref32 = tiled_online_softmax_attention(q.float(), k.float(), v.float())
    assert not torch.isnan(out).any() and not torch.isinf(out).any()
    assert ((out.float() - ref32).abs() <= 2e-3 * ref32.abs() + 1e-2).all()


def test_sdpa_backend_detection_runs():
    q, k, v = make_qkv(64, torch.float32, "cpu", requires_grad=False)
    assert detect_sdpa_backend(q, k, v)["dispatched"] != ""


# ------------------------------------------------------------------ Part B
@pytest.mark.parametrize("K", [4, 8, 16])
@pytest.mark.parametrize("regime", REGIMES)
def test_mqar_frozen_sets_and_train_stream_invariants(K, regime):
    for split in ("dev", "test"):
        b = make_eval_set(K, regime, split, DATA_ROOT, n=64)
        check_invariants(b)
        assert b["tokens"].shape == (64, 256) and b["qpos"].shape == (64, K)
    tr = train_batch(K, regime, 6304, 0, 8)
    check_invariants(tr)
    assert not np.array_equal(tr["tokens"][0], make_eval_set(K, regime, "test", DATA_ROOT, n=1)["tokens"][0])
    if regime == "near":
        assert tr["dist"].max() < 64
    if regime == "far":
        assert tr["dist"].min() >= 128 and tr["dist"].max() <= 250


def test_mqar_train_streams_disjoint_across_seeds():
    a = train_batch(8, "mixed", 6304, 5, 4)["ids"]
    b = train_batch(8, "mixed", 6305, 5, 4)["ids"]
    assert set(a).isdisjoint(b)


def test_linear_attention_parallel_cumsum_recurrent_agree():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 70, 32, dtype=torch.float64) for _ in range(3))
    r = linear_attention_recurrent(q, k, v)
    assert (linear_attention_parallel(q, k, v) - r).abs().max() < 1e-10 and (linear_attention_cumsum(q, k, v) - r).abs().max() < 1e-10
    r2 = linear_attention_recurrent(q, k, v, normalised=False, l2=True)
    assert (linear_attention_parallel(q, k, v, False, True) - r2).abs().max() < 1e-10


def test_window_and_dilated_efficient_kernels_match_mask_forms():
    torch.manual_seed(1)
    q, k, v = (torch.randn(2, 4, 300, 32, dtype=torch.float64) for _ in range(3))
    cfg = MQARConfig()
    for variant in ("window", "sparse"):
        m = head_masks(cfg, 300, "cpu", variant)
        ref = F.scaled_dot_product_attention(q, k, v, attn_mask=m[None])
        if variant == "window":
            assert (window_attention_chunked(q, k, v, 64) - ref).abs().max() < 1e-10
        else:
            assert (window_attention_chunked(q[:, :2], k[:, :2], v[:, :2], 64) - ref[:, :2]).abs().max() < 1e-10
            assert (dilated_attention_gather(q[:, 2:], k[:, 2:], v[:, 2:], 4, 64) - ref[:, 2:]).abs().max() < 1e-10


def test_sparse_mask_structure():
    m = head_masks(MQARConfig(), 256, "cpu", "sparse")
    last = 255
    assert m[0, last].nonzero().flatten().tolist() == list(range(192, 256))                    # preceding 64 positions
    assert m[2, last].nonzero().flatten().tolist() == [last - 4 * i for i in range(63, -1, -1)]   # offsets 0,4,...,252
    assert m[2, 100, 100] and not m[2, 100, 101]                                                # causal


def test_delta_rule_overwrites_while_additive_accumulates():
    d = 8
    k = torch.zeros(1, 1, 3, d); k[..., 0] = 5.0                  # identical key direction each step
    q = k.clone()
    v = torch.stack([torch.ones(d), 2 * torch.ones(d), 3 * torch.ones(d)])[None, None]
    beta = torch.ones(1, 1, 3)
    out = delta_rule_attention(q, k, v, beta)
    assert torch.allclose(out[0, 0, 2], 3 * torch.ones(d), atol=1e-5)     # latest value replaces the old ones
    add = linear_attention_parallel(q, k, v, normalised=False, l2=True)
    assert torch.allclose(add[0, 0, 2], 6 * torch.ones(d), atol=1e-5)     # additive write superposes 1+2+3


def test_variants_forward_and_parameter_and_state_accounting():
    base = MQARTransformer(MQARConfig(attn="linear")).n_params()
    x = torch.randint(0, 256, (2, 64))
    for a in ("dense", "window", "linear", "sparse", "linear_l2", "delta"):
        m = MQARTransformer(MQARConfig(attn=a))
        assert m(x).shape == (2, 64, 256)
        assert abs(m.n_params() - base) / base < 0.05                      # B3: within 5% of the baseline parameter count
    acc = {a: accounting(MQARConfig(attn=a), 256) for a in ("dense", "window", "linear", "delta", "sparse")}
    assert acc["dense"]["pairs_per_layer"] == 4 * 256 * 257 // 2
    assert acc["delta"]["state_bytes_per_layer_fp32"] <= acc["linear"]["state_bytes_per_layer_fp32"]   # matched persistent state
    assert acc["window"]["pairs_per_layer"] < acc["dense"]["pairs_per_layer"]
    assert acc["linear"]["pairs_per_layer"] == 0


def test_lr_schedule_and_architecture_spec():
    assert math.isclose(lr_at(0), 8e-4 / 200) and math.isclose(lr_at(199), 8e-4) and math.isclose(lr_at(200), 8e-4)
    assert math.isclose(lr_at(4499), 8e-4 * (1 - 0.85 * (4499 - 200) / 4300), rel_tol=1e-9) and lr_at(4499) > 0.15 * 8e-4
    m = MQARTransformer(MQARConfig())
    assert len(m.blocks) == 2 and m.cfg.d_model == 128 and m.cfg.n_heads == 4 and m.cfg.d_head == 32
    assert m.blocks[0].mlp[0].out_features == 512 and m.emb.num_embeddings == 256
    assert PROTOCOL["updates"] == 4500 and PROTOCOL["batch_size"] == 32 and PROTOCOL["clip"] == 1.0


def test_window_testtime_receptive_field_limit():
    # two layers of window 64 reach at most 2*(64-1) = 126 positions back
    b = make_eval_set(16, "mixed", "test", DATA_ROOT, n=256)
    assert (b["dist"] > 126).any() and (b["dist"] <= 126).any()
    m = MQARTransformer(MQARConfig(attn="dense"))
    r = evaluate(m, b, variant="window", window=64)
    assert r["n_sequences"] == 256 and "bins" in r
