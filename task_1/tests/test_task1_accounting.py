"""Eligible-weight enumeration, tied parameters, and storage accounting."""
import math

import numpy as np
import pytest
import torch

from pa4common.tiny import make_tiny_model
from pa4t1.constants import GROUP_SIZE
from pa4t1.model_utils import enumerate_report, enumerate_units, unique_params
from pa4t1.quant import (pack_bits, pack_matrix, quant_packed_bytes, quantize_groupwise, unpack_bits, unpack_matrix)
from pa4t1 import storage

QWEN25_1P5B = dict(vocab_size=151936, hidden_size=1536, intermediate_size=8960, num_hidden_layers=28, num_attention_heads=12,
                   num_key_value_heads=2, tie_word_embeddings=True, max_position_embeddings=32768)


def test_eligible_count_matches_formula_tiny(model):
    cfg = model.config
    units = enumerate_units(model)
    assert len(units) == 7 * cfg.num_hidden_layers
    hd = cfg.hidden_size // cfg.num_attention_heads
    kv = cfg.num_key_value_heads * hd
    h, i = cfg.hidden_size, cfg.intermediate_size
    per_layer = 2 * h * h + 2 * kv * h + 3 * h * i
    assert sum(u.numel for u in units) == per_layer * cfg.num_hidden_layers
    rep = enumerate_report(model)
    assert rep["eligible_params"] == per_layer * cfg.num_hidden_layers
    # embeddings / norms / biases / lm_head never eligible
    assert set(rep["by_class"]) >= {"eligible_weight", "embedding", "norm", "bias"}
    assert "lm_head" not in rep["by_class"]  # tied: the head is the embedding tensor, counted once


def test_real_qwen25_1p5b_architecture_counts_without_weights():
    """Instantiate the real architecture on the meta device (no download) and check counts."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(**QWEN25_1P5B)
    with torch.device("meta"):
        m = Qwen2ForCausalLM(cfg)
    rep = enumerate_report(m)
    assert rep["n_units"] == 196
    assert rep["eligible_params"] == 1_310_195_712
    assert rep["tied_lm_head_embedding"] is True
    assert rep["unique_params_total"] == 1_543_714_304  # tied embedding/lm_head counted once
    assert rep["params_if_tied_double_counted"] == rep["unique_params_total"] + 151936 * 1536  # naive count would double-count the tied head
    assert rep["by_class"]["embedding"]["params"] == 151936 * 1536
    assert "lm_head" not in rep["by_class"]
    # every eligible d_in is a multiple of 128 so Qwen2.5-1.5B has no partial quantization group
    for u in enumerate_units(m):
        assert u.shape[1] % GROUP_SIZE == 0
        assert u.shape[1] % 4 == 0  # 2:4 needs d_in divisible by 4


def test_tied_parameters_counted_once():
    tied = make_tiny_model(tie=True)
    untied = make_tiny_model(tie=False)
    assert tied.lm_head.weight is tied.model.embed_tokens.weight
    assert untied.lm_head.weight is not untied.model.embed_tokens.weight
    d_t = storage.dense_report(tied, enumerate_units(tied))
    d_u = storage.dense_report(untied, enumerate_units(untied))
    emb = tied.model.embed_tokens.weight.numel()
    assert d_u["total_params_unique"] - d_t["total_params_unique"] == emb
    assert d_t["dense_fp16_total_bytes"] == 2 * sum(p.numel() for _, p in unique_params(tied))


def test_sparse_storage_formula(model, units):
    # zero out ~half of every matrix deterministically
    for u in units:
        u.weight.data[:, ::2] = 0
    rep = storage.sparse_report(model, units)
    expect = 0
    for u in units:
        nnz = int(torch.count_nonzero(u.weight))
        expect += math.ceil((u.numel + 16 * nnz) / 8)
    assert rep["eligible_sparse_bytes"] == expect
    assert rep["sparse_total_bytes"] == expect + rep["other_fp16_bytes"]
    # 50% sparsity => 1 + 16*0.5 = 9 bits per weight
    assert abs(rep["nominal_bits_per_eligible_weight"] - 9.0) < 0.01


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("shape", [(16, 256), (8, 200), (5, 130)])
def test_packed_bytes_equal_actual_packed_length(bits, shape):
    torch.manual_seed(0)
    W = torch.randn(*shape) * 0.05
    qr = quantize_groupwise(W, bits, GROUP_SIZE)
    blobs = pack_matrix(qr)
    got = sum(len(v) for v in blobs.values())
    formula = quant_packed_bytes(shape, bits, GROUP_SIZE)
    assert got == formula["total"]
    assert (len(blobs["payload"]), len(blobs["scale"]), len(blobs["zero_point"])) == (formula["payload"], formula["scale"], formula["zero_point"])
    back = unpack_matrix(blobs, shape, bits, GROUP_SIZE)
    assert torch.equal(back, qr.deq)  # packed -> dequant reproduces the evaluated weights exactly


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_bit_packing_roundtrip(bits):
    rng = np.random.default_rng(0)
    v = rng.integers(0, 2 ** bits, size=1001).astype(np.uint8)
    assert np.array_equal(unpack_bits(pack_bits(v, bits), bits, len(v)), v)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_quantizer_error_bound_and_zero_representable(bits):
    torch.manual_seed(1)
    W = torch.randn(6, 256) * 0.05
    W[:, 5] = 0.0
    qr = quantize_groupwise(W, bits, 128)
    assert qr.q.max() <= 2 ** bits - 1 and qr.zero.max() <= 2 ** bits - 1
    s = qr.scale.float().repeat_interleave(128, dim=1)
    err = (qr.deq - W).abs()
    assert (err <= s / 2 + 1e-6).all()          # round-to-nearest bound inside the representable range
    assert torch.equal(qr.deq[:, 5], torch.zeros(6))  # zero is exactly representable (z is an integer)


def test_quantizer_8bit_nearly_lossless_vs_3bit():
    torch.manual_seed(2)
    W = torch.randn(32, 256) * 0.05
    e = {b: (quantize_groupwise(W, b).deq - W).pow(2).mean().item() for b in (8, 4, 3, 2)}
    assert e[8] < e[4] < e[3] < e[2]


def test_dequantised_weights_fit_fp16_pipeline():
    W = (torch.randn(8, 128) * 0.02).half()
    qr = quantize_groupwise(W, 4, 128)
    assert qr.scale.dtype == torch.float16
    # scale used for dequantisation is exactly the stored fp16 scale
    assert torch.equal(((qr.q.float() - qr.zero.float()) * qr.scale.float()), qr.deq)
