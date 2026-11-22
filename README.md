# ATML PA4 (EE-5102 / CS-6304, Fall 2026) — Task 1: Pruning and Quantization of Qwen2.5-1.5B

Status: Task 1 pipeline implemented and unit-tested on CPU with a tiny random Qwen2. **No results from the real
model are included yet**: the full runs need a CUDA GPU (T4) and the course manifests.

## Layout
```
pa4/common/   seed (6304), env/checkpoint metadata, CUDA-synced timing + peak memory, paired bootstrap, JSON/JSONL I/O
pa4/data/     manifest loaders (schema assumptions documented in manifests.py), tiny offline stand-ins for tests
pa4/task1/    model_utils (eligible units, tied params, snapshot/restore)   storage (dense / 16-bit sparse / packed bytes)
              evaluation (PPL, HellaSwag)  profiling  calibration (layer-wise Gram statistics)  pruning  sensitivity
              allocation (power-law + equal-slope MCKP)  quant  awq  runner (all conditions)  analysis  cli
configs/      task1.json (run config), task1_selection_registry.json (PRE-REGISTERED P3/Q3 candidates + criterion)
tests/        63 tests (enumeration, storage vs real packing, PPL counts, exact sparsity, AWQ invariance, budgets, ...)
```
Tasks 2 and 3 will reuse `pa4/common` (seeding, metadata, timing, bootstrap, result I/O).

## Reproduce
```
pip install -r requirements.txt
python scripts/setup_data.py --public            # in the course repo atml-pa4-datasets; point configs/task1.json "manifests" at its output
python -m pa4.data.inspect_manifests <manifest>   # check layout against pa4/data/manifests.py
python -m pytest tests -q                         # CPU sanity tests (also run before the full experiment)
python -m pa4.task1.cli --smoke --stage all       # plumbing check on a tiny model (numbers meaningless)
python -m pa4.task1.cli --config configs/task1.json --stage all      # full suite (GPU); or stage by stage:
#   baseline | p1 | p2 | structured | sensitivity | select_p3 | p3 | q1 | q2 | qtables | select_q3 | q3 | analyze
```
Outputs: `results/task1/conditions/<id>/{result.json, per_window_and_example.npz, hellaswag_scores.jsonl, per_unit_records.json, ...}`,
`results/task1/{sensitivity,selection,analysis}/`, `environment.json`, `enumeration.json`.

## Design decisions (all fixed before any held-out result)
* **Units / conventions.** Eligible = q,k,v,o,gate,up,down of every block (196 projections, 1,310,195,712 weights for Qwen2.5-1.5B; tied
  embedding/LM head counted once; asserted in tests against the real architecture on the meta device). Every condition restarts from the
  pristine FP16 snapshot (fingerprint-checked).
* **Calibration.** All statistics come from the 128 calibration sequences propagated through the *dense* blocks one by one
  (Gram matrices X^T X, so layer-output errors are exact: tr(dW G dW^T)). Wanda norms = sqrt(diag G). Deviation from the Wanda reference
  code: activations come from the dense model, not from already-pruned earlier layers (same inputs for every sparsity and every method).
* **Pruning.** Per-output-row ranking, k = round(s·d_in) zeros per row, stable index tie-break; 2:4 keeps the 2 best of each 4 contiguous
  input channels. P1/P2/P3 zero *exactly* the same number of eligible weights (asserted).
* **P3.** Unit = individual projection, s in [0.10, 0.85] (integer row counts, exact repair so the total equals the P1/P2 budget).
  Candidate designs (registry file): OWL outlier ratio + power-law rule (γ ∈ {0.5,1,2}); local layer-output error table (relative / absolute) and
  Fisher-Taylor predicted loss table + equal-slope multiple-choice-knapsack allocation. Selection by mean NLL on the 32-sequence development
  manifest only; all candidates and their dev scores are retained. Test stages refuse to run without the lock file.
* **Quantizer (shared by Q1/Q2/Q3).** Group 128 along input columns, asymmetric min/max (range includes 0), FP16 scale, integer zero-point,
  round-half-even, no clipping search. Packed bytes = b-bit bit-packed payload + 2 B/group scale + b bits/group zero-point, each stream
  byte-aligned per matrix; verified against actual bit-packing (`verify_packing`). Weights are de-quantized to FP16 for evaluation (fake quantization).
* **Q2 (AWQ-style).** Scale search over α ∈ {0,0.1,…,1} on local output error per scale group, exact inverse folding into RMSNorm / v_proj / up_proj
  (GQA handled by tying scales across heads sharing a KV head), per-block full-precision invariance check before rounding. Not reproduced:
  AWQ clipping search, block-output objective, fused kernels. Compensation scale bytes = 0 (everything folds).
* **Q3.** Unit = projection, bits ∈ {2,3,4,8}, same AWQ quantizer; designs: local error tables or HAWQ-V2-style Fisher-trace × weight MSE,
  allocated by equal-slope MCKP under the *total* packed-byte budget of Q2 INT4 / INT3 (incl. metadata and FP16 non-eligible parameters).
* **Limitations to state in the report.** Sensitivities are isolated per-unit probes (dense inputs) and ignore interactions between compressed layers.
  Nothing is run with sparse or low-bit kernels: latency / peak memory are for dense FP16 tensors and logical bytes are modelled storage.

## Attribution
Algorithms follow Sun et al. 2024 (Wanda), Lin et al. 2024 (AWQ), Yin et al. 2024 (OWL), Dong et al. 2020 (HAWQ-V2), Molchanov et al. 2019;
HellaSwag preprocessing/scoring conventions follow EleutherAI lm-evaluation-harness (`hellaswag` task v1.0), re-implemented here without copying code.
Code was written with an LLM coding assistant (Claude Code); the PDF report must be written by the student without generative AI.
