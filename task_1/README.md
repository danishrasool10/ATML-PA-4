# Task 1 — Pruning and quantization of Qwen2.5-1.5B

Magnitude (P1), Wanda (P2) and nonuniform (P3) unstructured pruning, the 2:4 diagnostic, and uniform INT4 / AWQ-style (Q2) / mixed-precision (Q3) quantization, with
WikiText-2 / C4 perplexity, HellaSwag, exact storage accounting and paired bootstrap CIs.

## Layout
```
configs/      task1.json (run config), task1_selection_registry.json (PRE-REGISTERED P3/Q3 candidates + selection criterion)
src/pa4t1/    model_utils (eligible units, tied params, snapshot/restore)  storage  evaluation (PPL, HellaSwag)  profiling  calibration
              pruning  sensitivity  allocation (power-law + equal-slope MCKP)  quant  awq  runner (all conditions)  analysis  cli
              manifests (loaders for the real locked layouts)  jobs / run_all (scheduler entry)
scripts/      build_notebook.py
notebooks/    pa4_task1_kaggle.ipynb
tests/        64 CPU tests (enumeration, storage vs real packing, PPL counts, exact sparsity, AWQ invariance, budgets, locked-layout loaders, ...)
```

## Run
```
cd task_1
python -m pa4t1.cli --smoke --stage all                       # tiny model, synthetic manifests (numbers meaningless)
python -m pa4t1.cli --config configs/task1.json --stage all   # full suite (GPU; needs `setup_data.py --public`); or stage by stage:
#   baseline | p1 | p2 | structured | sensitivity | select_p3 | p3 | q1 | q2 | qtables | select_q3 | q3 | analyze
python -m pa4t1.run_all                                       # the same, as resumable per-stage jobs (what the Kaggle notebook calls)
```
Outputs: `results/conditions/<id>/{result.json, per_window_and_example.npz, hellaswag_scores.jsonl, per_unit_records.json, ...}`, `results/{sensitivity,selection,analysis}/`,
`environment.json`, `enumeration.json`. Manifest paths are in `configs/task1.json` (`data/public_locked/*_locked.jsonl.gz`, resolved against `$PA4_DATA_ROOT`).

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


## Open interpretation point
Wanda statistics come from dense-propagated activations (same inputs for every sparsity and method), not from already-pruned earlier layers.
