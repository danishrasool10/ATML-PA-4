# ATML PA4 (EE-5102 / CS-6304, Fall 2026) — code for Tasks 1, 2 and 3

Status: all three tasks are implemented and unit-tested on CPU with tiny random Qwen2 models and a byte tokenizer, and every pipeline has a
`--smoke` mode. **No results from the real models are included yet**: the full runs need a CUDA GPU (T4 or better), network access to
Hugging Face, and the locked course data (`atml-pa4-datasets`, `setup_data.py --public` and `--prefix-banks`).

## Layout
```
pa4/common/   seed (6304), env/checkpoint metadata, CUDA-synced timing + peak memory, paired bootstrap, JSON/JSONL I/O
pa4/data/     manifest loaders (schema assumptions documented in manifests.py), tiny offline stand-ins for tests
pa4/task1/    model_utils (eligible units, tied params, snapshot/restore)   storage (dense / 16-bit sparse / packed bytes)
              evaluation (PPL, HellaSwag)  profiling  calibration (layer-wise Gram statistics)  pruning  sensitivity
              allocation (power-law + equal-slope MCKP)  quant  awq  runner (all conditions)  analysis  cli
configs/      task1.json (run config), task1_selection_registry.json (PRE-REGISTERED P3/Q3 candidates + criterion)
pa4/task2/    attention_ops + bench_attention (Part A) | mqar, mqar_models, mqar_train, mqar_analysis (Part B)
              kvcache/{data,quant,engine,policies,evaluate,bench} (Part C)
pa4/task3/    lora, lm (cached LM with rollback), specdec (exact speculative sampler), toy (chi-square), validate, data,
              distill (off/on-policy), banks (prefix-bank alignment), bench (steps 2/6/7), pipeline (CLI)
tests/        ~140 tests (Task 1 accounting/AWQ/budgets; Part A/B operators; Part C quantizers + engine-vs-HF; Task 3 chi-square 100k,
              rollback, merge, distillation, benchmarks, end-to-end smoke)
```
All tasks reuse `pa4/common` (seeding, metadata, CUDA-synced timing, bootstrap, result I/O). Data paths: the Task 1 config has a
`data_root` (default `../atml-pa4-datasets`), the Task 2/3 CLIs take `--data-root` / `--cases`.

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

## Task 2 — Efficient attention and context memory
```
python -m pa4.task2.bench_attention [--smoke]                     # Part A: dense / tiled online-softmax / SDPA, B=2,H=4,d=64, L=256..8192, FP16+FP32
python -m pa4.task2.mqar_train --stage all --data-root ../atml-pa4-datasets [--smoke]    # Part B: B1/B2/B3 x seeds 6304-6306, inference sweep, analysis
python -m pa4.task2.mqar_train --stage analyze                     # final tables, learning curves, distance bins, paired bootstrap (B3) only
python -m pa4.task2.kvcache.evaluate --cases ../atml-pa4-datasets/data/kv_retrieval/kv_retrieval_cases.jsonl.gz --out results/task2_partC
python -m pa4.task2.kvcache.bench --out results/task2_partC/packed_benchmark.json
```
* **Part A.** The tiled operator keeps fp32 accumulators and up-casts tiles before QK^T, so fp16 inputs cannot overflow where the plain dense
  operator can; the tiled backward is plain autograd (not a fused kernel) and says so in the results. SDPA backends are detected with the profiler
  and forced-backend probing; OOM / unsupported conditions are recorded, not skipped.
* **Part B.** Uses the staff generator (vendored unchanged in `pa4/data/mqar_generator.py`) and the frozen dev/test records; invariants are checked before
  training. Loss/accuracy are query-position classification over the 64 value ids (`value64`; `stored_k` restricts to the K stored values). B3 intervention:
  delta-rule (overwriting) linear attention with a matched persistent state, plus an additive L2-normalised control; the prediction is written before evaluation.
* **Part C.** `kvcache/engine.py` re-implements the Qwen2 decoder forward on top of the HF weights with an explicit per-layer cache (absolute RoPE position ids
  for every token, physical slot independent), SDPA attention, and an FP32 attention-capture pass for the question suffix. C1 validates it against the
  unmodified HF forward/`generate` (`c1_validation.json`). Prompt = chat template; the cache that gets compressed is the prefix before `QUESTION_BEGIN`
  (its length L is the budget argument); the suffix (question + generation prompt) is processed on the compressed cache. Evidence offsets are used only to
  measure retention. Budget `M_4(L)`; eviction/hybrid retention is **shared across layers and heads**, 2 B per retained index charged to the budget; keys are
  stored post-RoPE; quantization of new tokens happens when 64 aligned slots are full (hybrid keeps new tokens FP16). Case `...-00` of each cell is the dev case
  (pipeline debugging only; nothing is tuned) and the report is given on all 180 prompts and on the 162 non-dev prompts.

## Task 3 — Policy distillation and speculative decoding
```
python -m pa4.task3.pipeline validate    --data-root ../atml-pa4-datasets --out results/task3     # Step 1 gate (chi-square 100k, rollback, merge, tokenizers)
python -m pa4.task3.pipeline all         --data-root ../atml-pa4-datasets --out results/task3     # distill x4 -> banks -> step 6 -> step 7 -> report
python -m pa4.task3.pipeline all --smoke --out /tmp/t3 --toy-n 20000                              # CPU plumbing check
```
Stages can be run separately (`distill --mode off_policy --seed 6304`, `banks`, `spec_bench`, `throughput`, `standalone`, `analyze`); everything is
resumable (JSONL appends). Sweeps refuse to run until `results/task3/validation.json` says `passed`.
* **Sampler.** `specdec.speculative_generate`: min(1, p/q) acceptance, normalised positive-residual resampling, bonus token, rollback by truncating both caches
  to `len(sequence) - 1` (cache slot == absolute position). The buggy control (`residual_mode='p'`) is rejected by the chi-square test (p << 0.01) and the exact
  sampler is not (100,000 length-3 sequences, 26 d.o.f., min expected count > 10).
* **LoRA** is hand-rolled (no `peft`): FP32 A/B on q/k/v/o, r=8, alpha=16, dropout 0; merging adds `(alpha/r) B A` to a *fresh* FP16 backbone.
* **Distillation.** Same seeded prompt schedule, quotas (30k math + 10k short teacher-scored tokens), caps 192/32, update every >= 256 scored tokens (whole
  responses accumulated; gradient divided by the number of supervised tokens before unscale/clip), AdamW 3e-4/0.01, FP16 autocast with dynamic loss scaling,
  clip 1.0, normalised forward token-KL over the full vocabulary. Off-policy replays a frozen teacher pool; on-policy samples each response from the current
  student. Skipped/overflow updates are logged. No checkpoint selection: the final adapter is used.
* **Timers (Step 6).** draft proposal / target verification / acceptance+residual / cache rollback+advancement + prefill, CUDA-synchronised; blocks of
  {target-only, gamma=2, gamma=4} per prompt and repetition in counterbalanced order. c, v_gamma, alpha (MLE: accepted / (accepted + rejection events)), analytic
  S_hat, measured S (decode-only and end-to-end), break-even alpha. Absolute latencies are those of this reference implementation (python loop, `torch.cat`
  cache), not of a production decoder.
* **Banks.** Staff `teacher` / `original_draft` banks are used when `prefix_banks_locked.jsonl.gz` exists (else regenerated with the same position rule);
  `exact_spec_original_gamma4` and one own-policy bank per trained student are generated here and frozen; all drafts are evaluated on all banks (cross-evaluation).

## Running everything on Kaggle (2 x T4)
`notebooks/kaggle_2xT4.ipynb` clones both repos, installs the pinned dependencies, logs into Hugging Face from the Kaggle secret `HF_TOKEN`, builds the locked data
(`setup_data.py --public` and `--prefix-banks`) and then calls `scripts/kaggle_runner.py`, a resumable scheduler with one lane per GPU: Task 1, the Task 3
validation gate -> four distillations -> banks, Part C and MQAR training run concurrently; the timing-critical benchmarks (Part A, MQAR inference sweep, packed-KV
benchmark, speculative decoding / throughput) run afterwards on an otherwise idle machine, then the analysis jobs. `--deadline-hours` stops cleanly before Kaggle's
12 h limit and the next session resumes (results restored through `RESTORE_DIR`). `python scripts/kaggle_runner.py --smoke --gpus ""` runs the whole suite on tiny models.

## Open interpretation points (flagged for the report)
1. MQAR "K classification logits": default is 64-way value classification at the query position (`stored_k` variant available).
2. B3 intervention: delta rule (+ additive L2 control), chosen by the dev rule written in `mqar_train.py`.
3. Part C question suffix = `QUESTION_BEGIN ...` through the generation prompt; reference-answer NLL = NLL of the gold five digits.
4. Task 3 standalone quality is evaluated greedily (512 / 32 new tokens); updates accumulate whole responses so an update has >= 256 tokens.
5. Task 1: Wanda statistics from dense-propagated activations (see above).

## Attribution
Algorithms follow Sun et al. 2024 (Wanda), Lin et al. 2024 (AWQ), Yin et al. 2024 (OWL), Dong et al. 2020 (HAWQ-V2), Molchanov et al. 2019;
HellaSwag preprocessing/scoring conventions follow EleutherAI lm-evaluation-harness (`hellaswag` task v1.0), re-implemented here without copying code.
Code was written with an LLM coding assistant (Claude Code); the PDF report must be written by the student without generative AI.
