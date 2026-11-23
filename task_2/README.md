# Task 2 — Efficient attention and context memory

Part A exact attention operators and benchmark, Part B MQAR associative recall, Part C pretrained KV-cache compression (Qwen2.5-1.5B-Instruct).

## Layout
```
src/pa4t2/    attention_ops, bench_attention (Part A)  |  mqar, mqar_generator (staff, vendored), mqar_models, mqar_train, mqar_analysis (Part B)
              kvcache/{data, quant, policies, evaluate, bench} (Part C; the cache engine is common/pa4common/qwen_engine.py)
              jobs / run_all (scheduler entry)
scripts/      build_notebook.py          notebooks/  pa4_task2_kaggle.ipynb
tests/        51 CPU tests (operators vs fp64 reference, MQAR invariants/kernels, quantizer packing, engine vs HF, policies, end-to-end tiny run)
```
Protocols are constants in `mqar_train.py` and `kvcache/`; there is no config folder.

## Run
```
cd task_2
python -m pa4t2.bench_attention [--smoke] --out results/partA
python -m pa4t2.mqar_train --stage {b1,b2,b3,sweep,analyze,all} [--smoke] --out results/partB
python -m pa4t2.kvcache.evaluate [--smoke] --out results/partC          # C1-C5 on the 180 prompts; --limit N for a quick look
python -m pa4t2.kvcache.bench --out results/partC/packed_benchmark.json
python -m pa4t2.run_all [--smoke] --gpus 0,1                            # everything as resumable jobs (Kaggle notebook)
```

## Design notes
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


## Open interpretation points
1. MQAR "K classification logits": default is 64-way value classification at the query position (`stored_k` variant available).
2. B3 intervention: delta rule (+ additive L2 control), chosen by the dev rule written in `mqar_train.py`.
3. Part C question suffix = `QUESTION_BEGIN ...` through the generation prompt; reference-answer NLL = NLL of the gold five digits.
