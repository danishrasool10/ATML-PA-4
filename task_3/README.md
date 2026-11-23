# Task 3 — Policy distillation and exact speculative decoding

Qwen2.5-1.5B-Instruct (target) and Qwen2.5-0.5B-Instruct (draft, LoRA r=8): exact speculative sampling with KV rollback, off-/on-policy distillation, prefix-bank alignment,
merged speculative execution with a 4-timer decomposition, throughput / standalone quality and deployment break-even.

## Layout
```
src/pa4t3/    specdec (exact sampler + timers), toy (3-symbol Markov chi-square), lm (cached LM with rollback / TableLM), lora (hand-rolled LoRA, merge),
              distill (off-/on-policy), banks (prefix-bank alignment), bench (steps 2/6/7, break-even), validate (Step 1 gate), data, pipeline (stages CLI),
              jobs / run_all (scheduler entry)
scripts/      build_notebook.py          notebooks/  pa4_task3_kaggle.ipynb
tests/        21 CPU tests (chi-square 100,000, rollback, merged-vs-unmerged, distillation loop, benchmarks, end-to-end smoke pipeline)
```

## Run
```
cd task_3
python -m pa4t3.pipeline validate --out results            # Step 1 gate (must pass before any sweep)
python -m pa4t3.pipeline all --smoke --out /tmp/t3 --toy-n 20000
python -m pa4t3.run_all [--smoke] --gpus 0,1                 # data -> gate -> 4 distillations -> banks -> standalone -> benchmarks -> report (Kaggle notebook)
# stages: validate | distill --mode off_policy|on_policy --seed S | banks | spec_bench | throughput | standalone | analyze
```

## Design notes
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


## Open interpretation points
Standalone quality is evaluated greedily (512 / 32 new tokens); an update accumulates whole responses, so it has >= 256 scored tokens.
