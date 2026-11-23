"""Task 2 job list (``python -m pa4t2.run_all``): Part A, Part B (MQAR), Part C (KV cache)."""
from __future__ import annotations

from pa4common.scheduler import Job


def full_jobs(data: str, out: str = "results") -> list[Job]:
    d = f"--data-root {data}"
    return [
        Job("tests", "python -m pytest tests -q", gpu=False, fatal=False, note="CPU unit tests"),
        Job("c_eval", f"python -m pa4t2.kvcache.evaluate --cases {data}/data/kv_retrieval/kv_retrieval_cases.jsonl.gz --out {out}/partC", note="Part C: C1-C5 (180 prompts)"),
        Job("b_b1", f"python -m pa4t2.mqar_train --stage b1 {d} --out {out}/partB", note="MQAR B1, 3 seeds"),
        Job("b_b2", f"python -m pa4t2.mqar_train --stage b2 {d} --out {out}/partB", note="MQAR B2, 3 seeds"),
        Job("b_b3", f"python -m pa4t2.mqar_train --stage b3 {d} --out {out}/partB", note="MQAR B3, 3 seeds"),
        # quiet phase: timing-critical, run alone on an idle machine
        Job("a_bench", f"python -m pa4t2.bench_attention --out {out}/partA", quiet=True, note="Part A: operators + benchmark"),
        Job("b_sweep", f"python -m pa4t2.mqar_train --stage sweep {d} --out {out}/partB", quiet=True, note="MQAR forward-only inference sweep"),
        Job("c_packed_bench", f"python -m pa4t2.kvcache.bench --out {out}/partC/packed_benchmark.json", quiet=True, note="Part C: packed-cache encode/decode/decode-step"),
        Job("b_analyze", f"python -m pa4t2.mqar_train --stage analyze {d} --out {out}/partB", deps=["b_b1", "b_b2", "b_b3", "b_sweep"], quiet=True, note="MQAR tables, plots, B3 bootstrap"),
    ]


def smoke_jobs(data: str, out: str = "results_smoke") -> list[Job]:
    d = f"--data-root {data}"
    return [Job("s_tests", "python -m pytest tests -q", gpu=False, fatal=False),
            Job("s_a", f"python -m pa4t2.bench_attention --smoke --out {out}/partA", gpu=False),
            Job("s_b", f"python -m pa4t2.mqar_train --smoke --stage all {d} --out {out}/partB", gpu=False),
            Job("s_c", f"python -m pa4t2.kvcache.evaluate --smoke --cases {data}/data/kv_retrieval/kv_retrieval_cases.jsonl.gz --limit 4 --n-boot 200 --validate-n 1 --out {out}/partC", gpu=False),
            Job("s_c_bench", f"python -m pa4t2.kvcache.bench --smoke --reps 3 --out {out}/partC/packed_benchmark.json", gpu=False)]
