"""Task 3 job list (``python -m pa4t3.run_all``): Step 1 gate -> 4 distillations -> banks / standalone, then the timing benchmarks."""
from __future__ import annotations

from pa4common.scheduler import Job

DISTILL = [f"distill_{m}_{s}" for m in ("off_policy", "on_policy") for s in (6304, 6305)]


def full_jobs(data: str, out: str = "results") -> list[Job]:
    d = f"--data-root {data} --out {out}"
    P = "python -m pa4t3.pipeline"
    J = [Job("tests", "python -m pytest tests -q", gpu=False, fatal=False, note="CPU unit tests"),
         Job("data_public", f"cd {data} && python scripts/setup_data.py --public", gpu=False, note="GSM8K locked files (+ the Task 1 corpora)"),
         Job("data_check", f"cd {data} && python scripts/check_setup.py", deps=["data_public"], gpu=False, fatal=False),
         Job("data_prefix_banks", f"cd {data} && python scripts/setup_data.py --prefix-banks", note="staff teacher / original-draft prefix banks (GPU)"),
         Job("validate", f"{P} validate {d}", deps=["data_public"], note="Step 1 gate: chi-square 100k, rollback, merged-vs-unmerged, tokenizers")]
    for m in ("off_policy", "on_policy"):
        for s in (6304, 6305):
            J.append(Job(f"distill_{m}_{s}", f"{P} distill {d} --mode {m} --seed {s}", deps=["validate"], note="Steps 3/4"))
    J += [Job("banks", f"{P} banks {d}", deps=DISTILL + ["data_prefix_banks"], note="Step 5 prefix-bank alignment"),
          Job("standalone", f"{P} standalone {d}", deps=DISTILL, note="Step 7 standalone quality"),
          Job("spec_bench", f"{P} spec_bench {d}", deps=DISTILL, quiet=True, note="Steps 2 and 6"),
          Job("throughput", f"{P} throughput {d}", deps=DISTILL, quiet=True, note="Step 7 fixed-64 throughput"),
          Job("analyze", f"{P} analyze {d}", deps=["banks", "standalone", "spec_bench", "throughput"], quiet=True, note="report + deployment break-even")]
    return J


def smoke_jobs(data: str, out: str = "results_smoke") -> list[Job]:
    return [Job("s_tests", "python -m pytest tests -q", gpu=False, fatal=False),
            Job("s_t3", f"python -m pa4t3.pipeline all --smoke --toy-n 20000 --out {out}", gpu=False)]
