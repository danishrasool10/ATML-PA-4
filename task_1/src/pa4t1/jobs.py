"""Task 1 job list for the resumable scheduler (``python -m pa4t1.run_all``). One stage = one job so a restarted session resumes at stage level."""
from __future__ import annotations

from pa4common.scheduler import Job

STAGES = ["baseline", "p1", "p2", "structured", "sensitivity", "select_p3", "p3", "q1", "q2", "qtables", "select_q3", "q3", "analyze"]


def full_jobs(data: str, out: str = "results") -> list[Job]:
    J = [Job("tests", "python -m pytest tests -q", gpu=False, fatal=False, note="CPU unit tests"),
         Job("data_public", f"cd {data} && python scripts/setup_data.py --public", gpu=False, note="C4 / WikiText-2 / HellaSwag locked files"),
         Job("data_check", f"cd {data} && python scripts/check_setup.py", deps=["data_public"], gpu=False, fatal=False)]
    prev = "data_public"
    for st in STAGES:                                   # sequential chain on one GPU (stages share cached statistics / the selection lock)
        J.append(Job(f"t1_{st}", f"python -m pa4t1.cli --config configs/task1.json --stage {st}", deps=[prev], note=f"Task 1 stage {st}"))
        prev = f"t1_{st}"
    return J


def smoke_jobs(data: str, out: str = "results_smoke") -> list[Job]:
    return [Job("s_tests", "python -m pytest tests -q", gpu=False, fatal=False),
            Job("s_t1", f"python -m pa4t1.cli --smoke --stage all --smoke-dir {out}/t1", gpu=False)]
