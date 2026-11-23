"""Resumable two-GPU job scheduler for the full PA4 run (used by notebooks/kaggle_2xT4.ipynb, also runs anywhere).

Every experiment is one shell command (a *job*) that is already resumable on its own; the scheduler adds
  * dependency handling (data first, distillation before banks/benchmarks, ...),
  * one worker per GPU (``CUDA_VISIBLE_DEVICES`` pinned per job) plus one CPU-only slot for data / test jobs,
  * a *quiet phase*: timing-critical jobs (attention benchmark, MQAR inference sweep, packed-KV benchmark, speculative-decoding
    benchmarks) run alone after all other jobs finished, so no other process competes for the host CPU while they are timed,
  * done-markers (``<results>/.jobs/<name>.done``) so a restarted session continues where it stopped, and one log per job.
Training / evaluation jobs that also record timings (Task 1 latency, Part C per-prompt times, distillation time accounting) run with
the other GPU lane active; the lane layout is stored in ``<results>/.jobs/schedule.json`` so the report can state it.

    python scripts/kaggle_runner.py --repo . --data ../atml-pa4-datasets --gpus 0,1          # full run
    python scripts/kaggle_runner.py --repo . --data ../atml-pa4-datasets --smoke --gpus ""   # CPU plumbing check of the whole suite
    python scripts/kaggle_runner.py ... --list | --only t3_distill_off_6304 | --force t1_all
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Job:
    name: str
    cmd: str
    deps: list[str] = field(default_factory=list)
    gpu: bool = True
    quiet: bool = False          # timing-critical: run alone, after every non-quiet job
    fatal: bool = True           # False: failure is reported but does not block dependents
    note: str = ""


def full_jobs(data: str, out: str = "results") -> list[Job]:
    d = f"--data-root {data}"
    t3 = f"python -m pa4.task3.pipeline"
    distill = [f"t3_distill_{m}_{s}" for m in ("off_policy", "on_policy") for s in (6304, 6305)]
    J = [
        Job("tests", "python -m pytest tests -q", gpu=False, fatal=False, note="CPU unit tests"),
        Job("data_public", f"cd {data} && python scripts/setup_data.py --public", gpu=False, note="C4/WikiText/HellaSwag/GSM8K locked files"),
        Job("data_check", f"cd {data} && python scripts/check_setup.py", deps=["data_public"], gpu=False, fatal=False),
        Job("data_prefix_banks", f"cd {data} && python scripts/setup_data.py --prefix-banks", note="staff teacher/original-draft prefix banks (GPU)"),
        Job("t1_all", f"python -m pa4.task1.cli --config configs/task1.json --stage all", deps=["data_public"], note="Task 1, all stages (longest job: starts first)"),
        Job("t3_validate", f"{t3} validate {d} --out {out}/task3", deps=["data_public"], note="Step 1 gate (chi-square 100k, rollback, merge)"),
    ]
    for m in ("off_policy", "on_policy"):
        for s in (6304, 6305):
            J.append(Job(f"t3_distill_{m}_{s}", f"{t3} distill {d} --out {out}/task3 --mode {m} --seed {s}", deps=["t3_validate"], note="Steps 3/4"))
    J += [
        Job("t3_banks", f"{t3} banks {d} --out {out}/task3", deps=distill + ["data_prefix_banks"], note="Step 5 prefix-bank alignment"),
        Job("t3_standalone", f"{t3} standalone {d} --out {out}/task3", deps=distill, note="Step 7 standalone quality"),
        Job("t2c_eval", f"python -m pa4.task2.kvcache.evaluate --cases {data}/data/kv_retrieval/kv_retrieval_cases.jsonl.gz --out {out}/task2/partC", note="Part C C1-C5"),
        Job("t2b_b1", f"python -m pa4.task2.mqar_train --stage b1 {d} --out {out}/task2/partB", note="MQAR B1 (x3 seeds)"),
        Job("t2b_b2", f"python -m pa4.task2.mqar_train --stage b2 {d} --out {out}/task2/partB", note="MQAR B2 (x3 seeds)"),
        Job("t2b_b3", f"python -m pa4.task2.mqar_train --stage b3 {d} --out {out}/task2/partB", note="MQAR B3 (x3 seeds)"),
        # ---- quiet phase -------------------------------------------------------------------------------------------
        Job("t2a_bench", f"python -m pa4.task2.bench_attention --out {out}/task2/partA", quiet=True, note="Part A"),
        Job("t2b_sweep", f"python -m pa4.task2.mqar_train --stage sweep {d} --out {out}/task2/partB", quiet=True, note="MQAR inference sweep"),
        Job("t2c_packed_bench", f"python -m pa4.task2.kvcache.bench --out {out}/task2/partC/packed_benchmark.json", quiet=True, note="Part C packed-cache benchmark"),
        Job("t3_spec_bench", f"{t3} spec_bench {d} --out {out}/task3", deps=distill, quiet=True, note="Steps 2/6"),
        Job("t3_throughput", f"{t3} throughput {d} --out {out}/task3", deps=distill, quiet=True, note="Step 7 fixed-64 throughput"),
        Job("t2b_analyze", f"python -m pa4.task2.mqar_train --stage analyze {d} --out {out}/task2/partB", deps=["t2b_b1", "t2b_b2", "t2b_b3", "t2b_sweep"], quiet=True, gpu=True,
            note="MQAR tables/plots/bootstrap"),
        Job("t3_analyze", f"{t3} analyze {d} --out {out}/task3", deps=["t3_banks", "t3_standalone", "t3_spec_bench", "t3_throughput"], quiet=True, note="Task 3 report + break-even"),
    ]
    return J


def smoke_jobs(data: str, out: str = "results_smoke") -> list[Job]:
    d = f"--data-root {data}"
    return [
        Job("s_tests", "python -m pytest tests -q", gpu=False, fatal=False),
        Job("s_t1", f"python -m pa4.task1.cli --smoke --stage all --smoke-dir {out}/t1", gpu=False),
        Job("s_t2a", f"python -m pa4.task2.bench_attention --smoke --out {out}/t2a", gpu=False),
        Job("s_t2b", f"python -m pa4.task2.mqar_train --smoke --stage all {d} --out {out}/t2b", gpu=False),
        Job("s_t2c", f"python -m pa4.task2.kvcache.evaluate --smoke --cases {data}/data/kv_retrieval/kv_retrieval_cases.jsonl.gz --limit 4 --n-boot 200 --validate-n 1 --out {out}/t2c", gpu=False),
        Job("s_t2c_bench", f"python -m pa4.task2.kvcache.bench --smoke --reps 3 --out {out}/t2c/packed_benchmark.json", gpu=False),
        Job("s_t3", f"python -m pa4.task3.pipeline all --smoke --toy-n 20000 --out {out}/t3", gpu=False),
    ]


def run(jobs: list[Job], repo: Path, results: Path, gpus: list[str], env_extra: dict, only: set[str] | None, force: set[str], log=print,
        deadline_h: float | None = None, grace_min: float = 15.0) -> dict:
    jd = results / ".jobs"
    logs = results / "logs"
    jd.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    byname = {j.name: j for j in jobs}
    for n in force:
        (jd / f"{n}.done").unlink(missing_ok=True)
    if only:                                              # keep the requested jobs and their dependencies
        keep = set()
        stack = list(only)
        while stack:
            n = stack.pop()
            if n not in keep:
                keep.add(n)
                stack.extend(byname[n].deps)
        jobs = [j for j in jobs if j.name in keep]
    (jd / "schedule.json").write_text(json.dumps({"gpus": gpus, "lanes": "one job per GPU concurrently; quiet jobs run alone on the first GPU",
                                                  "jobs": [j.__dict__ for j in jobs]}, indent=1))
    state = {j.name: ("done" if (jd / f"{j.name}.done").exists() else "pending") for j in jobs}
    lock = threading.Lock()
    running: dict[str, str] = {}                           # job -> slot
    procs: dict[str, subprocess.Popen] = {}
    stopped = [False]
    free_gpus = list(gpus)
    cpu_free = [1 if gpus else 2]                          # CPU-only runs may overlap two jobs
    t0 = time.time()

    def ready(j: Job) -> bool:
        if state[j.name] != "pending":
            return False
        for d in j.deps:
            if d not in state:
                continue
            if state[d] in ("done", "failed_nonfatal"):
                continue
            return False
        return True

    def blocked(j: Job) -> bool:
        return state[j.name] == "pending" and any(state.get(d) in ("failed", "blocked") for d in j.deps)

    def work(j: Job, slot: str):
        env = {**os.environ, "PYTHONUNBUFFERED": "1", **env_extra}
        env["CUDA_VISIBLE_DEVICES"] = "" if slot == "cpu" else slot
        lf = logs / f"{j.name}.log"
        start = time.time()
        with open(lf, "a") as f:
            f.write(f"\n=== {time.ctime()} :: {j.cmd}  (CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']!r})\n")
            f.flush()
            pr = subprocess.Popen(j.cmd, shell=True, cwd=repo, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            procs[j.name] = pr
            rc = pr.wait()
        with lock:
            el = time.time() - start
            procs.pop(j.name, None)
            if rc == 0:
                state[j.name] = "done"
                (jd / f"{j.name}.done").write_text(json.dumps({"seconds": el, "slot": slot, "finished": time.ctime()}))
            elif stopped[0]:
                state[j.name] = "interrupted"                 # killed at the deadline: resumes on the next session
            else:
                state[j.name] = "failed" if j.fatal else "failed_nonfatal"
            running.pop(j.name)
            if slot == "cpu":
                cpu_free[0] += 1
            else:
                free_gpus.append(slot)
            log(f"[{time.time() - t0:7.0f}s] {j.name}: {state[j.name]} ({el:.0f}s, slot {slot}), log: {lf}")
            if rc != 0 and not stopped[0]:
                tail = "".join(open(lf).readlines()[-12:])
                log(f"   last log lines:\n{tail}")

    threads = []
    while True:
        with lock:
            for j in jobs:
                if blocked(j):
                    state[j.name] = "blocked"
                    log(f"[{time.time() - t0:7.0f}s] {j.name}: blocked by failed dependency")
            pending = [j for j in jobs if state[j.name] == "pending"]
            if not pending and not running:
                break
            if deadline_h is not None:
                el_h = (time.time() - t0) / 3600
                if el_h > deadline_h and not stopped[0] and not running and pending:
                    log(f"deadline {deadline_h}h reached: not starting {len(pending)} pending jobs (re-run in a new session to continue)")
                    for j in pending:
                        state[j.name] = "not_started"
                    break
                if el_h > deadline_h + grace_min / 60 and running and not stopped[0]:
                    stopped[0] = True
                    log(f"deadline + {grace_min} min grace reached: terminating {list(running)}")
                    for pr in list(procs.values()):
                        try:
                            os.killpg(os.getpgid(pr.pid), signal.SIGTERM)
                        except ProcessLookupError:
                            pass
            if deadline_h is not None and (time.time() - t0) / 3600 > deadline_h:
                time.sleep(1.0)
                continue                                          # past the deadline: only wait for running jobs
            normal_pending = [j for j in pending if not j.quiet]
            quiet_running = any(byname[n].quiet for n in running)
            launched = False
            for j in pending:
                if quiet_running or not ready(j):
                    continue
                use_gpu = j.gpu and bool(gpus)
                if j.quiet:
                    if normal_pending or running:             # quiet jobs wait for a fully idle machine
                        continue
                    slot = gpus[0] if use_gpu else "cpu"
                elif use_gpu:
                    if not free_gpus:
                        continue
                    slot = free_gpus[0]
                else:
                    if cpu_free[0] < 1:
                        continue
                    slot = "cpu"
                if slot == "cpu":
                    cpu_free[0] -= 1
                else:
                    free_gpus.remove(slot)
                state[j.name] = "running"
                running[j.name] = slot
                log(f"[{time.time() - t0:7.0f}s] start {j.name} on {slot}  -- {j.note}")
                th = threading.Thread(target=work, args=(j, slot), daemon=True)
                th.start()
                threads.append(th)
                launched = True
                if j.quiet:
                    break
            if not launched and not running and pending and not any(ready(j) for j in pending):
                log("deadlock: nothing runnable, remaining: " + ", ".join(j.name for j in pending))
                break
        time.sleep(1.0)
    for t in threads:
        t.join()
    summary = {n: s for n, s in state.items()}
    (jd / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--data", default="../atml-pa4-datasets", help="dataset repo checkout (relative paths are resolved from --repo)")
    ap.add_argument("--gpus", default="0,1", help='comma-separated GPU ids; "" = CPU only')
    ap.add_argument("--smoke", action="store_true", help="tiny-model plumbing check of the whole suite")
    ap.add_argument("--only", nargs="*", help="run these jobs (and their dependencies)")
    ap.add_argument("--force", nargs="*", default=[], help="re-run these jobs")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--deadline-hours", type=float, default=None, help="stop launching jobs after this many hours; kill the rest after a 15 min grace")
    a = ap.parse_args(argv)
    repo = Path(a.repo).resolve()
    out = "results_smoke" if a.smoke else "results"
    jobs = smoke_jobs(a.data, out) if a.smoke else full_jobs(a.data, out)
    if a.list:
        for j in jobs:
            print(f"{j.name:22s} {'quiet ' if j.quiet else '      '}{'gpu' if j.gpu else 'cpu'}  deps={j.deps}  {j.note}")
        return 0
    gpus = [g for g in a.gpus.split(",") if g != ""]
    env_extra = {"HF_HOME": os.environ.get("HF_HOME", "/tmp/hf")} if not a.smoke else {}
    summary = run(jobs, repo, repo / out, gpus, env_extra, set(a.only) if a.only else None, set(a.force), deadline_h=a.deadline_hours)
    bad = {n: s for n, s in summary.items() if s in ("failed", "blocked", "interrupted", "not_started")}
    print("\nSUMMARY:", json.dumps(summary, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
