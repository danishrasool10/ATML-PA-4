"""Resumable multi-GPU job scheduler shared by the three task notebooks (each task defines its jobs in ``pa4tN/jobs.py``).

Every experiment is one shell command (a *job*) that is already resumable on its own; the scheduler adds
  * dependency handling (data first, distillation before banks/benchmarks, ...),
  * one worker per GPU (``CUDA_VISIBLE_DEVICES`` pinned per job) plus one CPU-only slot for data / test jobs,
  * a *quiet phase*: timing-critical jobs (attention benchmark, MQAR inference sweep, packed-KV benchmark, speculative-decoding
    benchmarks) run alone after all other jobs finished, so no other process competes for the host CPU while they are timed,
  * done-markers (``<results>/.jobs/<name>.done``) so a restarted session continues where it stopped, and one log per job.
Training / evaluation jobs that also record timings (Task 1 latency, Part C per-prompt times, distillation time accounting) run with
the other GPU lane active; the lane layout is stored in ``<results>/.jobs/schedule.json`` so the report can state it.

    cd task_3 && python -m pa4t3.run_all --gpus 0,1                 # full task run (from the task folder)
    cd task_3 && python -m pa4t3.run_all --smoke --gpus ""          # CPU plumbing check of the task on tiny models
    python -m pa4t3.run_all --list | --only t3_distill_off_policy_6304 | --force t3_banks | --deadline-hours 11
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


def cli(full_jobs, smoke_jobs, argv=None, task_dir: str | None = None) -> int:
    """``full_jobs(data, out)`` / ``smoke_jobs(data, out)`` -> list[Job]. Runs from ``--repo`` (default: current directory = the task folder)."""
    from .paths import data_root

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="working directory of the jobs (the task folder)")
    ap.add_argument("--data", default=data_root(), help="course data repo checkout (atml-pa4-datasets)")
    ap.add_argument("--gpus", default="0,1", help='comma-separated GPU ids; "" = CPU only')
    ap.add_argument("--smoke", action="store_true", help="tiny-model plumbing check of the whole task")
    ap.add_argument("--only", nargs="*", help="run these jobs (and their dependencies)")
    ap.add_argument("--force", nargs="*", default=[], help="re-run these jobs")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--deadline-hours", type=float, default=None, help="stop launching jobs after this many hours; kill the rest after a 15 min grace")
    a = ap.parse_args(argv)
    repo = Path(a.repo).resolve()
    out = "results_smoke" if a.smoke else "results"
    data = str(Path(a.data).resolve())
    jobs = smoke_jobs(data, out) if a.smoke else full_jobs(data, out)
    if a.list:
        for j in jobs:
            print(f"{j.name:28s} {'quiet ' if j.quiet else '      '}{'gpu' if j.gpu else 'cpu'}  deps={j.deps}  {j.note}")
        return 0
    gpus = [g for g in a.gpus.split(",") if g != ""]
    root = repo.parent
    env_extra = {"PYTHONPATH": os.pathsep.join([str(root / "common"), str(repo / "src")] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])),
                 "PA4_DATA_ROOT": data}
    if not a.smoke:
        env_extra["HF_HOME"] = os.environ.get("HF_HOME", "/tmp/hf")
    summary = run(jobs, repo, repo / out, gpus, env_extra, set(a.only) if a.only else None, set(a.force), deadline_h=a.deadline_hours)
    bad = {n: s for n, s in summary.items() if s in ("failed", "blocked", "interrupted", "not_started")}
    print("\nSUMMARY:", json.dumps(summary, indent=1))
    return 1 if bad else 0
