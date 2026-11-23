"""Generates the per-task Kaggle notebooks (called by ``task_N/scripts/build_notebook.py``; same pattern as ATML-PA3)."""
from __future__ import annotations

import json
from pathlib import Path

CODE_URL = "https://github.com/danishrasool10/ATML-PA-4.git"
DATA_URL = "https://github.com/AbDu11aHHH/atml-pa4-datasets.git"


def _md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def _code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}


def build(out_path: str, task: int, module: str, title: str, what_runs: str, gpus_used: str, extra_notes: str = "") -> Path:
    T = f"task_{task}"
    cells = [
        _md(f"""
# ATML PA4 · Task {task} — {title}
**CS-6304 / EE-5102** · code: https://github.com/danishrasool10/ATML-PA-4 (`{T}/`, shared code in `common/`)

### Before running
1. **Settings → Accelerator: GPU T4 ×2**, **Internet: On**.
2. **Add-ons → Secrets**: add `HF_TOKEN` (Hugging Face read token, notebook access on). Optional `GH_TOKEN` (GitHub token; only for a private repo or `PUSH_RESULTS`).
3. Run the cells top to bottom, or **Save Version → Save & Run All (Commit)** to run unattended.

### What happens
{what_runs}

GPU use: {gpus_used}. FP16 everywhere (T4 has no bf16). Everything is resumable: finished jobs are skipped when you run the main cell again. The runner stops
launching jobs after `DEADLINE_HOURS` (Kaggle's limit is 12 h); to continue, **Save Version**, add that output as an *Input* and set `RESTORE_DIR`.
{extra_notes}
"""),
        _code("""
import os, subprocess, sys, shutil, json, time
import torch
n = torch.cuda.device_count()
print("GPUs:", n, [torch.cuda.get_device_name(i) for i in range(n)])
print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True, capture_output=True, text=True).stdout)
assert n >= 1, "turn on a GPU accelerator (T4 x2) in the notebook settings"
"""),
        _md("## 1. Settings"),
        _code(f"""
TASK = {task}
MODULE = "{module}"
WORK = "/kaggle/working"
CODE_URL = "{CODE_URL}"
CODE_REF = "main"
DATA_URL = "{DATA_URL}"
GPUS = ",".join(str(i) for i in range(min(2, n)))   # "0,1"
DEADLINE_HOURS = 11.0        # stop launching jobs after this; running jobs get a 15 min grace
RESTORE_DIR = ""             # e.g. "/kaggle/input/<previous-output>/ATML-PA-4/{T}/results" to continue an earlier session
RUN_SMOKE = False            # True: first run this task on tiny models (CPU plumbing check, no real results)
PUSH_RESULTS = False         # True: commit {T}/results to main (needs a GH_TOKEN secret with write access)
REPO = f"{{WORK}}/ATML-PA-4"
DATA = f"{{WORK}}/atml-pa4-datasets"
TASK_DIR = f"{{REPO}}/{T}"
"""),
        _md("## 2. Secrets, code, dependencies"),
        _code("""
secrets = {}
try:
    from kaggle_secrets import UserSecretsClient
    us = UserSecretsClient()
    for k in ("HF_TOKEN", "GH_TOKEN"):
        try:
            secrets[k] = us.get_secret(k)
        except Exception:
            print(f"secret {k}: not set")
except ImportError:
    secrets = {k: os.environ[k] for k in ("HF_TOKEN", "GH_TOKEN") if k in os.environ}
if "HF_TOKEN" in secrets:
    os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"] = secrets["HF_TOKEN"]
print("HF token set:", "HF_TOKEN" in secrets)

def clone(url, dest, ref=None):
    auth = url.replace("https://", f"https://x-access-token:{secrets['GH_TOKEN']}@") if "GH_TOKEN" in secrets else url
    if os.path.isdir(dest + "/.git"):
        subprocess.run(["git", "-C", dest, "fetch", "-q", auth] + ([ref] if ref else []), check=True)
        subprocess.run(["git", "-C", dest, "checkout", "-q", "-B", ref or "main", "FETCH_HEAD"], check=True)
    else:
        subprocess.run(["git", "clone", "-q"] + (["-b", ref] if ref else []) + [auth, dest], check=True)
        subprocess.run(["git", "-C", dest, "remote", "set-url", "origin", url], check=True)   # keep the token out of .git/config

clone(CODE_URL, REPO, CODE_REF)
clone(DATA_URL, DATA)
print(subprocess.run(f"git -C {REPO} log --oneline -1; git -C {DATA} log --oneline -1", shell=True, capture_output=True, text=True).stdout)
"""),
        _code("""
# keep Kaggle's CUDA torch; install the rest (pins transformers==4.57.1, huggingface_hub==0.36.2)
tv = subprocess.run([sys.executable, "-c", "import torch;print(torch.__version__.split('+')[0])"], capture_output=True, text=True).stdout.strip()
open("/tmp/constraints.txt", "w").write(f"torch=={tv}\\n")
r = subprocess.run(f"pip install -q -c /tmp/constraints.txt -r {REPO}/requirements.txt -r {DATA}/requirements-public.txt", shell=True, capture_output=True, text=True)
print(r.stdout[-1200:], r.stderr[-1200:])
print(subprocess.run([sys.executable, "-c", "import torch,transformers,huggingface_hub as h;print('torch',torch.__version__,'transformers',transformers.__version__,'hub',h.__version__)"], capture_output=True, text=True).stdout)
from huggingface_hub import login
if "HF_TOKEN" in secrets:
    login(token=secrets["HF_TOKEN"], add_to_git_credential=False)
os.environ["PA4_DATA_ROOT"] = DATA
os.environ["PYTHONPATH"] = f"{REPO}/common:{TASK_DIR}/src"
sys.path[:0] = [f"{REPO}/common", f"{TASK_DIR}/src"]
os.makedirs(f"{TASK_DIR}/results", exist_ok=True)
if RESTORE_DIR and os.path.isdir(RESTORE_DIR):
    shutil.copytree(RESTORE_DIR, f"{TASK_DIR}/results", dirs_exist_ok=True)
    print("restored", RESTORE_DIR)
"""),
        _md("## 3. Optional: plumbing check on tiny models (CPU, no real results)"),
        _code("""
if RUN_SMOKE:
    !cd {TASK_DIR} && python -u -m {MODULE}.run_all --smoke --gpus ""
"""),
        _md("## 4. Run the task"),
        _code("""
!cd {TASK_DIR} && python -u -m {MODULE}.run_all --gpus {GPUS} --deadline-hours {DEADLINE_HOURS}
"""),
        _md("## 5. What finished / what failed"),
        _code("""
p = f"{TASK_DIR}/results/.jobs/summary.json"
if os.path.exists(p):
    s = json.load(open(p))
    for k, v in s.items():
        print(f"{v:12s} {k}")
    todo = [k for k, v in s.items() if v != "done"]
    print("\\nNOT DONE:", todo or "nothing")
    for k in todo:
        lf = f"{TASK_DIR}/results/logs/{k}.log"
        if os.path.exists(lf):
            print(f"\\n--- {k} (last lines) ---\\n" + "".join(open(lf).readlines()[-15:]))
"""),
        _md("## 6. Package the results\nWrites `/kaggle/working/pa4_task{N}_results.zip`; press *Save Version* afterwards and download it from the Output tab.".replace("{N}", str(task))),
        _code("""
os.chdir(WORK)
shutil.make_archive(f"{WORK}/pa4_task{TASK}_results", "zip", TASK_DIR, "results")
print(subprocess.run(f"du -sh {WORK}/pa4_task{TASK}_results.zip; du -sh {TASK_DIR}/results/*/ 2>/dev/null | sort -h | tail", shell=True, capture_output=True, text=True).stdout)
"""),
        _code("""
# optional: commit the results to main (like ATML-PA3 keeps task_N/results in the repo). Needs GH_TOKEN with write access.
if PUSH_RESULTS:
    assert "GH_TOKEN" in secrets, "add a GH_TOKEN secret first"
    auth = CODE_URL.replace("https://", f"https://x-access-token:{secrets['GH_TOKEN']}@")
    sh = (f"cd {REPO} && git add -f task_{TASK}/results && git -c user.name=kaggle-run -c user.email=noreply@example.com "
          f"commit -qm 'Task {TASK}: Kaggle 2xT4 results' && git pull -q --rebase {auth} main && git push -q {auth} HEAD:main")
    r = subprocess.run(sh, shell=True, capture_output=True, text=True)
    print((r.stdout + r.stderr).replace(secrets["GH_TOKEN"], "***")[-800:])
"""),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}, "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True}},
          "nbformat": 4, "nbformat_minor": 5}
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(nb, indent=1))
    return p
