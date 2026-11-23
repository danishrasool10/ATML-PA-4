# ATML PA4 — CS-6304 / EE-5102 (LUMS)

```
common/            shared, task-agnostic code (import name: pa4common)
  pa4common/seed.py, results.py, timing.py, bootstrap.py, env_meta.py, paths.py
                           seeding (6304), JSON/JSONL I/O, CUDA-synced timing + peak memory, paired bootstrap, environment metadata, data-repo lookup
  pa4common/qwen_engine.py explicit-KV-cache Qwen2 forward (used by task_2 Part C and task_3 speculative decoding)
  pa4common/scheduler.py   resumable multi-GPU job scheduler with a quiet phase for timing-critical jobs (used by every task's run_all)
  pa4common/notebook.py    generator of the per-task Kaggle notebooks;  tiny.py: tiny offline Qwen2 / byte tokenizer for tests and --smoke
task_1/            model compression: pruning + quantization of Qwen2.5-1.5B (see task_1/README.md)
  notebooks/pa4_task1_kaggle.ipynb
task_2/            efficient attention and context memory: operators, MQAR, KV-cache compression (see task_2/README.md)
  notebooks/pa4_task2_kaggle.ipynb
task_3/            policy distillation and exact speculative decoding (see task_3/README.md)
  notebooks/pa4_task3_kaggle.ipynb
requirements.txt   pinned transformers==4.57.1 / huggingface_hub==0.36.2 (required by Task 2 Part C), the rest from the course environment
```

Each `task_N/` has the same shape: `README.md`, `configs/`, `src/pa4tN/` (package), `scripts/build_notebook.py`, `notebooks/`, `tests/`; results go to `task_N/results/`.
The course data repo [`atml-pa4-datasets`](https://github.com/AbDu11aHHH/atml-pa4-datasets) is expected next to this repo (or point `PA4_DATA_ROOT` at it).

## Status
Everything is implemented and unit-tested on CPU with tiny random Qwen2 models and a byte tokenizer; every task has a `--smoke` mode. **No results from the
real models are included yet**: they need a CUDA GPU, Hugging Face access and the locked course data (`setup_data.py --public` / `--prefix-banks`, run by the notebooks).

## Run a task on Kaggle (2 x T4)
Open `task_N/notebooks/pa4_task{N}_kaggle.ipynb` on Kaggle, set **Accelerator: GPU T4 x2**, **Internet: On**, add the secret **`HF_TOKEN`**, then *Save & Run All* (or run the
cells). The notebook clones this repo (`main`) and the dataset repo, installs the pinned dependencies, builds the locked data, runs the task with `python -m pa4tN.run_all`, and
zips `task_N/results/`. Jobs are resumable; the runner stops launching jobs after `DEADLINE_HOURS` (default 11 h) so a session never hits Kaggle's 12 h limit.

## Run a task anywhere
```
pip install -r requirements.txt
git clone https://github.com/AbDu11aHHH/atml-pa4-datasets ../atml-pa4-datasets      # then, inside it: python scripts/setup_data.py --public [--prefix-banks]
export PYTHONPATH=$PWD/common:$PWD/task_N/src          # (the notebooks and `run_all` set this for you; pytest uses each task's conftest.py)
python -m pytest task_N/tests -q                        # CPU unit tests
cd task_N && python -m pa4tN.run_all --smoke --gpus ""  # plumbing check on tiny models (numbers are meaningless)
cd task_N && python -m pa4tN.run_all --gpus 0,1         # the full task (GPU); --list shows the jobs, --only/--force/--deadline-hours control them
```

## Attribution
Algorithms follow Sun et al. 2024 (Wanda), Lin et al. 2024 (AWQ), Yin et al. 2024 (OWL), Dong et al. 2020 (HAWQ-V2), Molchanov et al. 2019;
HellaSwag preprocessing/scoring conventions follow EleutherAI lm-evaluation-harness (`hellaswag` task v1.0), re-implemented here without copying code.
Code was written with an LLM coding assistant (Claude Code); the PDF report must be written by the student without generative AI.
