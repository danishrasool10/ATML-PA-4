"""Command line entry point:  python -m pa4t1.cli --config configs/task1.json --stage all

Stages: baseline, p1, p2, structured, sensitivity, select_p3, p3, q1, q2, qtables, select_q3, q3, all, analyze.
``--smoke`` runs the whole pipeline on a tiny randomly initialised Qwen2 with synthetic manifests (CPU,
offline); its outputs are for plumbing checks only and must never be reported.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from pa4common.results import read_json
from pa4common.tiny import ByteTokenizer, make_synthetic_manifests, make_tiny_model
from .constants import BITS_UNIFORM, SPARSITIES
from .runner import Task1Runner, default_config

STAGES = ["baseline", "p1", "p2", "structured", "sensitivity", "select_p3", "p3", "q1", "q2", "qtables", "select_q3", "q3", "all", "analyze"]


def build_smoke(out_dir: str | None):
    tmp = Path(out_dir or tempfile.mkdtemp(prefix="pa4_smoke_")).resolve()
    paths = make_synthetic_manifests(tmp / "manifests", seq_len=64)
    cfg = default_config()
    cfg.update(device="cpu", dtype="float32", window_len=64, data_root=None, results_dir=str(tmp / "results"), calib_batch_size=4, eval_batch_size=4,
               hs_batch_size=8, manifests=paths,
               expected={"calibration": None, "development": None, "hellaswag": None, "ppl_scored_tokens": None},
               profile={"warmup": 1, "reps": 2, "prefill_lengths": [64, 128], "decode_prompt_len": 64, "decode_new_tokens": 4},
               bootstrap={"n_boot": 200})
    return cfg, make_tiny_model(), ByteTokenizer()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="JSON overriding defaults (see configs/task1.json)")
    ap.add_argument("--stage", choices=STAGES, default="all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-dir", default=None)
    args = ap.parse_args(argv)

    if args.smoke:
        cfg, model, tok = build_smoke(args.smoke_dir)
        runner = Task1Runner(cfg, model, tok, force=args.force)
    else:
        cfg = default_config()
        if args.config:
            cfg.update(read_json(args.config))
        runner = None

    if args.stage == "analyze":
        from .analysis import analyze
        analyze(cfg)
        return
    if runner is None:
        runner = Task1Runner(cfg, force=args.force)
    st = args.stage
    if st in ("baseline", "all"):
        runner.run_baseline()
    if st == "p1":
        [runner.run_p1(s) for s in SPARSITIES]
    if st == "p2":
        [runner.run_p2(s) for s in SPARSITIES]
    if st == "structured":
        runner.run_structured_24()
    if st == "sensitivity":
        runner.run_sensitivity()
    if st == "select_p3":
        runner.select_p3()
    if st == "p3":
        [runner.run_p3(s) for s in (0.5, 0.75)]
    if st == "q1":
        [runner.run_q1(b) for b in BITS_UNIFORM]
    if st == "q2":
        [runner.run_q2(b) for b in BITS_UNIFORM]
    if st == "qtables":
        runner.run_qtables()
    if st == "select_q3":
        runner.select_q3()
    if st == "q3":
        [runner.run_q3(bn) for bn in ("int4", "int3")]
    if st == "all":
        runner.run_all()
        from .analysis import analyze
        analyze(cfg)


if __name__ == "__main__":
    main()
