"""Orchestration of all Task 1 conditions (stage functions are idempotent: existing results are skipped
unless ``force``). Every condition starts from the pristine FP16 snapshot and uses identical
evaluation code and data."""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..common.env_meta import collect_environment
from ..common.results import read_json, sha256_file, sha256_obj, write_json, write_jsonl
from ..common.seed import set_seed
from ..common.timing import Stopwatch
from ..data.manifests import load_hellaswag, load_ppl_windows, load_token_sequences
from . import allocation as alloc
from .awq import compress_quant_model
from .calibration import make_batches
from .constants import (AWQ_GRID, BITS_ALLOWED, BITS_UNIFORM, GROUP_SIZE, HELLASWAG_SUBSET_N, LEVELS, SPARSITIES, unit_id)
from .evaluation import eval_hellaswag, eval_windows, subset_metrics
from .model_utils import ModelSnapshot, count_nonzero_eligible, enumerate_report, enumerate_units, fingerprint, load_model
from .profiling import profile_model
from .pruning import (k_from_sparsity, nm_plan, plan_zero_count, prune_model, rowwise_prune_mask, score_wanda, uniform_plan)
from .quant import quant_packed_bytes
from .sensitivity import collect_pruning_tables, fisher_pass, loss_table
from . import storage

HS_FULL = {"baseline", "p1_s50", "p2_s50", "p3_s50", "q1_int4", "q2_int4", "q3_int4", "q1_int3", "q2_int3", "q3_int3"}


def default_config() -> dict:
    return {
        "model": "Qwen/Qwen2.5-1.5B", "revision": None, "dtype": "float16", "attn_implementation": "sdpa", "seed": 6304,
        "window_len": 512, "group_size": GROUP_SIZE, "device": "cuda",
        "data_root": "../atml-pa4-datasets",   # checkout of the course data repo after `python scripts/setup_data.py --public`
        "manifests": {"calibration": "data/public_locked/c4_calibration_locked.jsonl.gz",
                      "development": "data/public_locked/c4_development_locked.jsonl.gz",
                      "wikitext2_test": "data/public_locked/wikitext2_test_windows_locked.jsonl.gz",
                      "c4_validation": "data/public_locked/c4_validation_windows_locked.jsonl.gz",
                      "hellaswag": "data/public_locked/hellaswag_val_locked.jsonl.gz"},
        "expected": {"calibration": 128, "development": 32, "hellaswag": 2000, "ppl_scored_tokens": 50000},
        "eval_batch_size": 4, "hs_batch_size": 8, "calib_batch_size": 8,
        "profile": {"warmup": 2, "reps": 5, "prefill_lengths": [512, 2048], "decode_prompt_len": 512, "decode_new_tokens": 64},
        "awq_grid": list(AWQ_GRID), "bootstrap": {"n_boot": 10000},
        "verify_packing": True, "verify_invariance": True, "hs_full_for_all": False,
        "fisher_float32": True,
        "results_dir": "results/task1",
        "selection_registry": "configs/task1_selection_registry.json",
    }


class Task1Runner:
    def __init__(self, cfg: dict, model=None, tokenizer=None, force: bool = False, log=print):
        self.cfg = cfg
        self.force = force
        self.log = log
        self.out = Path(cfg["results_dir"])
        self.out.mkdir(parents=True, exist_ok=True)
        set_seed(cfg["seed"])
        self.device = cfg["device"]
        if model is None:
            dt = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[cfg["dtype"]]
            model, tokenizer = load_model(cfg["model"], cfg.get("revision"), dt, self.device, cfg["attn_implementation"])
        self.model, self.tok = model, tokenizer
        self.units = enumerate_units(self.model)
        self.report = enumerate_report(self.model)
        self.snap = ModelSnapshot(self.model)
        self._load_data()
        self.env = collect_environment(self.model, self.tok, extra={"config": cfg, "seed": cfg["seed"],
                                                                    "manifest_sha256": self.manifest_hashes})
        write_json(self.out / "environment.json", self.env)
        write_json(self.out / "enumeration.json", self.report)
        self.registry = read_json(cfg["selection_registry"])

    # ------------------------------------------------------------------ data
    def _load_data(self):
        c, ex = self.cfg, self.cfg["expected"]
        root = Path(c.get("data_root") or ".")
        m = {k: (v if Path(v).is_absolute() else str(root / v)) for k, v in c["manifests"].items()}
        W = c["window_len"]
        self.calib = load_token_sequences(m["calibration"], self.tok, name="calibration", expected_n=ex.get("calibration"), seq_len=W)
        self.dev = load_token_sequences(m["development"], self.tok, name="development", expected_n=ex.get("development"), seq_len=W)
        self.wiki = load_ppl_windows(m["wikitext2_test"], self.tok, name="wikitext2_test", window_len=W, expected_scored=ex.get("ppl_scored_tokens"))
        self.c4val = load_ppl_windows(m["c4_validation"], self.tok, name="c4_validation", window_len=W, expected_scored=ex.get("ppl_scored_tokens"))
        self.hs = load_hellaswag(m["hellaswag"], expected_n=ex.get("hellaswag"))
        self.manifest_hashes = {k: sha256_file(v) for k, v in m.items()}
        assert not (set(map(tuple, self.calib.ids)) & set(map(tuple, self.dev.ids))), "calibration and development sequences overlap"
        self.batches = make_batches(self.calib.ids, c["calib_batch_size"])
        self.log(f"[data] calib={len(self.calib)} dev={len(self.dev)} wiki N={self.wiki.n_scored_tokens} c4 N={self.c4val.n_scored_tokens} hs={len(self.hs)}")

    # ------------------------------------------------------------------ helpers
    def cond_dir(self, cid: str) -> Path:
        return self.out / "conditions" / cid

    def done(self, cid: str) -> bool:
        return (self.cond_dir(cid) / "result.json").exists() and not self.force

    def restore(self):
        self.snap.restore(self.model)

    def nll_on(self, seqs: list[list[int]]) -> dict:
        r = eval_windows(self.model, seqs, self.cfg["eval_batch_size"])
        return {"N": r["N"], "mean_nll": r["mean_nll"], "ppl": r["ppl"]}

    def evaluate(self, cid: str, family: str, params: dict, compress: dict, hs_mode: str | None = None) -> dict:
        """Evaluate the (already compressed) model and write result files for condition ``cid``."""
        d = self.cond_dir(cid)
        d.mkdir(parents=True, exist_ok=True)
        hs_mode = hs_mode or ("full" if (cid in HS_FULL or self.cfg["hs_full_for_all"]) else "subset")
        res = {"condition": cid, "family": family, "params": params, "hellaswag_mode": hs_mode,
               "model_fingerprint": fingerprint(self.model), "compression": compress.get("summary", {}),
               "timings_s": dict(compress.get("timings_s", {}))}
        ev = {}
        arrays = {}
        for key, ws in (("wikitext2_test", self.wiki), ("c4_validation", self.c4val)):
            with Stopwatch(self.device) as sw:
                r = eval_windows(self.model, ws.windows, self.cfg["eval_batch_size"])
            ev[key] = {k: r[k] for k in ("N", "n_windows", "mean_nll", "ppl")}
            ev[key]["eval_s"] = sw.elapsed_s
            arrays[f"{key}_nll_sum"] = r["window_nll_sum"]
            arrays[f"{key}_count"] = r["window_token_count"]
        ev["calibration_nll"] = self.nll_on(self.calib.ids)
        ev["development_nll"] = self.nll_on(self.dev.ids)
        n_hs = len(self.hs) if hs_mode == "full" else HELLASWAG_SUBSET_N
        n_hs = min(n_hs, len(self.hs))
        with Stopwatch(self.device) as sw:
            hs = eval_hellaswag(self.model, self.tok, self.hs.examples[:n_hs], self.hs.ids[:n_hs], self.cfg["hs_batch_size"])
        ev["hellaswag"] = {"n": hs["n"], "acc": hs["acc"], "acc_norm": hs["acc_norm"], "first512": subset_metrics(hs, min(HELLASWAG_SUBSET_N, hs["n"])),
                           "task": hs["task"], "prompt_format": hs["prompt_format"], "normalization": hs["normalization"], "eval_s": sw.elapsed_s}
        arrays["hs_correct"] = hs["correct"]
        arrays["hs_correct_norm"] = hs["correct_norm"]
        write_jsonl(d / "hellaswag_scores.jsonl", hs["per_example"])
        np.savez(d / "per_window_and_example.npz", **arrays)
        res["quality"] = ev
        res["storage"] = {"nonzeros": storage.nonzero_counts(self.model, self.units), "dense_fp16": storage.dense_report(self.model, self.units),
                          "sparse_16bit_model": storage.sparse_report(self.model, self.units), **compress.get("storage", {})}
        res["profile"] = profile_model(self.model, **self.cfg["profile"])
        write_json(d / "result.json", res)
        self.log(f"[{cid}] wiki ppl {ev['wikitext2_test']['ppl']:.3f} c4 ppl {ev['c4_validation']['ppl']:.3f} "
                 f"hs acc {ev['hellaswag']['acc']:.4f}/{ev['hellaswag']['acc_norm']:.4f} (n={hs['n']})")
        return res

    # ------------------------------------------------------------------ baseline
    def run_baseline(self):
        cid = "baseline"
        if self.done(cid):
            return
        self.restore()
        self.evaluate(cid, "baseline", {}, {"summary": {"description": "unchanged FP16 checkpoint"},
                                           "storage": {"sparse_format_note": "baseline is dense"}})

    # ------------------------------------------------------------------ pruning conditions
    def _prune_condition(self, cid, family, params, plan, diagnostics=False, extra_summary=None):
        if self.done(cid):
            return
        self.restore()
        n_elig = sum(u.numel for u in self.units)
        target = plan_zero_count(self.units, plan)
        out = prune_model(self.model, self.units, self.batches, plan, diagnostics=diagnostics)
        recs = out["records"]
        zeroed = sum(r["n_pruned_by_mask"] for r in recs.values())
        assert zeroed == target, f"mask zeros {zeroed} != planned {target}"
        nnz = count_nonzero_eligible(self.units)
        assert nnz <= n_elig - target
        summary = {"eligible_weights": n_elig, "zeroed_by_mask": zeroed, "global_sparsity_by_mask": zeroed / n_elig,
                   "eligible_nnz": nnz, "achieved_nonzero_fraction": nnz / n_elig,
                   "mean_rel_err": float(np.mean([r["rel_err"] for r in recs.values()])),
                   "energy_weighted_rel_err": sum(r["abs_err"] for r in recs.values()) / sum(r["ref_energy"] for r in recs.values()),
                   **(extra_summary or {})}
        d = self.cond_dir(cid)
        d.mkdir(parents=True, exist_ok=True)
        write_json(d / "per_unit_records.json", recs)
        timings = {"calibration_and_compression_s": out["info"]["wall_s"]}
        self.evaluate(cid, family, params, {"summary": summary, "timings_s": timings})

    def run_p1(self, s):
        self._prune_condition(f"p1_s{int(s*100)}", "P1_magnitude", {"sparsity": s}, uniform_plan(self.units, s, "magnitude"))

    def run_p2(self, s):
        self._prune_condition(f"p2_s{int(s*100)}", "P2_wanda", {"sparsity": s}, uniform_plan(self.units, s, "wanda"), diagnostics=True)

    def run_structured_24(self):
        # the unstructured 50% Wanda condition (p2_s50) is the matched comparison
        self._prune_condition("p2_24", "S2_wanda_2to4", {"n_keep": 2, "m": 4, "sparsity": 0.5}, nm_plan(self.units))

    # ------------------------------------------------------------------ sensitivity
    def run_sensitivity(self):
        sd = self.out / "sensitivity"
        sd.mkdir(parents=True, exist_ok=True)
        f_tab, f_fis = sd / "pruning_tables.pt", sd / "fisher.pt"
        if f_tab.exists() and f_fis.exists() and not self.force:
            return
        self.restore()
        t = collect_pruning_tables(self.model, self.units, self.batches)
        torch.save(t, f_tab)
        self.log(f"[sensitivity] pruning tables {t['info']['wall_s']:.1f}s")
        if self.cfg["fisher_float32"]:
            self.model.float()
        fis = fisher_pass(self.model, self.units, self.calib.ids, t)
        if self.cfg["fisher_float32"]:
            self.model.to(getattr(torch, self.cfg["dtype"]))  # exact: values were FP16 before the upcast
        self.restore()
        torch.save(fis, f_fis)
        self.log(f"[sensitivity] fisher {fis['wall_s']:.1f}s over {fis['n_sequences']} sequences")
        summary = {"uids": t["uids"], "owl_ratio": t["owl_ratio"], "local_rel": t["local_rel"], "levels": t["levels"],
                   "fisher_trace": fis["trace"], "taylor": fis["taylor"],
                   "timings_s": {"pruning_tables": t["info"]["wall_s"], "fisher": fis["wall_s"]}}
        write_json(sd / "sensitivity_summary.json", summary)

    def load_sensitivity(self):
        sd = self.out / "sensitivity"
        return torch.load(sd / "pruning_tables.pt", weights_only=False), torch.load(sd / "fisher.pt", weights_only=False)

    # ------------------------------------------------------------------ P3
    def _p3_allocation(self, cand: dict, s: float, tables, fisher):
        if cand["rule"] == "power_law":
            return alloc.power_law_allocation(self.units, tables["owl_ratio"], s, cand["gamma"], meta={"design": cand["design"], "name": cand["name"]})
        lt = loss_table(tables, fisher, cand["design"])
        return alloc.mckp_sparsity_allocation(self.units, lt, s, meta={"design": cand["design"], "name": cand["name"]})

    def _direct_wanda_prune(self, k_by_uid: dict[str, int], norms: dict):
        """Apply Wanda row-wise pruning using stored dense activation norms (no calibration pass);
        identical result to ``prune_model`` -- used for the cheap dev-set selection loop."""
        for u in self.units:
            W = u.module.weight.data
            mask = rowwise_prune_mask(score_wanda(W, norms[u.uid].to(W.device)), k_by_uid[u.uid])
            W.masked_fill_(mask, 0)

    def select_p3(self):
        """Choose the P3 design using ONLY the 32-sequence development manifest (never test data)."""
        f = self.out / "selection" / "p3_selection.json"
        if f.exists() and not self.force:
            return read_json(f)
        reg = self.registry["p3"]
        tables, fisher = self.load_sensitivity()
        norms = tables["act_norm"]
        rows, allocs = [], {}
        t0 = time.perf_counter()
        # reference: uniform Wanda on the dev set (not selectable; context for the table)
        for s in reg["targets"]:
            self.restore()
            self._direct_wanda_prune({u.uid: k_from_sparsity(s, u.shape[1]) for u in self.units}, norms)
            rows.append({"name": "uniform_wanda", "target": s, "dev_nll": self.nll_on(self.dev.ids)["mean_nll"], "selectable": False})
        for cand in reg["candidates"]:
            for s in reg["targets"]:
                a = self._p3_allocation(cand, s, tables, fisher)
                allocs[f"{cand['name']}@{s}"] = {"k": a.k, "s": a.s, "target_zeros": a.target_zeros, "achieved_zeros": a.achieved_zeros,
                                                  "residual": a.residual, "meta": a.meta}
                self.restore()
                self._direct_wanda_prune(a.k, norms)
                nll = self.nll_on(self.dev.ids)["mean_nll"]
                rows.append({"name": cand["name"], "target": s, "dev_nll": nll, "residual_zeros": a.residual,
                             "min_s": min(a.s.values()), "max_s": max(a.s.values()), "selectable": True})
                self.log(f"[p3-select] {cand['name']}@{s}: dev NLL {nll:.4f} (residual {a.residual})")
        self.restore()
        crit = reg["criterion"]
        assert crit == "mean_dev_nll_over_targets"
        score = {c["name"]: float(np.mean([r["dev_nll"] for r in rows if r["name"] == c["name"]])) for c in reg["candidates"]}
        chosen = min(score, key=score.get)
        out = {"criterion": crit, "registry_sha256": sha256_obj(reg), "scores": score, "chosen": chosen, "rows": rows,
               "locked_at": time.time(), "selection_wall_s": time.perf_counter() - t0, "uses_test_data": False,
               "n_candidate_evaluations": len(reg["candidates"]) * len(reg["targets"])}
        write_json(f.parent / "p3_allocations_all_candidates.json", allocs)
        write_json(f, out)
        return out

    def run_p3(self, s):
        sel_f = self.out / "selection" / "p3_selection.json"
        assert sel_f.exists(), "run select_p3 (dev manifest only) before any P3 test evaluation"
        sel = read_json(sel_f)
        assert sel["registry_sha256"] == sha256_obj(self.registry["p3"]), "selection registry changed after the lock"
        tables, fisher = self.load_sensitivity()
        cand = next(c for c in self.registry["p3"]["candidates"] if c["name"] == sel["chosen"])
        a = self._p3_allocation(cand, s, tables, fisher)
        assert a.residual == 0, f"P3 allocation does not match the P1/P2 budget exactly (residual {a.residual} weights)"
        assert a.achieved_zeros == int(round(s * sum(u.numel for u in self.units)))
        plan = {u.uid: {"rule": "wanda", "k": a.k[u.uid]} for u in self.units}
        write_json(self.out / "selection" / f"p3_allocation_s{int(s*100)}.json", {"k": a.k, "s": a.s, "meta": a.meta, "design": cand})
        self._prune_condition(f"p3_s{int(s*100)}", "P3_nonuniform_wanda", {"sparsity": s, "design": cand["name"]}, plan,
                              extra_summary={"min_unit_sparsity": min(a.s.values()), "max_unit_sparsity": max(a.s.values())})

    # ------------------------------------------------------------------ quantization
    def _quant_condition(self, cid, family, params, bits_by_unit, mode):
        if self.done(cid):
            return
        self.restore()
        out = compress_quant_model(self.model, self.units, self.batches, bits_by_unit, mode=mode, group_size=self.cfg["group_size"],
                                   grid=tuple(self.cfg["awq_grid"]), verify_invariance=self.cfg["verify_invariance"],
                                   verify_packing=self.cfg["verify_packing"])
        recs = out["records"]
        qrep = storage.quant_report(self.model, self.units, bits_by_unit, self.cfg["group_size"],
                                    compensation_bytes=out["info"]["compensation_scale_bytes"], detail=True)
        summary = {"mode": mode, "mean_rel_err": float(np.mean([r["rel_err"] for r in recs.values()])),
                   "energy_weighted_rel_err": sum(r["abs_err"] for r in recs.values()) / sum(r["ref_energy"] for r in recs.values()),
                   "packed_total_bytes": qrep["packed_total_bytes"], "mean_integer_bits": qrep["mean_integer_bits"],
                   "max_invariance_rel_err": out["info"]["max_invariance_rel_err"], "packing_verified_matrices": out["info"]["packing_verified"],
                   "execution_note": "weights de-quantized to FP16 (fake quantization); evaluated with dense FP16 kernels"}
        d = self.cond_dir(cid)
        d.mkdir(parents=True, exist_ok=True)
        write_json(d / "per_unit_records.json", recs)
        write_json(d / "awq_group_search.json", out["groups"])
        write_json(d / "bit_assignment.json", bits_by_unit)
        self.evaluate(cid, family, params, {"summary": summary, "storage": {"quantized_model": qrep},
                                            "timings_s": {"calibration_and_compression_s": out["info"]["wall_s"]}})

    def uniform_bits(self, b):
        return {u.uid: b for u in self.units}

    def run_q1(self, b):
        self._quant_condition(f"q1_int{b}", "Q1_ptq", {"bits": b}, self.uniform_bits(b), "q1")

    def run_q2(self, b):
        self._quant_condition(f"q2_int{b}", "Q2_awq", {"bits": b}, self.uniform_bits(b), "awq")

    def total_bytes_uniform(self, b) -> int:
        return storage.quant_report(self.model, self.units, self.uniform_bits(b), self.cfg["group_size"])["packed_total_bytes"]

    def run_qtables(self):
        f = self.out / "sensitivity" / "q_tables.json"
        if f.exists() and not self.force:
            return
        tab, wall = {}, 0.0
        for b in BITS_ALLOWED:
            self.restore()
            out = compress_quant_model(self.model, self.units, self.batches, self.uniform_bits(b), mode="awq", group_size=self.cfg["group_size"],
                                       grid=tuple(self.cfg["awq_grid"]), apply=False, verify_invariance=False)
            tab[str(b)] = {uid: {k: r[k] for k in ("rel_err", "abs_err", "weight_mse", "packed_bytes")} for uid, r in out["records"].items()}
            wall += out["info"]["wall_s"]
            self.log(f"[qtables] b={b} {out['info']['wall_s']:.1f}s")
        self.restore()
        write_json(f, {"tables": tab, "bits": list(BITS_ALLOWED), "wall_s": wall})

    def _q3_assignment(self, cand: dict, budget_name: str, tables, fisher):
        b_budget = {"int4": 4, "int3": 3}[budget_name]
        total_budget = self.total_bytes_uniform(b_budget)
        other = storage.quant_report(self.model, self.units, self.uniform_bits(4), self.cfg["group_size"])["other_fp16_bytes"]
        budget_elig = total_budget - other  # compensation scale bytes are 0 (all inverse scales are folded)
        bits = list(BITS_ALLOWED)
        n = len(self.units)
        byte_t = np.array([[quant_packed_bytes(u.shape, b, self.cfg["group_size"])["total"] for b in bits] for u in self.units], dtype=np.float64)
        q = tables["tables"]
        if cand["design"] == "local_rel":
            loss = np.array([[q[str(b)][u.uid]["rel_err"] for b in bits] for u in self.units])
        elif cand["design"] == "local_abs":
            loss = np.array([[q[str(b)][u.uid]["abs_err"] for b in bits] for u in self.units])
        elif cand["design"] == "hawq":
            # HAWQ-V2: Omega = Tr(F_u)/n_u * ||W - W_hat||_F^2 = Tr(F_u) * mse   (n_u cancels)
            tr = fisher["trace"]
            loss = np.array([[tr[i] * q[str(b)][u.uid]["weight_mse"] for b in bits] for i, u in enumerate(self.units)])
        else:
            raise KeyError(cand["design"])
        chosen = alloc.mckp_bit_allocation(self.units, tuple(bits), loss, byte_t, budget_elig)
        by_uid = {u.uid: chosen[i] for i, u in enumerate(self.units)}
        rep = storage.quant_report(self.model, self.units, by_uid, self.cfg["group_size"])
        assert rep["packed_total_bytes"] <= total_budget, (rep["packed_total_bytes"], total_budget)
        return by_uid, {"budget_total_bytes": total_budget, "packed_total_bytes": rep["packed_total_bytes"],
                        "slack_bytes": total_budget - rep["packed_total_bytes"], "mean_integer_bits": rep["mean_integer_bits"],
                        "bit_histogram": {str(b): int(sum(1 for v in by_uid.values() if v == b)) for b in bits}}

    def select_q3(self):
        f = self.out / "selection" / "q3_selection.json"
        if f.exists() and not self.force:
            return read_json(f)
        reg = self.registry["q3"]
        tables = read_json(self.out / "sensitivity" / "q_tables.json")
        _, fisher = self.load_sensitivity()
        rows, allocs = [], {}
        t0 = time.perf_counter()
        for cand in reg["candidates"]:
            for bn in reg["budgets"]:
                by_uid, info = self._q3_assignment(cand, bn, tables, fisher)
                allocs[f"{cand['name']}@{bn}"] = {"bits": by_uid, **info}
                self.restore()
                compress_quant_model(self.model, self.units, self.batches, by_uid, mode="awq", group_size=self.cfg["group_size"],
                                     grid=tuple(self.cfg["awq_grid"]), verify_invariance=False)
                nll = self.nll_on(self.dev.ids)["mean_nll"]
                rows.append({"name": cand["name"], "budget": bn, "dev_nll": nll, **info})
                self.log(f"[q3-select] {cand['name']}@{bn}: dev NLL {nll:.4f} bytes {info['packed_total_bytes']}/{info['budget_total_bytes']}")
        self.restore()
        assert reg["criterion"] == "mean_dev_nll_over_budgets"
        score = {c["name"]: float(np.mean([r["dev_nll"] for r in rows if r["name"] == c["name"]])) for c in reg["candidates"]}
        chosen = min(score, key=score.get)
        out = {"criterion": reg["criterion"], "registry_sha256": sha256_obj(reg), "scores": score, "chosen": chosen, "rows": rows,
               "locked_at": time.time(), "selection_wall_s": time.perf_counter() - t0, "uses_test_data": False}
        write_json(f.parent / "q3_allocations_all_candidates.json", allocs)
        write_json(f, out)
        return out

    def run_q3(self, budget_name: str):
        sel_f = self.out / "selection" / "q3_selection.json"
        assert sel_f.exists(), "run select_q3 (dev manifest only) before any Q3 test evaluation"
        sel = read_json(sel_f)
        assert sel["registry_sha256"] == sha256_obj(self.registry["q3"]), "selection registry changed after the lock"
        cand = next(c for c in self.registry["q3"]["candidates"] if c["name"] == sel["chosen"])
        tables = read_json(self.out / "sensitivity" / "q_tables.json")
        _, fisher = self.load_sensitivity()
        by_uid, info = self._q3_assignment(cand, budget_name, tables, fisher)
        self._quant_condition(f"q3_{budget_name}", "Q3_mixed_awq", {"budget": budget_name, "design": cand["name"], **info}, by_uid, "awq")
        # matched-budget check against the Q2 uniform result file
        r = read_json(self.cond_dir(f"q3_{budget_name}") / "result.json")
        q2 = read_json(self.cond_dir(f"q2_{budget_name}") / "result.json") if self.done(f"q2_{budget_name}") or (self.cond_dir(f"q2_{budget_name}") / "result.json").exists() else None
        if q2 is not None:
            b3 = r["storage"]["quantized_model"]["packed_total_bytes"]
            b2 = q2["storage"]["quantized_model"]["packed_total_bytes"]
            assert b3 <= b2, f"Q3 exceeds Q2 budget: {b3} > {b2}"

    # ------------------------------------------------------------------ everything
    def run_all(self):
        self.run_baseline()
        for s in SPARSITIES:
            self.run_p1(s)
            self.run_p2(s)
        self.run_structured_24()
        self.run_sensitivity()
        self.select_p3()
        for s in (0.5, 0.75):
            self.run_p3(s)
        for b in BITS_UNIFORM:
            self.run_q1(b)
            self.run_q2(b)
        self.run_qtables()
        self.select_q3()
        for bn in ("int4", "int3"):
            self.run_q3(bn)
