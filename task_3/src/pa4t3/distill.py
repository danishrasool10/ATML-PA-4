"""Steps 3-4: off-policy and on-policy LoRA distillation of Qwen2.5-0.5B-Instruct from Qwen2.5-1.5B-Instruct.

Both methods score the *same budget* (30,000 math + 10,000 short-form teacher-scored response tokens by default), use the
same seeded prompt schedule, the same normalised forward token-KL

    L = 1/|H| sum_h KL( p_T(. | h) || q_phi(. | h) )     (full vocabulary, h = a response position, FP32)

and one AdamW update per >= 256 teacher-scored tokens (whole responses are accumulated; the final update may be shorter; the
last response is truncated mid-way when the category quota is hit). Gradients of the *summed* KL of each response are
accumulated with the dynamic loss scale and divided by the number of supervised tokens of the update before unscaling and
clipping, so each update minimises the token-mean KL exactly.

  off-policy : teacher responses are sampled once up front (frozen pool, temperature 1, top_p 1) and replayed,
  on-policy  : each response is sampled from the *current* student (with its adapter) and then scored by the teacher.

Time accounting (CUDA synchronised): teacher_generation, teacher_scoring, student_rollout, backward_update (student
forward + backward + optimizer), adapter_merge.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from pa4common.results import write_json
from pa4common.seed import DEFAULT_SEED, set_seed
from pa4common.timing import Stopwatch
from .data import Item, QuotaSchedule, build_chat_ids
from .lm import CachedLM
from .lora import inject_lora, lora_parameters, lora_state, merge_into
from .specdec import autoregressive_generate

QUOTAS = {"math": 30_000, "short": 10_000}
CAPS = {"math": 192, "short": 32}
UPDATE_TOKENS = 256


@dataclass
class DistillConfig:
    seed: int = DEFAULT_SEED
    quotas: dict = field(default_factory=lambda: dict(QUOTAS))
    caps: dict = field(default_factory=lambda: dict(CAPS))
    update_tokens: int = UPDATE_TOKENS
    lr: float = 3e-4
    weight_decay: float = 0.01
    clip: float = 1.0
    r: int = 8
    alpha: float = 16.0
    temperature: float = 1.0
    amp: bool = True


@torch.no_grad()
def score_rows(lm: CachedLM, ids: list[int], n: int) -> torch.Tensor:
    """Logits (n, V) of the n positions that predict the last n tokens of ``ids`` + [one more]: feed ids[:-1]."""
    t = torch.tensor([ids[:-1]], device=lm.device)
    pos = torch.arange(len(ids) - 1, device=lm.device)[None]
    return lm.engine.forward(t, pos, None, update=False, logits_last=n)[0].float()


def forward_kl_sum(t_logits: torch.Tensor, s_logits: torch.Tensor) -> torch.Tensor:
    lp = torch.log_softmax(t_logits.float(), -1)
    lq = torch.log_softmax(s_logits.float(), -1)
    return (lp.exp() * (lp - lq)).sum()


def student_logits_train(student, ids: list[int], n: int, amp: bool, device) -> torch.Tensor:
    t = torch.tensor([ids[:-1]], device=device)
    ctx = torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=amp and torch.device(device).type == "cuda")
    with ctx:
        out = student(input_ids=t, use_cache=False, logits_to_keep=n).logits[0]
    return out


def build_off_policy_pool(teacher: CachedLM, tok, schedule: QuotaSchedule, cfg: DistillConfig, eos_ids, gen, device):
    pool, t_gen = [], 0.0
    while not schedule.exhausted():
        item, cap, left = schedule.next()
        prompt = build_chat_ids(tok, item.prompt)
        with Stopwatch(device) as w:
            resp, _ = autoregressive_generate(teacher, prompt, cap, gen, eos_ids, cfg.temperature, device, timed=False)
        t_gen += w.elapsed_s
        resp = resp[:left]
        schedule.consume(item.domain, len(resp))
        pool.append((item, prompt, resp))
    return pool, t_gen


def distill(mode: str, student, teacher: CachedLM, tok, math_items: list[Item], short_items: list[Item], cfg: DistillConfig, out_dir: Path,
            eos_ids, device, log=print) -> dict:
    """``student``: HF Qwen2 (no adapters yet); modified in place (adapters injected, trained). Returns the summary dict."""
    assert mode in ("off_policy", "on_policy")
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    inject_lora(student, cfg.r, cfg.alpha)
    params = lora_parameters(student)
    assert all(p.dtype == torch.float32 and p.requires_grad for p in params)
    assert not any(p.requires_grad for n, p in student.named_parameters() if "lora_" not in n)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    cuda = torch.device(device).type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and cuda)
    student.eval()                                                  # dropout is 0; eval keeps HF caches off
    gen = torch.Generator(device=device).manual_seed(cfg.seed + (0 if mode == "off_policy" else 1))
    sched = QuotaSchedule(math_items, short_items, cfg.seed, cfg.quotas, cfg.caps)
    student_lm = CachedLM(student, "student")
    T = {"teacher_generation_s": 0.0, "teacher_scoring_s": 0.0, "student_rollout_s": 0.0, "backward_update_s": 0.0, "adapter_merge_s": 0.0}
    t_wall = time.time()
    pool = None
    if mode == "off_policy":
        pool, T["teacher_generation_s"] = build_off_policy_pool(teacher, tok, sched, cfg, eos_ids, gen, device)
        log(f"[off_policy seed={cfg.seed}] teacher pool: {len(pool)} responses, {sum(len(r) for _, _, r in pool)} tokens, {T['teacher_generation_s']:.1f}s")
        stream = iter(pool)
    n_updates = n_skipped = n_tokens = n_resp = acc = 0
    acc_kl = 0.0
    updates_log, seq_log = [], []
    per_domain = {"math": 0, "short": 0}

    def do_update():
        nonlocal acc, acc_kl, n_updates, n_skipped
        with Stopwatch(device) as w:
            for p in params:
                if p.grad is not None:
                    p.grad.div_(acc)
            if scaler.is_enabled():
                scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(params, cfg.clip)
            finite = bool(torch.isfinite(gn))
            if scaler.is_enabled():
                scale0 = scaler.get_scale()
                scaler.step(opt)                                  # skips the step itself when inf/NaN grads were found
                scaler.update()
                skipped = scaler.get_scale() < scale0 or not finite
            else:
                if finite:
                    opt.step()
                skipped = not finite
            opt.zero_grad(set_to_none=True)
        T["backward_update_s"] += w.elapsed_s
        n_updates += 1
        n_skipped += int(skipped)
        updates_log.append({"update": n_updates, "tokens": acc, "kl_per_token": acc_kl / acc, "grad_norm": float(gn),
                            "skipped": bool(skipped), "loss_scale": scaler.get_scale() if scaler.is_enabled() else 1.0})
        acc, acc_kl = 0, 0.0

    while True:
        if mode == "off_policy":
            try:
                item, prompt, resp = next(stream)
            except StopIteration:
                break
        else:
            if sched.exhausted():
                break
            item, cap, left = sched.next()
            prompt = build_chat_ids(tok, item.prompt)
            with Stopwatch(device) as w:
                resp, _ = autoregressive_generate(student_lm, prompt, cap, gen, eos_ids, cfg.temperature, device, timed=False)
            T["student_rollout_s"] += w.elapsed_s
            resp = resp[:left]
            sched.consume(item.domain, len(resp))
        n = len(resp)
        if n == 0:
            continue
        ids = prompt + resp
        with Stopwatch(device) as w:
            tl = score_rows(teacher, ids, n)
        T["teacher_scoring_s"] += w.elapsed_s
        with Stopwatch(device) as w:
            sl = student_logits_train(student, ids, n, cfg.amp, device)
            kl = forward_kl_sum(tl, sl)
            (scaler.scale(kl) if scaler.is_enabled() else kl).backward()
        T["backward_update_s"] += w.elapsed_s
        acc += n
        acc_kl += float(kl.detach())
        n_tokens += n
        n_resp += 1
        per_domain[item.domain] += n
        seq_log.append({"id": item.id, "domain": item.domain, "n_resp": n, "kl_per_token": float(kl.detach()) / n})
        if acc >= cfg.update_tokens:
            do_update()
    if acc > 0:
        do_update()                                                    # final short update
    wall = time.time() - t_wall
    state = lora_state(student)
    torch.save(state, out_dir / "adapter.pt")
    with open(out_dir / "updates.jsonl", "w") as f:
        for r in updates_log:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "sequences.jsonl", "w") as f:
        for r in seq_log:
            f.write(json.dumps(r) + "\n")
    summary = {"mode": mode, "seed": cfg.seed, "config": cfg.__dict__, "scored_tokens": n_tokens, "scored_tokens_by_domain": per_domain,
               "responses": n_resp, "updates": n_updates, "skipped_or_overflow_updates": n_skipped, "times_s": T, "wall_s": wall,
               "trainable_params": sum(p.numel() for p in params), "first_updates_kl": [u["kl_per_token"] for u in updates_log[:5]],
               "last_updates_kl": [u["kl_per_token"] for u in updates_log[-5:]]}
    write_json(out_dir / "summary.json", summary)
    return summary


def merge_adapter(fresh_backbone, state, cfg: DistillConfig, device) -> tuple[torch.nn.Module, float]:
    with Stopwatch(device) as w:
        merge_into(fresh_backbone, state, cfg.r, cfg.alpha)
    return fresh_backbone, w.elapsed_s
