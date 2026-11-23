"""Exact speculative sampling (Leviathan et al. / Chen et al.) with explicit KV-cache rollback and round-level timers.

Round (``gamma`` proposals, temperature ``T``, top_p = 1):
  1. draft proposal: the draft consumes every token it has not seen yet and samples x_1..x_gamma ~ q_i (gamma passes),
  2. one target pass over [last committed token, x_1..x_gamma] gives p_0..p_gamma,
  3. accept x_i with probability min(1, p_{i-1}(x_i) / q_i(x_i)); on the first rejection sample the replacement from the
     normalised positive residual r(y) = [p(y) - q(y)]_+ / sum_z [p(z) - q(z)]_+ and stop; if all gamma are accepted sample
     a bonus token from p_gamma,
  4. cache advancement/rollback: both caches are truncated to ``len(sequence) - 1`` (the newest token is never fed until
     the next round), so a cache position always equals the absolute position of its token.

``residual_mode='p'`` is the deliberately buggy control used by the chi-square validation (resample from p, not from the
residual). Every decision is made on-device with one generator; there is exactly one host sync per round in step 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from pa4common.timing import Stopwatch


def probs_of(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return torch.softmax(logits.float() / temperature, dim=-1) if logits.dtype != torch.float64 else torch.softmax(logits / temperature, dim=-1)


def residual_distribution(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Normalised positive part of p - q (falls back to p if the positive mass is zero)."""
    r = (p - q).clamp_min(0)
    z = r.sum(-1, keepdim=True)
    return torch.where(z > 0, r / z.clamp_min(1e-300), p)


def sample(prob: torch.Tensor, gen: torch.Generator) -> int:
    return int(torch.multinomial(prob, 1, generator=gen).item())


@dataclass
class SpecStats:
    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    rejections: int = 0                   # rounds that ended with a rejection (the rest ended with a bonus token)
    draft_passes: int = 0
    target_passes: int = 0
    t_prefill_s: float = 0.0
    t_draft_s: float = 0.0                # proposal passes (incl. sampling the proposals)
    t_verify_s: float = 0.0               # single target verification pass per round
    t_accept_s: float = 0.0               # acceptance test + residual / bonus sampling
    t_cache_s: float = 0.0                # cache rollback / advancement
    t_total_s: float = 0.0
    round_lengths: list[int] = field(default_factory=list)

    @property
    def alpha_mle(self) -> float:
        """Per-token acceptance probability under the iid model (each tested proposal is a Bernoulli trial)."""
        return self.accepted / max(self.accepted + self.rejections, 1)


def speculative_generate(target, draft, prompt: list[int], gamma: int, max_new: int, gen: torch.Generator, eos_ids=frozenset(),
                         temperature: float = 1.0, residual_mode: str = "exact", device="cpu", check_cache: bool = False,
                         timed: bool = True):
    """Returns (new tokens, SpecStats). ``eos_ids``: generation stops after the first EOS (EOS included). Output is truncated
    to ``max_new`` tokens (a prefix of an exact sample is an exact sample of the prefix)."""
    assert residual_mode in ("exact", "p")
    st = SpecStats()
    sw = (lambda: Stopwatch(device)) if timed else (lambda: _NullWatch())
    seq = list(prompt)
    t_all = sw()
    t_all.__enter__()
    with sw() as w:
        target.reset()
        draft.reset()
        if len(seq) > 1:
            target.feed(seq[:-1])
            draft.feed(seq[:-1])
    st.t_prefill_s = w.elapsed_s
    n_prompt = len(seq)
    done = False
    while not done and len(seq) - n_prompt < max_new:
        st.rounds += 1
        # ---- 1. draft proposals --------------------------------------------------------------------------------
        with sw() as w:
            assert draft.tokens == seq[: len(draft)] and len(draft) <= len(seq) - 1 + 1
            lg = draft.feed(seq[len(draft):])
            st.draft_passes += 1
            qs, xs = [], []
            for i in range(gamma):
                q = probs_of(lg[-1], temperature)
                x = sample(q, gen)
                qs.append(q)
                xs.append(x)
                if i < gamma - 1:
                    lg = draft.feed([x])
                    st.draft_passes += 1
        st.t_draft_s += w.elapsed_s
        # ---- 2. target verification (one pass) -------------------------------------------------------------------
        with sw() as w:
            assert target.tokens == seq[: len(target)] and len(target) == len(seq) - 1
            lgt = target.feed([seq[-1]] + xs)
            st.target_passes += 1
        st.t_verify_s += w.elapsed_s
        # ---- 3. accept / residual / bonus ------------------------------------------------------------------------
        with sw() as w:
            P = probs_of(lgt, temperature)                           # (gamma+1, V)
            Q = torch.stack(qs)                                      # (gamma, V)
            idx = torch.tensor(xs, device=P.device)
            ar = torch.arange(gamma, device=P.device)
            p_x, q_x = P[ar, idx], Q[ar, idx]
            u = torch.rand(gamma, generator=gen, device=P.device, dtype=P.dtype)
            ok = (u * q_x <= p_x)                                    # u <= min(1, p/q), q_x > 0 as x ~ q
            n_acc = int(torch.cumprod(ok.to(torch.int64), 0).sum().item())
            if n_acc == gamma:
                y = sample(P[gamma], gen)
            else:
                st.rejections += 1
                r = residual_distribution(P[n_acc], Q[n_acc]) if residual_mode == "exact" else P[n_acc]
                y = sample(r, gen)
            new = xs[:n_acc] + [y]
        st.t_accept_s += w.elapsed_s
        st.proposed += gamma
        st.accepted += n_acc
        # ---- 4. commit + cache rollback / advancement ----------------------------------------------------------
        for j, tok in enumerate(new):
            if tok in eos_ids:
                new = new[: j + 1]
                done = True
                break
        seq.extend(new)
        st.round_lengths.append(len(new))
        with sw() as w:
            target.truncate(len(seq) - 1)
            draft.truncate(min(len(draft), len(seq) - 1))
        st.t_cache_s += w.elapsed_s
        if check_cache:
            assert target.tokens == seq[: len(target)] and draft.tokens == seq[: len(draft)]
    t_all.__exit__(None, None, None)
    st.t_total_s = t_all.elapsed_s
    out = seq[n_prompt:]
    return out[:max_new], st


def autoregressive_generate(target, prompt: list[int], max_new: int, gen: torch.Generator, eos_ids=frozenset(), temperature: float = 1.0,
                            device="cpu", ignore_eos: bool = False, timed: bool = True):
    """Target-only ancestral sampling with the same cache machinery (the timing baseline); ``temperature=0`` = greedy."""
    sw = (lambda: Stopwatch(device)) if timed else (lambda: _NullWatch())
    out = []
    t_all = sw()
    t_all.__enter__()
    with sw() as w:
        target.reset()
        lg = target.feed(prompt)
    t_prefill = w.elapsed_s
    n_steps, t_dec = 0, 0.0
    for i in range(max_new):
        with sw() as w:
            x = int(lg[-1].argmax().item()) if temperature == 0 else sample(probs_of(lg[-1], temperature), gen)
        out.append(x)
        if x in eos_ids and not ignore_eos:
            t_dec += w.elapsed_s
            break
        if i == max_new - 1:
            t_dec += w.elapsed_s
            break
        with sw() as w2:
            lg = target.feed([x])
        n_steps += 1
        t_dec += w.elapsed_s + w2.elapsed_s
    t_all.__exit__(None, None, None)
    return out, {"t_prefill_s": t_prefill, "t_decode_s": t_dec, "t_total_s": t_all.elapsed_s, "decode_passes": n_steps, "n_tokens": len(out)}


class _NullWatch:
    elapsed_s = 0.0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
