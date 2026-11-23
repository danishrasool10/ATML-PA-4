"""Constrained allocation rules shared by P3 (sparsity) and Q3 (bit-width).

Two rule families are implemented (both are fully deterministic):

``power_law_allocation``  (OWL-style, scalar sensitivity -> sparsity)
    s_u = clip(lambda * (sigma_u / geomean(sigma))^(-gamma), 0.10, 0.85); lambda is found by
    bisection so that the total number of zeroed eligible weights hits the target; then integer
    per-row counts are rounded and the residual is repaired exactly (see ``fix_exact``).

``mckp_allocate``  (equal-slope / Lagrangian greedy for the multiple-choice knapsack)
    every unit has a menu of options (sparsity level or bit width) with a resource cost and a
    sensitivity-derived loss L_u(option); minimise sum_u L_u subject to sum_u cost_u <= budget.
    Greedy on the upper convex hull of each unit's (cost, loss) points = equal marginal
    loss-per-resource across units (HAWQ-V2 / BRECQ-style constrained assignment), followed by a
    best-single-upgrade fill of the remaining budget. Never exceeds the budget.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from .constants import LEVELS, S_MAX, S_MIN
from .pruning import k_bounds


# ----------------------------------------------------------------------------- MCKP

def _hull_steps(costs: np.ndarray, losses: np.ndarray):
    """Indices of options on the lower-left convex hull, ordered by increasing cost."""
    order = np.argsort(costs, kind="stable")
    pts = [(costs[i], losses[i], i) for i in order]
    # drop dominated points (more cost, not less loss)
    nd, best = [], math.inf
    for c, l, i in pts:
        if l < best - 1e-18:
            nd.append((c, l, i))
            best = l
    hull = []
    for p in nd:
        while len(hull) >= 2:
            (c1, l1, _), (c2, l2, _) = hull[-2], hull[-1]
            # remove middle point if it lies above the line from hull[-2] to p
            if (l2 - l1) * (p[0] - c1) >= (p[1] - l1) * (c2 - c1):
                hull.pop()
            else:
                break
        hull.append(p)
    return [h[2] for h in hull]


def mckp_allocate(costs: list[np.ndarray], losses: list[np.ndarray], budget: float) -> list[int]:
    """Return the chosen option index per unit. ``costs[u][o]`` / ``losses[u][o]`` for option o."""
    n = len(costs)
    costs = [np.asarray(c, dtype=np.float64) for c in costs]
    losses = [np.asarray(l, dtype=np.float64) for l in losses]
    hulls = [_hull_steps(costs[u], losses[u]) for u in range(n)]
    pos = [0] * n  # position on hull
    choice = [hulls[u][0] for u in range(n)]
    total = sum(costs[u][choice[u]] for u in range(n))
    if total > budget + 1e-9:
        raise ValueError(f"infeasible: cheapest assignment costs {total} > budget {budget}")

    def push(u, heap):
        if pos[u] + 1 < len(hulls[u]):
            a, b = hulls[u][pos[u]], hulls[u][pos[u] + 1]
            dc = costs[u][b] - costs[u][a]
            dl = losses[u][a] - losses[u][b]
            heapq.heappush(heap, (-dl / max(dc, 1e-12), u, pos[u]))

    heap: list = []
    for u in range(n):
        push(u, heap)
    stuck = []
    while heap:
        neg, u, p = heapq.heappop(heap)
        if p != pos[u]:
            continue
        a, b = hulls[u][pos[u]], hulls[u][pos[u] + 1]
        dc = costs[u][b] - costs[u][a]
        if total + dc <= budget + 1e-9:
            total += dc
            pos[u] += 1
            choice[u] = b
            push(u, heap)
        else:
            stuck.append(u)
    # fill: best single upgrade (any option, not only hull) that still fits
    while True:
        best = None
        for u in range(n):
            cu = costs[u][choice[u]]
            lu = losses[u][choice[u]]
            for o in range(len(costs[u])):
                dc = costs[u][o] - cu
                dl = lu - losses[u][o]
                if dl > 1e-18 and total + dc <= budget + 1e-9 and (dc > 0 or dc <= 0):
                    ratio = dl / max(dc, 1e-12) if dc > 0 else math.inf
                    if best is None or ratio > best[0]:
                        best = (ratio, u, o, dc)
        if best is None:
            break
        _, u, o, dc = best
        total += dc
        choice[u] = o
    assert total <= budget + 1e-6
    return choice


# ----------------------------------------------------------------------------- exact integer repair

def fix_exact(k: np.ndarray, dout: np.ndarray, lo: np.ndarray, hi: np.ndarray, target_zeros: int,
              penalty_fn) -> tuple[np.ndarray, int]:
    """Adjust integer per-row counts ``k`` (+-1 steps) until sum(dout*k) == target_zeros if possible.

    ``penalty_fn(u, k_new)`` is the cost of unit u having count k_new (lower = preferred); at every
    step the move with the smallest penalty increase per zeroed weight is taken. Returns (k, residual).
    """
    k = k.copy()
    resid = int(target_zeros - int((dout * k).sum()))
    guard = 0
    while resid != 0 and guard < 10_000_000:
        guard += 1
        best = None
        sign = 1 if resid > 0 else -1
        for u in range(len(k)):
            step = int(dout[u])
            if step > abs(resid):
                continue
            kn = k[u] + sign
            if kn < lo[u] or kn > hi[u]:
                continue
            d = (penalty_fn(u, kn) - penalty_fn(u, k[u])) / step
            if best is None or d < best[0]:
                best = (d, u, kn)
        if best is None:
            break
        _, u, kn = best
        resid -= sign * int(dout[u])
        k[u] = kn
    return k, resid


@dataclass
class SparsityAllocation:
    k: dict[str, int]                 # zeroed weights per output row for every unit
    s: dict[str, float]               # k / d_in
    target_zeros: int
    achieved_zeros: int
    residual: int
    meta: dict

    def total_zeros(self, units) -> int:
        return sum(u.shape[0] * self.k[u.uid] for u in units)


def target_zero_count(units, s_global: float) -> int:
    n = sum(u.numel for u in units)
    return int(round(s_global * n))


# ----------------------------------------------------------------------------- P3 rules

def power_law_allocation(units, sigma: np.ndarray, s_global: float, gamma: float, meta: dict | None = None) -> SparsityAllocation:
    din = np.array([u.shape[1] for u in units])
    dout = np.array([u.shape[0] for u in units])
    n = dout * din
    lo_k = np.array([k_bounds(d)[0] for d in din])
    hi_k = np.array([k_bounds(d)[1] for d in din])
    target = target_zero_count(units, s_global)
    sig = np.clip(np.asarray(sigma, np.float64), 1e-12, None)
    sig_t = np.exp(-gamma * (np.log(sig) - np.log(sig).mean()))   # (sigma/geomean)^(-gamma)
    s_lo, s_hi = lo_k / din, hi_k / din

    def total(lam):
        s = np.clip(lam * sig_t, s_lo, s_hi)
        return float((n * s).sum())

    lam_lo, lam_hi = 1e-9, 1e9
    for _ in range(200):
        mid = math.sqrt(lam_lo * lam_hi)
        if total(mid) < target:
            lam_lo = mid
        else:
            lam_hi = mid
    s_star = np.clip(lam_hi * sig_t, s_lo, s_hi)
    k0 = np.clip(np.rint(s_star * din).astype(np.int64), lo_k, hi_k)

    def pen(u, kk):
        return ((kk / din[u]) - s_star[u]) ** 2 * n[u]

    k, resid = fix_exact(k0, dout, lo_k, hi_k, target, pen)
    meta = {**(meta or {}), "rule": "power_law", "gamma": gamma, "lambda": lam_hi}
    return SparsityAllocation({u.uid: int(k[i]) for i, u in enumerate(units)}, {u.uid: float(k[i] / din[i]) for i, u in enumerate(units)},
                              target, int((dout * k).sum()), resid, meta)


def mckp_sparsity_allocation(units, loss_tables: np.ndarray, s_global: float, levels=LEVELS, meta: dict | None = None) -> SparsityAllocation:
    """``loss_tables[u, l]`` = loss of pruning unit u at sparsity ``levels[l]`` (monotone-ish, convex-ish)."""
    din = np.array([u.shape[1] for u in units])
    dout = np.array([u.shape[0] for u in units])
    lo_k = np.array([k_bounds(d)[0] for d in din])
    hi_k = np.array([k_bounds(d)[1] for d in din])
    target = target_zero_count(units, s_global)
    n_total = int((dout * din).sum())
    k_levels = [np.clip(np.rint(np.array(levels)[None, :] * din[:, None]).astype(np.int64)[i], lo_k[i], hi_k[i]) for i in range(len(units))]
    # cost = kept weights; budget = N - target (at least `target` zeros)
    costs = [dout[i] * (din[i] - k_levels[i]) for i in range(len(units))]
    losses = [np.asarray(loss_tables[i], np.float64) for i in range(len(units))]
    ch = mckp_allocate(costs, losses, n_total - target)
    k0 = np.array([k_levels[i][ch[i]] for i in range(len(units))])

    def pen(u, kk):  # piecewise-linear interpolation of the loss table in k
        return float(np.interp(kk, k_levels[u], losses[u]))

    k, resid = fix_exact(k0, dout, lo_k, hi_k, target, pen)
    meta = {**(meta or {}), "rule": "mckp_equal_slope", "levels": list(levels)}
    return SparsityAllocation({u.uid: int(k[i]) for i, u in enumerate(units)}, {u.uid: float(k[i] / din[i]) for i, u in enumerate(units)},
                              target, int((dout * k).sum()), resid, meta)


# ----------------------------------------------------------------------------- Q3 rule

def mckp_bit_allocation(units, bit_options: tuple[int, ...], loss_tables: np.ndarray, byte_tables: np.ndarray,
                        budget_eligible_bytes: int) -> list[int]:
    """``loss_tables[u, j]``, ``byte_tables[u, j]`` for bit option j. Returns the chosen bits per unit."""
    ch = mckp_allocate([byte_tables[i] for i in range(len(units))], [loss_tables[i] for i in range(len(units))], budget_eligible_bytes)
    return [bit_options[c] for c in ch]
