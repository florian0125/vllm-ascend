"""Fused NPU proportional admission and unchanged token-wise LRC/LRU eviction.

Candidate order: current route count descending, historical frequency descending,
expert ID ascending. Victims: unprotected slots, LRC score ascending (or, on the
LRU branch, last-used step ascending — classic recency-only), slot ID ascending;
empty slots come first. Counts use exact rational integer rounding. The LRU
branch compiles out the LRC history/EMA/router state entirely: only ``Last`` and
``Step`` stay live, so it is strictly cheaper than LRC per step.
"""
import math

import torch
import triton
import triton.language as tl
from vllm_ascend.expert_offload.frequency_transfer_policy import FREQUENCY_DECAY


@triton.jit
def _initialize(SlotHeat, Protected, Candidates, Victims, S: tl.constexpr,
                L: tl.constexpr, B: tl.constexpr):
    x = tl.arange(0, B)
    tl.store(SlotHeat + x, float('-inf'), x < S)
    tl.store(Protected + x, 0, x < S)
    tl.store(Candidates + x, 0, x < L)
    tl.store(Victims + x, 0, x < L)


@triton.jit
def _counts_hotness(Ids, Weights, Mapping, Counts, Scores,
                    History, Seed, Ema, Router, Last, Step,
                    SlotHeat, Protected, Heat,
                    N: tl.constexpr, K: tl.constexpr, E: tl.constexpr,
                    W: tl.constexpr, S: tl.constexpr, HAS_WEIGHTS: tl.constexpr,
                    LRC: tl.constexpr, LRU: tl.constexpr,
                    BETA: tl.constexpr, LOG_BETA: tl.constexpr,
                    RW: tl.constexpr, EW: tl.constexpr, SW: tl.constexpr, AW: tl.constexpr,
                    BN: tl.constexpr, BW: tl.constexpr, BE: tl.constexpr, DECAY: tl.constexpr):
    e = tl.program_id(0) * BE + tl.arange(0, BE)
    r = tl.arange(0, BN)
    counts = tl.full((BN, BE), 0., tl.float32)
    latest = tl.full((BE,), -1, tl.int32)
    for j in tl.static_range(K):
        ids = tl.load(Ids + r * K + j, r < N, other=-1).to(tl.int32)
        match = (r[:, None] < N) & (e[None, :] < E) & (ids[:, None] == e[None, :])
        counts += match.to(tl.float32)
        latest = tl.maximum(latest, tl.max(tl.where(match, r[:, None] * K + j, -1), axis=0))
    total = tl.sum(counts, axis=0)
    old_score = tl.load(Scores + e, e < E, other=0)
    score = old_score * DECAY + total
    tl.store(Scores + e, score, e < E)
    tl.store(Counts + e, total, e < E)
    if LRC:
        h = tl.arange(0, BW)
        old_h = tl.load(History + (h[:, None] + N) * E + e[None, :],
                        (h[:, None] < W) & (h[:, None] + N < W) & (e[None, :] < E), other=0)
        new_r = N - W + h
        new_h = tl.full((BW, BE), 0., tl.float32)
        for j in tl.static_range(K):
            ids = tl.load(Ids + tl.maximum(new_r, 0) * K + j, (new_r >= 0) & (new_r < N), other=-1).to(tl.int32)
            match = (new_r[:, None] >= 0) & (new_r[:, None] < N) & (ids[:, None] == e[None, :])
            new_h += match.to(tl.float32)
        history = tl.where((h[:, None] + N) < W, old_h, new_h)
        tl.store(History + h[:, None] * E + e[None, :], history,
                 (h[:, None] < W) & (e[None, :] < E))
        if BETA == 0.:
            decay = (r == N - 1).to(tl.float32)
        else:
            decay = tl.exp(LOG_BETA * (N - 1 - r))
        addition = tl.sum(tl.where(r[:, None] < N, (counts > 0).to(tl.float32) * decay[:, None], 0.), axis=0)
        ema = tl.load(Ema + e, e < E, other=0) * (BETA ** N) + addition * (1. - BETA)
        tl.store(Ema + e, ema, e < E)
    if LRC or LRU:
        # Recency state shared by both policies: LRU consumes it directly as
        # the heat, LRC folds it into the age penalty.
        step = tl.load(Step)
        last_row = tl.max(tl.where(counts > 0, r[:, None] + 1, 0), axis=0)
        last = tl.load(Last + e, e < E, other=-1)
        last = tl.where(last_row > 0, step + last_row, last)
        tl.store(Last + e, last, e < E)
    if LRC:
        router = tl.load(Router + e, e < E, other=0)
        if HAS_WEIGHTS:
            latest_weight = tl.load(Weights + tl.maximum(latest, 0), (e < E) & (latest >= 0), other=0).to(tl.float32)
            router = tl.where(latest >= 0, latest_weight, router)
            tl.store(Router + e, router, e < E)
        seed = tl.load(Seed + e, e < E, other=0)
        recent = tl.sum(tl.where(h[:, None] < W, history, 0.), axis=0)
        age = tl.where(last < 0, 0., step + N - last)
        heat = RW * (seed + recent) + EW * ema + SW * router - AW * age
    elif LRU:
        # Classic LRU: heat IS the last-used step, so the ascending victim
        # order evicts least-recently-used first; never-used (-1) is coldest.
        heat = last
    else:
        heat = score
    tl.store(Heat + e, heat, e < E)
    slot = tl.load(Mapping + e, e < E, other=-1).to(tl.int32)
    tl.store(SlotHeat + tl.maximum(slot, 0), heat, (e < E) & (slot >= 0) & (slot < S))
    tl.store(Protected + tl.maximum(slot, 0), (total > 0).to(tl.int32), (e < E) & (slot >= 0) & (slot < S))


@triton.jit
def _rank_and_balance(Counts, Scores, Mapping, SlotHeat, Protected,
                      Candidates, Victims, Active, Chosen,
                      E: tl.constexpr, S: tl.constexpr, L: tl.constexpr,
                      NUM: tl.constexpr, DEN: tl.constexpr,
                      B_E: tl.constexpr, B_S: tl.constexpr, TILE: tl.constexpr):
    e = tl.arange(0, B_E)
    s = tl.arange(0, B_S)
    count = tl.load(Counts + e, e < E, other=0)
    score = tl.load(Scores + e, e < E, other=0)
    mapping = tl.load(Mapping + e, e < E, other=-1)
    missing = (e < E) & (count > 0) & (mapping < 0)
    available = (s < S) & (tl.load(Protected + s, s < S, other=1) == 0)
    cold = tl.load(SlotHeat + s, s < S, other=float('inf'))
    n_missing = tl.sum(missing.to(tl.int32), axis=0)
    n_available = tl.sum(available.to(tl.int32), axis=0)
    chosen = tl.minimum(tl.minimum(n_missing * NUM // DEN, n_available), L)
    x = tl.program_id(0) * TILE + tl.arange(0, TILE)
    my_count = tl.load(Counts + x, x < E, other=0)
    my_score = tl.load(Scores + x, x < E, other=0)
    my_map = tl.load(Mapping + x, x < E, other=-1)
    better = (count[None, :] > my_count[:, None]) | (
        (count[None, :] == my_count[:, None]) & (
            (score[None, :] > my_score[:, None]) | (
                (score[None, :] == my_score[:, None]) & (e[None, :] < x[:, None]))))
    rank = tl.sum((missing[None, :] & better).to(tl.int32), axis=1)
    tl.store(Candidates + rank, x, (x < E) & (my_count > 0) & (my_map < 0) & (rank < L))
    my_cold = tl.load(SlotHeat + x, x < S, other=float('inf'))
    my_available = (x < S) & (tl.load(Protected + x, x < S, other=1) == 0)
    colder = (cold[None, :] < my_cold[:, None]) | (
        (cold[None, :] == my_cold[:, None]) & (s[None, :] < x[:, None]))
    v_rank = tl.sum((available[None, :] & colder).to(tl.int32), axis=1)
    tl.store(Victims + v_rank, x, my_available & (v_rank < L))
    tl.store(Active + x, x < chosen, x < L)
    if tl.program_id(0) == 0:
        tl.store(Chosen, chosen)


@triton.jit
def _update_mapping(Mapping, Candidates, Victims, Active, Step,
                    E: tl.constexpr, L: tl.constexpr, N: tl.constexpr,
                    LRC: tl.constexpr, LRU: tl.constexpr,
                    B_E: tl.constexpr, B_L: tl.constexpr):
    e = tl.arange(0, B_E)
    k = tl.arange(0, B_L)
    mapping = tl.load(Mapping + e, e < E, other=-1)
    candidates = tl.load(Candidates + k, k < L, other=0).to(tl.int32)
    victims = tl.load(Victims + k, k < L, other=0).to(tl.int32)
    active = tl.load(Active + k, k < L, other=0).to(tl.int1)
    evicted = tl.sum(((mapping[:, None] == victims[None, :]) & active[None, :]).to(tl.int32), axis=1) > 0
    admitted = (e[:, None] == candidates[None, :]) & active[None, :]
    slot = tl.max(tl.where(admitted, victims[None, :], -1), axis=1)
    updated = tl.where(slot >= 0, slot, tl.where(evicted, -1, mapping))
    tl.store(Mapping + e, updated, e < E)
    if LRC or LRU:
        tl.store(Step, tl.load(Step) + N)


class ModelNew:
    """Allocation/metadata-only wrapper; all arithmetic runs in Triton kernels."""
    def __init__(self, policy):
        self.policy = policy
        self.e = policy.counts.numel()
        self.s = policy.slots.numel()
        self.l = policy.limit
        dev = policy.counts.device
        self.slot_heat = torch.empty(self.s, dtype=torch.float32, device=dev)
        self.protected = torch.empty(self.s, dtype=torch.int32, device=dev)
        self.heat = torch.empty(self.e, dtype=torch.float32, device=dev)
        self.candidates = torch.empty(self.l, dtype=torch.int64, device=dev)
        self.victims = torch.empty(self.l, dtype=torch.int64, device=dev)
        self.active = torch.empty(self.l, dtype=torch.bool, device=dev)
        self.chosen = torch.empty((), dtype=torch.int64, device=dev)

    def forward(self, ids, mapping, weights=None):
        if ids.ndim != 2 or ids.stride(1) != 1 or ids.stride(0) != ids.shape[1]:
            raise ValueError('Proportional policy requires contiguous rank-two IDs')
        if weights is not None and (weights.shape != ids.shape or weights.stride() != ids.stride()):
            raise ValueError('Proportional policy requires matching contiguous weights')
        p = self.policy
        lrc = getattr(p, 'lrc', None)
        lru = lrc is not None and bool(getattr(lrc, 'is_lru', False))
        # The attribute is named `lrc` for both flavors; only a non-LRU
        # policy object may drive the LRC kernel branches, which read/write
        # the LRC state tensors (history/ema/router) the LRU object lacks.
        lrc_flag = lrc is not None and not lru
        w = lrc.window if lrc is not None else 1
        beta = lrc.beta if lrc is not None else .9
        rw, ew, sw, aw = lrc.weights if lrc is not None else (0., 0., 0., 0.)
        dummy = p.scores
        _initialize[(1,)](self.slot_heat, self.protected, self.candidates, self.victims,
                           self.s, self.l, triton.next_power_of_2(max(self.s, self.l)))
        _counts_hotness[(triton.cdiv(self.e, 16),)](
            ids, weights if weights is not None else dummy, mapping, p.counts, p.scores,
            getattr(lrc, 'history', dummy), getattr(lrc, 'seed', dummy),
            getattr(lrc, 'ema', dummy), getattr(lrc, 'router_score', dummy),
            getattr(lrc, 'last_used', dummy), getattr(lrc, 'step', dummy),
            self.slot_heat, self.protected, self.heat,
            ids.shape[0], ids.shape[1], self.e, w, self.s, weights is not None,
            lrc_flag, lru, beta, math.log(beta) if beta else 0., rw, ew, sw, aw,
            triton.next_power_of_2(max(1, ids.shape[0])), triton.next_power_of_2(w), 16, FREQUENCY_DECAY)
        _rank_and_balance[(triton.cdiv(max(self.e, self.s), 16),)](
            p.counts, p.scores, mapping, self.slot_heat, self.protected,
            self.candidates, self.victims, self.active, self.chosen,
            self.e, self.s, self.l, p.ratio_num, p.ratio_den,
            triton.next_power_of_2(self.e), triton.next_power_of_2(self.s), 16)
        _update_mapping[(1,)](mapping, self.candidates, self.victims, self.active,
                               getattr(lrc, 'step', dummy),
                               self.e, self.l, ids.shape[0], lrc_flag, lru,
                               triton.next_power_of_2(self.e), triton.next_power_of_2(self.l))
        return self.candidates, self.victims, self.active
