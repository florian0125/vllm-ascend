"""Proportional CPU/copy balance; LRC selects protected cold evictions.

For M missing experts: k=floor(M*t_cpu/(t_cpu+t_copy)), capped by available
slots and balance_max_transfers. Fixed launch costs and NPU compute estimates
are intentionally excluded. Admission order is current route count, then
historical frequency; LRC affects victims only.
"""
from fractions import Fraction
import math

import torch

from vllm_ascend.expert_offload.frequency_transfer_policy import FREQUENCY_DECAY


class LoadBalanceTransferPolicy:
    def __init__(self, experts, slots, device, config):
        self.experts = torch.arange(experts, device=device, dtype=torch.int64)
        self.slots = torch.arange(slots, device=device, dtype=torch.int64)
        self.scores = torch.zeros(experts, device=device, dtype=torch.float32)
        self.counts = torch.zeros_like(self.scores)
        self.limit = min(config.balance_max_transfers, slots, experts)
        if self.limit < 1:
            raise ValueError('Load balancing requires at least one NPU expert slot')
        self.entries = torch.arange(self.limit, device=device)
        self.cpu_expert_us = float(config.balance_cpu_expert_us)
        self.copy_us = float(config.balance_copy_us)
        if (not math.isfinite(self.cpu_expert_us) or not math.isfinite(self.copy_us)
                or self.cpu_expert_us < 0 or self.copy_us < 0
                or self.cpu_expert_us + self.copy_us <= 0):
            raise ValueError('CPU/copy unit times must be finite, nonnegative and not both zero')
        cpu = Fraction(str(self.cpu_expert_us))
        copy = Fraction(str(self.copy_us))
        ratio = cpu / (cpu + copy)
        self.ratio_num, self.ratio_den = ratio.numerator, ratio.denominator
        if self.ratio_num * experts > 2**31 - 1 or self.ratio_den > 2**31 - 1:
            raise ValueError('Unit-time precision exceeds exact int32 proportional balance range')
        self._fused = None

    def select(self, ids, mapping, weights=None):
        if ids.device.type == 'npu':
            if self._fused is None:
                from vllm_ascend.expert_offload.proportional_lrc_kernel import ModelNew
                self._fused = ModelNew(self)
                self.chosen_count = self._fused.chosen
            return self._fused.forward(ids, mapping, weights)
        return self.reference(ids, mapping, weights)

    def reference(self, ids, mapping, weights=None):
        """Trusted PyTorch reference, also the generic non-NPU implementation."""
        flat = ids.reshape(-1).long()
        valid = (flat >= 0) & (flat < mapping.numel())
        safe = flat.clamp(0, mapping.numel() - 1)
        self.counts.zero_().scatter_add_(0, safe, valid.float())
        self.scores.mul_(FREQUENCY_DECAY).add_(self.counts)
        routed = self.counts > 0
        missing = routed & (mapping < 0)
        owners = mapping[:, None] == self.slots[None, :]
        available = ~(owners & routed[:, None]).any(dim=0)
        n_missing = missing.to(torch.int64).sum()
        chosen = torch.minimum(n_missing * self.ratio_num // self.ratio_den,
                               available.to(torch.int64).sum()).clamp(max=self.limit)
        # Stable lexicographic sorting makes ties deterministic across backends.
        order = torch.argsort(self.scores, descending=True, stable=True)
        demand = torch.where(missing, self.counts, -1.)
        order = order[torch.argsort(demand[order], descending=True, stable=True)]
        candidates = torch.where(self.entries < n_missing, order[:self.limit], 0)
        lrc = getattr(self, 'lrc', None)
        heat = lrc.observe(ids, weights) if lrc is not None else self.scores
        slot_heat = torch.where(owners, heat[:, None], float('-inf')).amax(dim=0)
        victims = torch.argsort(torch.where(available, slot_heat, float('inf')), stable=True)[:self.limit]
        victims = torch.where(self.entries < available.to(torch.int64).sum(), victims, 0)
        active = self.entries < chosen
        evicted = ((mapping[:, None] == victims[None, :]) & active[None, :]).any(dim=1)
        updated = torch.where(evicted, -1, mapping)
        admitted = (self.experts[:, None] == candidates[None, :]) & active[None, :]
        new_slots = torch.where(admitted, victims[None, :], -1).amax(dim=1)
        mapping.copy_(torch.where(new_slots >= 0, new_slots.to(mapping.dtype), updated))
        return candidates, victims, active
