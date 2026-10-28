"""Fixed-shape device mirror of token-wise LRC hotness; graph replay owns state."""

import math

import torch

try:
    import triton
    from vllm_ascend.expert_offload.proportional_lrc_kernel import _counts_hotness
    from vllm_ascend.expert_offload.frequency_transfer_policy import FREQUENCY_DECAY
except ImportError:
    triton = None
    _counts_hotness = None
    FREQUENCY_DECAY = 0.99


class DeviceLRCHotness:
    def __init__(self, experts, device, config, seed_state=None):
        self.experts = experts
        self.window = config.cache_recent_window
        self.beta = config.cache_ema_beta
        self.weights = (
            config.cache_recent_weight,
            config.cache_ema_weight,
            config.cache_router_weight,
            config.cache_age_weight,
        )
        self.history = torch.zeros((self.window, experts), device=device)
        self.seed = torch.zeros(experts, device=device)
        self.ema = torch.zeros(experts, device=device)
        self.router_score = torch.zeros(experts, device=device)
        self.last_used = torch.full((experts,), -1., device=device)
        self.step = torch.zeros((), device=device)
        self._hotness = None
        self._fused_counts = torch.zeros(experts, device=device)
        self._fused_scores = torch.zeros(experts, device=device)
        self._fused_mapping = torch.full((experts,), -1, dtype=torch.int32, device=device)
        self._fused_slot_heat = torch.zeros(1, dtype=torch.float32, device=device)
        self._fused_protected = torch.zeros(1, dtype=torch.int32, device=device)
        if seed_state is not None:
            self.seed.copy_(torch.tensor(seed_state.freq, dtype=torch.float32, device=device))
            self.ema.copy_(torch.as_tensor(seed_state.ema, dtype=torch.float32, device=device))
            self.router_score.copy_(torch.tensor(seed_state.router_score, device=device))
            self.last_used.copy_(torch.tensor(seed_state.last_used, dtype=torch.float32, device=device))
            self.step.fill_(seed_state.step)

    def observe(self, ids, weights=None):
        n, k = ids.shape
        if (
            _counts_hotness is not None
            and weights is not None
            and ids.device.type != "cpu"
            and ids.is_contiguous()
            and weights.is_contiguous()
            and n > 0
            and k > 0
            and n <= self.window
        ):
            out = torch.empty_like(self.seed)
            rw, ew, sw, aw = self.weights
            _counts_hotness[(triton.cdiv(self.experts, 16),)](
                ids, weights, self._fused_mapping,
                self._fused_counts, self._fused_scores,
                self.history, self.seed, self.ema, self.router_score,
                self.last_used, self.step,
                self._fused_slot_heat, self._fused_protected, out,
                n, k, self.experts, self.window, 1, True, True, False,
                self.beta, math.log(self.beta) if self.beta else 0.0,
                rw, ew, sw, aw,
                triton.next_power_of_2(max(1, n)),
                triton.next_power_of_2(self.window), 16, FREQUENCY_DECAY,
            )
            self.step.add_(n)
            self._hotness = out
            return out

        valid = (ids >= 0) & (ids < self.experts)
        safe = ids.clamp(0, self.experts - 1).long()
        counts = torch.zeros((n, self.experts), device=ids.device)
        counts.scatter_add_(1, safe, valid.float())
        self.history.copy_(torch.cat((self.history, counts), dim=0)[-self.window:])
        rows = torch.arange(n, device=ids.device, dtype=torch.float32)
        decay = torch.pow(torch.full_like(rows, self.beta), n - 1 - rows)
        self.ema.mul_(self.beta ** n).add_(((counts > 0).float() * decay[:, None]).sum(0) * (1 - self.beta))
        last_row = torch.where(counts > 0, rows[:, None] + 1, 0).amax(0)
        self.last_used.copy_(torch.where(last_row > 0, self.step + last_row, self.last_used))
        if weights is not None:
            positions = torch.arange(n*k, device=ids.device).reshape(n, k)
            expert_ids = torch.arange(self.experts, device=ids.device)
            latest = torch.where(
                (safe.reshape(-1, 1) == expert_ids[None, :]) & valid.reshape(-1, 1),
                positions.reshape(-1, 1), -1
            ).amax(0)
            latest_score = weights.reshape(-1).float().gather(0, latest.clamp(min=0))
            self.router_score.copy_(torch.where(latest >= 0, latest_score, self.router_score))
        self.step.add_(n)
        self._hotness = self._compute_hotness()
        return self._hotness

    def _compute_hotness(self):
        rw, ew, sw, aw = self.weights
        age = torch.where(self.last_used < 0, 0., self.step - self.last_used)
        return rw * (self.seed + self.history.sum(0)) + ew*self.ema + sw*self.router_score - aw*age

    def hotness(self):
        if self._hotness is None:
            self._hotness = self._compute_hotness()
        return self._hotness
