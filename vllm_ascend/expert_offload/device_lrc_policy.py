"""Fixed-shape device mirror of token-wise LRC hotness; graph replay owns state."""
import torch

class DeviceLRCHotness:
    def __init__(self, experts, device, config, seed_state=None):
        self.experts = experts
        self.window = config.cache_recent_window
        self.beta = config.cache_ema_beta
        self.weights = (config.cache_recent_weight, config.cache_ema_weight,
                        config.cache_router_weight, config.cache_age_weight)
        self.history = torch.zeros((self.window, experts), device=device)
        self.seed = torch.zeros(experts, device=device)
        self.ema = torch.zeros(experts, device=device)
        self.router_score = torch.zeros(experts, device=device)
        self.last_used = torch.full((experts,), -1., device=device)
        self.step = torch.zeros((), device=device)
        if seed_state is not None:
            self.seed.copy_(torch.tensor(seed_state.freq, dtype=torch.float32, device=device))
            self.ema.copy_(torch.as_tensor(seed_state.ema, dtype=torch.float32, device=device))
            self.router_score.copy_(torch.tensor(seed_state.router_score, device=device))
            self.last_used.copy_(torch.tensor(seed_state.last_used, dtype=torch.float32, device=device))
            self.step.fill_(seed_state.step)

    def observe(self, ids, weights=None):
        n, k = ids.shape
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
            latest = torch.where((safe.reshape(-1, 1) == expert_ids[None, :]) & valid.reshape(-1, 1),
                                 positions.reshape(-1, 1), -1).amax(0)
            latest_score = weights.reshape(-1).float().gather(0, latest.clamp(min=0))
            self.router_score.copy_(torch.where(latest >= 0, latest_score, self.router_score))
        self.step.add_(n)
        return self.hotness()

    def hotness(self):
        rw, ew, sw, aw = self.weights
        age = torch.where(self.last_used < 0, 0., self.step - self.last_used)
        return rw * (self.seed + self.history.sum(0)) + ew*self.ema + sw*self.router_score - aw*age
