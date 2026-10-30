"""Fixed-shape device mirror of recency-only (LRU) hotness; graph replay owns state.

Classic-LRU counterpart of :class:`DeviceLRCHotness` for the fused NPU policy
stack. Heat IS the last-used step, so victim ranking (heat ascending) evicts
the least-recently-used resident first; never-used experts keep ``last_used ==
-1`` and are colder than anything ever routed. The fused Triton kernel skips
the LRC history/EMA/router state entirely on this branch, so the only mutable
device state is ``last_used`` plus the step counter.
"""
import torch


class DeviceLRUHotness:
    # The fused kernel launcher reads this to pick the LRU heat branch.
    is_lru = True

    def __init__(self, experts, device, config, seed_state=None):
        self.experts = experts
        # Interface fields ModelNew.forward reads unconditionally; unused on
        # the LRU branch (compiled out) but must exist and be well-typed.
        self.window = 1
        self.beta = 0.9
        self.weights = (0.0, 0.0, 0.0, 0.0)
        self.last_used = torch.full((experts,), -1., device=device)
        self.step = torch.zeros((), device=device)
        if seed_state is not None:
            # Seed from the same offline hot table LRC uses so a hot-preloaded
            # cache is not torn down before runtime observation builds real
            # recency: hottest expert gets the newest timestamp, non-preloaded
            # experts stay -1 (evicted before any preloaded one).
            seeded = [eid for eid in range(experts) if seed_state.freq[eid] > 0]
            seeded.sort(key=lambda eid: (-seed_state.freq[eid], -seed_state.router_score[eid], eid))
            stamps = torch.full((experts,), -1.)
            for rank, eid in enumerate(reversed(seeded)):
                stamps[eid] = rank
            self.last_used.copy_(stamps.to(device))
            self.step.fill_(seed_state.step)

    def observe(self, ids, weights=None):
        """Pure-torch reference update; the fused kernel mirrors this exactly."""
        n, k = ids.shape
        valid = (ids >= 0) & (ids < self.experts)
        safe = ids.clamp(0, self.experts - 1).long()
        counts = torch.zeros((n, self.experts), device=ids.device)
        counts.scatter_add_(1, safe, valid.float())
        rows = torch.arange(1, n + 1, device=ids.device, dtype=torch.float32)
        last_row = torch.where(counts > 0, rows[:, None], 0).amax(0)
        self.last_used.copy_(torch.where(last_row > 0, self.step + last_row, self.last_used))
        self.step.add_(n)
        return self.hotness()

    def hotness(self):
        return self.last_used
