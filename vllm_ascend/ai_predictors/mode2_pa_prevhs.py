"""mode2 / pa+prevhs learned predictor (on-NPU, weights on HBM).

    x = [ pre_attn[ℓ]  ‖  router_input[ℓ-1] ]      (in_dim = 2H)

Concrete predictor: defines ONLY the input composition. All shared machinery —
checkpoint load to HBM, per-arch head dispatch, z-scoring, the MoE-index ->
head-index offset mapping, and the on-NPU forward driver — lives in
``LearnedNPUPredictor`` (vllm_ascend/expert_offload/expert_predictor.py).
"""

from __future__ import annotations

import torch

from vllm_ascend.expert_offload.expert_predictor import (
    AIPredictCtx,
    LearnedNPUPredictor,
    register_predictor,
)


@register_predictor("mode2_pa_prevhs")
class Mode2PaPrevhsPredictor(LearnedNPUPredictor):
    """mode2 (within-token, between-layer), composition 'pa+prevhs':

        x = [ pre_attn[ℓ]  ‖  router_input[ℓ-1] ]      (in_dim = 2H)

    ``router_input[ℓ-1]`` is zero-filled at the first COVERED layer (head 0, no
    predecessor), matching the training feature builder. The concat order is
    mandatory: it must match the order the checkpoint was trained with.
    """

    IN_FEATURES = 2
    INPUT_SPEC = frozenset({"pre_attn", "router_input_prev"})

    def assemble(self, layer_idx: int, ctx: AIPredictCtx) -> torch.Tensor:
        pre_attn = ctx.pre_attn                                 # [n, H]
        router_prev = ctx.router_input_prev                    # [n, H] | None @ head 0
        if router_prev is None:
            router_prev = torch.zeros(ctx.n_tokens, self.H,
                                      dtype=torch.float32, device=pre_attn.device)
        return torch.cat([pre_attn, router_prev], dim=-1)      # pa ‖ prevhs (mandatory)