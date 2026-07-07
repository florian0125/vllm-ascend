"""mode2 / prevpa+prevhs learned predictor (on-NPU, weights on HBM), twotower head.

    x = [ pre_attn[ℓ-1]  ‖  router_input[ℓ-1] ]      (in_dim = 2H)

A *next-layer* / lookahead predictor (study FETCH_ROOM=1): the features for layer
ℓ come from layer ℓ-1, so at runtime layer n we predict layer n+1. The manager's
next-layer driver (predicts_next_layer) captures pre_attn[n] before attention and,
after layer n's on-demand load, assembles pre_attn[n] ‖ router_input[n] -> predicts
layer n+1's router logits -> prefetches n+1 (H2D overlaps layer n's MoE MLP).

head 0 (first covered layer) has no trained predecessor: training fills its router
half with zeros (study build_mode_base out[:,0,H:]=0). The driver passes
ctx.router_input=None at head 0 so assemble() zeros that half — matching training.
The concat order (prevpa then prevhs) is mandatory: it must match the checkpoint.
"""

from __future__ import annotations

import torch

from vllm_ascend.expert_offload.expert_predictor import (
    AIPredictCtx,
    LearnedNPUPredictor,
    register_predictor,
)


@register_predictor("mode2_prevpa_prevhs_2tower")
class Mode2PrevpaPrevhsTwoTowerPredictor(LearnedNPUPredictor):
    """mode2 (within-token, between-layer), composition 'prevpa+prevhs',
    twotower head:

        x = [ pre_attn[ℓ-1]  ‖  router_input[ℓ-1] ]      (in_dim = 2H)

    Driven by the next-layer flow (predicts_next_layer = True): predicts layer
    n+1 at layer n after the on-demand load. head 0 zeros the router half to match
    the training feature builder.
    """

    IN_FEATURES = 2
    INPUT_SPEC = frozenset({"pre_attn", "router_input"})
    predicts_next_layer = True

    def assemble(self, layer_idx: int, ctx: AIPredictCtx) -> torch.Tensor:
        pre_attn = ctx.pre_attn                                 # [n, H]
        router_cur = ctx.router_input                           # [n, H] | None @ head 0
        if router_cur is None:
            router_cur = torch.zeros(ctx.n_tokens, self.H,
                                     dtype=torch.float32, device=pre_attn.device)
        return torch.cat([pre_attn, router_cur], dim=-1)       # prevpa ‖ prevhs (mandatory)