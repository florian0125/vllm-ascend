"""mode2 / pa+prevhs+ptlg learned predictor (on-NPU, weights on HBM).

    x = [ pre_attn[ℓ]  ‖  router_input[ℓ-1]  ‖  router_logits[t-1, ℓ] ]
        (in_dim = 2H + E)

``mode2_pa_prevhs`` plus one atom: "ptlg" = the PREVIOUS decode token's router
logits at THIS layer. Prefetch lead is unchanged — fetch_room_of("pa+prevhs+ptlg")
= min(pa:0, prevhs:1, ptlg:3) = 0, still bottlenecked by pre_attn[ℓ] — so this
predictor uses the SAME current-layer driver as mode2_pa_prevhs (predicts_next_layer
stays False): predict before attention, prefetch overlapping attention + gate.
The ptlg block costs nothing in lead: it is complete before the token even starts.

Concrete predictor: defines ONLY the input composition. All shared machinery —
checkpoint load to HBM, per-arch head dispatch, z-scoring, the MoE-index -> head-index
offset mapping, and the on-NPU forward driver — lives in ``LearnedNPUPredictor``
(vllm_ascend/expert_offload/expert_predictor.py). The ptlg cache itself lives in the
manager (``_ai_lg_buf``, filled by ``ai_save_router_logits``).
"""

from __future__ import annotations

import torch

from vllm_ascend.expert_offload.expert_predictor import (
    AIPredictCtx,
    LearnedNPUPredictor,
    register_predictor,
)


@register_predictor("mode2_pa_prevhs_ptlg")
class Mode2PaPrevhsPtlgPredictor(LearnedNPUPredictor):
    """mode2 (within-token, between-layer), composition 'pa+prevhs+ptlg':

        x = [ pre_attn[ℓ]  ‖  router_input[ℓ-1]  ‖  router_logits[t-1, ℓ] ]
            (in_dim = 2H + E)

    This mode is NOT one of the study's six legacy bespoke modes, so it is built by
    the GENERIC atom path (study build_mode_base line ~2109: concatenate
    build_atom(a) for a in mode.split("+")). Its first two blocks come out
    bit-identical to the legacy "pa+prevhs" construction, so only the third block
    is new here:

      * pa      — pre_attn[ℓ], no fallback (study: out[:, :, :H] = pa_cpu).
      * prevhs  — router_input[ℓ-1], ZERO-filled at the first COVERED layer
                  (head 0, no predecessor): build_atom zeroes it because its ℓ=0
                  fallback hidden[0] arrives too late to prefetch (study lines
                  ~2045-2049 / ~2061-2062), matching legacy out[:, 0, H:] = 0.
                  Supplied as ctx.router_input_prev=None by the driver at head 0.
      * ptlg    — router_logits[t-1, ℓ]. NO layer shift (ATOM_SPEC["ptlg"] =
                  ("lg", False, True, True)), so head 0 uses the previous token's
                  head-0 logits — a REAL feature, not zeros. The ONLY zero case is
                  a prompt's FIRST decode token, which has no predecessor (study
                  build_atom line ~2069: out[prev_tok_bad] = 0); the manager
                  produces that by zeroing _ai_lg_buf on the preceding prefill pass.

    IN_FEATURES=2 / E_FEATURES=1 tell the base class in_dim = 2H + 1*E so it can
    recover H. The concat order (pa, prevhs, ptlg) is mandatory: it must match the
    order the checkpoint was trained with (_mode_atoms order in build_mode_base).
    """

    IN_FEATURES = 2
    E_FEATURES = 1
    INPUT_SPEC = frozenset({"pre_attn", "router_input_prev", "router_logits_prev"})

    def assemble(self, layer_idx: int, ctx: AIPredictCtx) -> torch.Tensor:
        pre_attn = ctx.pre_attn                                 # [n, H]
        router_prev = ctx.router_input_prev                     # [n, H] | None @ head 0
        if router_prev is None:
            router_prev = torch.zeros(ctx.n_tokens, self.H,
                                      dtype=torch.float32, device=pre_attn.device)
        logits_prev = ctx.router_logits_prev                   # [n, E] | None
        if logits_prev is None:
            # Belt-and-braces: with the current driver the manager always hands
            # over the cache slice (zeroed at a prompt's first decode token), so
            # this is only hit if a future driver cannot supply the block.
            logits_prev = torch.zeros(ctx.n_tokens, self.E,
                                      dtype=torch.float32, device=pre_attn.device)
        return torch.cat([pre_attn, router_prev, logits_prev], dim=-1)   # pa ‖ prevhs ‖ ptlg (mandatory)
        