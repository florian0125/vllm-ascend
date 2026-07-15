"""mode2 / prevpa+prevhs+ptlg learned predictor (on-NPU, weights on HBM), lowrank head.

    x = [ pre_attn[ℓ-1]  ‖  router_input[ℓ-1]  ‖  router_logits[t-1, ℓ] ]
        (in_dim = 2H + E)

``mode2_prevpa_prevhs_2tower``'s composition plus one atom: "ptlg" = the PREVIOUS
decode token's router logits at the PREDICTED layer. Two things differ from that
predictor and BOTH are checkpoint metadata, not code:
  * arch  — this checkpoint is "lowrank" (LowRankProbe), so meta["arch"] dispatches
            to _LowRankHead instead of _TwoTowerHead. No head work: _LowRankHead
            already mirrors the study's LowRankProbe.forward (study lines 1055-1061)
            and reads the same param names the study saves (Wd/W1/b1/Wo/bo).
  * in_dim — 2H + E instead of 2H, handled by IN_FEATURES=2 / E_FEATURES=1.

A *next-layer* / lookahead predictor (study FETCH_ROOM=1), same as the 2tower
sibling: fetch_room_of("prevpa+prevhs+ptlg") = min(prevpa:1, prevhs:1, ptlg:3) = 1,
so the prevpa/prevhs atoms still bottleneck the lead and the ptlg atom is free
(complete before the token even starts). Hence predicts_next_layer = True and the
timing is unchanged: at runtime layer n we predict layer n+1, the manager's
next-layer driver captures pre_attn[n] before attention and, after layer n's
on-demand load, assembles pre_attn[n] ‖ router_input[n] ‖ logits[t-1, n+1] ->
predicts layer n+1's router logits -> prefetches n+1 (H2D overlaps layer n's MoE MLP).

Corner cases, all matching the training feature builder exactly:
  * head 0 (first covered layer) has no trained predecessor. prevpa EDGE-PADS to
    the layer's own pre_attn (study build_atom lines ~2040-2041/2064: the "pa"
    stream is ready before layer 0's attention, so it keeps its prefetch lead),
    while prevhs ZERO-fills (its ℓ=0 fallback hidden[0] is the current token's
    layer-0 router input — too late to prefetch; study lines ~2045-2049/2061-2062).
    The driver runs head 0 as a CURRENT-layer predict from ai_capture_pre_attn and
    passes ctx.router_input=None so assemble() zeros that block.
  * ptlg has NO layer shift, so head 0 does NOT zero it: it carries the previous
    token's head-0 logits, a REAL feature. The ONLY zero case is a prompt's FIRST
    decode token, which has no predecessor (study build_atom line ~2069:
    out[prev_tok_bad] = 0); the manager produces that by zeroing _ai_lg_buf on the
    preceding prefill pass (see ai_save_router_logits).

Zero-filling happens PRE-z-score, exactly as in training: assemble() returns the raw
x and predict_logits_npu applies (x - mu[head]) / sd[head] afterwards, so a zeroed
block becomes -mu/sd, not 0. The checkpoint's mu/sd were fit over a train partition
that contains those same zero rows, so this is correct — do not special-case it.

The concat order (prevpa, prevhs, ptlg) is mandatory: it must match the order the
checkpoint was trained with (_mode_atoms order in build_mode_base's generic
atom path, study line ~2109).
"""

from __future__ import annotations

import torch

from vllm_ascend.expert_offload.expert_predictor import (
    AIPredictCtx,
    LearnedNPUPredictor,
    register_predictor,
)


@register_predictor("mode2_prevpa_prevhs_ptlg")
class Mode2PrevpaPrevhsPtlgPredictor(LearnedNPUPredictor):
    """mode2 (within-token, between-layer), composition 'prevpa+prevhs+ptlg',
    lowrank head:

        x = [ pre_attn[ℓ-1]  ‖  router_input[ℓ-1]  ‖  router_logits[t-1, ℓ] ]
            (in_dim = 2H + E)

    Driven by the next-layer flow (predicts_next_layer = True): predicts layer
    n+1 at layer n after the on-demand load. head 0 zeros the router-input block
    (no trained predecessor) but keeps a real ptlg block; a prompt's first decode
    token zeros ptlg. IN_FEATURES=2 / E_FEATURES=1 tell the base class
    in_dim = 2H + 1*E so it can recover H.

    NOTE: the field names below are current-layer ("pre_attn", "router_input")
    because the driver runs at layer n and TARGETS n+1 — the semantics are the
    study's prevpa/prevhs, identical to Mode2PrevpaPrevhsTwoTowerPredictor.
    ctx.router_logits_prev is the exception: the manager reads it at the TARGET
    layer's head, so it is already logits[t-1, ℓ] for the predicted ℓ.
    """

    IN_FEATURES = 2
    E_FEATURES = 1
    INPUT_SPEC = frozenset({"pre_attn", "router_input", "router_logits_prev"})
    predicts_next_layer = True

    def assemble(self, layer_idx: int, ctx: AIPredictCtx) -> torch.Tensor:
        pre_attn = ctx.pre_attn                                 # [n, H]
        router_cur = ctx.router_input                           # [n, H] | None @ head 0
        if router_cur is None:
            router_cur = torch.zeros(ctx.n_tokens, self.H,
                                     dtype=torch.float32, device=pre_attn.device)
        logits_prev = ctx.router_logits_prev                    # [n, E] | None
        if logits_prev is None:
            # Belt-and-braces: with the current driver the manager always hands
            # over the cache slice (zeroed at a prompt's first decode token), so
            # this is only hit if a future driver cannot supply the block.
            logits_prev = torch.zeros(ctx.n_tokens, self.E,
                                      dtype=torch.float32, device=pre_attn.device)
        return torch.cat([pre_attn, router_cur, logits_prev], dim=-1)   # prevpa ‖ prevhs ‖ ptlg (mandatory)
    