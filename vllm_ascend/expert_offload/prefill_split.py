"""Prefill split-GMM: compute decode-resident experts in place.

Prefill expert paging, split mode: rows whose expert is already resident
in the decode pool are computed by GMM-1 in place (zero copies), the rest
by GMM-2 over the paged-in pool experts.  The resident set is NOT a
contiguous id prefix but the decode expert pool --
the per-layer ``num_device_experts`` slots tracked by ``layer.log2phy``.  Rows
routed to resident experts are therefore scattered across the dispatch output
(not split at a single boundary), so the row partition is a stable sort on a
per-row residency flag instead of one slice point.

Ownership model (single-card prefill only):

* Resident experts (``log2phy[eid] >= 0``): their weights stay in the decode
  slots of ``layer.w13_weight``/``layer.w2_weight``.  GMM-1 reads them where
  they are -- zero copies.  Rows are grouped by SLOT order (bincount over the
  slot index of each row) because the decode weight dim0 is the slot axis.
* Non-resident experts: H2D-loaded into the prefill pool (identity layout,
  resident ids left as holes).  GMM-2 runs over the pool with a group_list
  that has zero counts at the resident ids.

Dispatch still sees the FULL local expert count through the identity
``_prefill_log2phy`` (num_local_experts lesson: never
let dispatch derive the expert space from a shrunk weight).  The split happens
after ``build_mlp_compute_input``, so dispatch/combine are untouched.
"""

import dataclasses
from dataclasses import dataclass

import torch

from vllm_ascend import envs
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights


def _debug_enabled() -> bool:
    return envs.VLLM_ASCEND_PREFILL_SPLIT_DEBUG == "1"


@dataclass(frozen=True, slots=True)
class PrefillSplitCtx:
    """Per-forward state handed from the offload prefill integration to
    MoECommMethod.fused_experts, describing how to split the MLP compute.

    resident_slots: int32/int64 [num_total_experts] on device.
        resident_slots[eid] = decode slot index when expert eid is
        decode-pool-resident, else -1.
    decode_weights: MoEWeights built from the layer's decode-slot tensors,
        using the calling quant method's own list/tensor conventions.
    decode_layer: the ORIGINAL layer object carrying the decode-slot weights.
        main0928's quant methods (e.g. w4a8_mxfp4 apply_gmm1/gmm2) read
        weights from ``mlp_compute_input.layer`` rather than ``.weights``,
        so GMM-1 must swap the layer back from the pool view.
    num_slots: dim0 of the decode weights (= len of the slot axis).
    layer_idx: for debug logs only.
    """

    resident_slots: torch.Tensor
    decode_weights: object
    decode_layer: object = None
    num_slots: int = 0
    layer_idx: int = -1


def prefill_split_apply(
    mlp_compute_input: MoEMlpComputeInput,
    ctx: PrefillSplitCtx,
    apply_mlp,
):
    """Run the MLP stage as two GMMs: decode-slot experts, then pool experts.

    ``apply_mlp`` is ``MoECommMethod._apply_mlp`` (returns
    ``(output, before_gmm2_evt)``).  Falls back to a single fused call when
    the layout assumption (count-type group_list over the full local expert
    space) does not hold.
    """
    group_list = mlp_compute_input.group_list

    # The AllGather prefill dispatcher emits count-type group_list covering
    # every local expert id in ascending order.  Anything else (MC2 cumsum,
    # All2All shard space) cannot be split by a re-group.  Failing loudly is
    # the only safe option here: with split_gmm enabled the pool deliberately
    # carries holes at the resident ids, so a single-GMM fallback would read
    # stale weights silently.
    if mlp_compute_input.group_list_type != 1:
        raise RuntimeError(
            "prefill_split_gmm requires a count-type group_list over all "
            f"local experts (AllGather prefill); got group_list_type="
            f"{mlp_compute_input.group_list_type}. Disable prefill_split_gmm "
            "or switch the MoE comm method.")

    from vllm_ascend.ascend_forward_context import _EXTRA_CTX

    if _EXTRA_CTX.capturing:
        # The split syncs on n_res (host .item()) -- impossible under graph
        # capture.  Prefill must run eager with this feature.
        raise RuntimeError(
            "prefill_split_gmm is incompatible with graph capture; "
            "run prefill with --enforce-eager.")

    hidden_states = mlp_compute_input.hidden_states
    ntotal = group_list.shape[0]
    num_rows = hidden_states.shape[0]
    resident_slots = ctx.resident_slots
    device = hidden_states.device

    # Per-row expert id, rebuilt from the count-type group_list: rows are
    # sorted by expert id ascending, so repeat_interleave over the counts
    # restores each row's owner.  The AllGather dispatcher emits exactly
    # group_list.sum() rows; anything else means the layout contract is
    # broken (mis-grouped rows would silently read the wrong weights).
    row_eid = torch.repeat_interleave(
        torch.arange(ntotal, dtype=torch.int64, device=device),
        group_list.to(torch.int64),
    )
    if row_eid.shape[0] != num_rows:
        raise RuntimeError(
            "prefill_split_gmm expects dispatch rows == group_list sum "
            f"(AllGather layout); got {row_eid.shape[0]} rows vs "
            f"{int(group_list.sum().item())} routed. Disable "
            "prefill_split_gmm for this comm layout.")

    slot_of_row = resident_slots[row_eid].to(torch.int64)
    is_resident = slot_of_row >= 0
    n_res = int(is_resident.sum().item())
    n_non = num_rows - n_res

    if n_res == 0:
        # Nothing routed to the decode pool: pure pool GMM, no permutation.
        return apply_mlp(mlp_compute_input)

    # Stable partition, resident rows first: sort key 0 for resident, 1 for
    # the rest.  Stability keeps eid order inside each half, which is what
    # the pool group_list re-grouping below relies on.
    perm = torch.argsort((~is_resident).to(torch.int8), stable=True)
    idx_res = perm[:n_res]
    idx_non = perm[n_res:]

    debug = _debug_enabled()

    # ---- GMM-1: decode-resident experts, read in place from the slots ----
    # The decode weight dim0 is the SLOT axis, so the resident half must be
    # re-sorted by slot (stable) and grouped in slot order -- eid order and
    # slot order differ because the pool residency is an arbitrary id->slot
    # mapping.  scatter_add_ instead of bincount -- same result, and an op
    # set known to be supported on NPU.
    slot_res = slot_of_row.index_select(0, idx_res)
    idx_res = idx_res.index_select(0, torch.argsort(slot_res, stable=True))
    gl_res = _group_counts(slot_res, ctx.num_slots)
    res_slice = _mlp_slice(
        mlp_compute_input,
        row_idx=idx_res,
        hidden_states=_row_take(hidden_states, idx_res),
        group_list=gl_res,
        weights=ctx.decode_weights,
        layer=ctx.decode_layer,
    )
    out_res, _ = apply_mlp(res_slice)
    if debug:
        print(f"[PREFILL_SPLIT] layer={ctx.layer_idx} rows={num_rows} "
              f"n_res={n_res} n_non={n_non}", flush=True)

    if n_non == 0:
        # Every row went through GMM-1, which reorders rows into slot
        # groups -- scatter back to dispatch-row order before returning.
        # The GMM output dtype (bf16/fp16), not the quantized input dtype.
        final = torch.empty(hidden_states.shape, dtype=out_res.dtype,
                            device=hidden_states.device)
        final.index_copy_(0, idx_res, out_res)
        return final, None

    # ---- GMM-2: remaining experts from the prefill pool (identity layout,
    # holes at the resident ids get zero counts) ----
    gl_non = _group_counts(row_eid.index_select(0, idx_non), ntotal)
    non_slice = _mlp_slice(
        mlp_compute_input,
        row_idx=idx_non,
        hidden_states=_row_take(hidden_states, idx_non),
        group_list=gl_non,
        weights=mlp_compute_input.weights,
    )
    out_non, evt = apply_mlp(non_slice)

    # ---- Reassemble in dispatch-row order ----
    # GMM outputs are bf16/fp16 even when the dispatch rows are fp8-quantized;
    # size the buffer from the GMM output, not empty_like(quantized input).
    final = torch.empty(hidden_states.shape, dtype=out_non.dtype,
                        device=hidden_states.device)
    final.index_copy_(0, idx_res, out_res)
    final.index_copy_(0, idx_non, out_non)
    return final, evt


def _group_counts(ids: torch.Tensor, size: int) -> torch.Tensor:
    """Per-group row counts (int64 [size]) -- bincount via scatter_add_."""
    counts = torch.zeros(size, dtype=torch.int64, device=ids.device)
    counts.scatter_add_(0, ids, torch.ones_like(ids))
    return counts


_ROW_TAKE_8BIT_DTYPES = None


def _row_take(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """index_select rows, working around NPU fp8 gaps.

    The aclnn index_select kernel is not implemented for 8-bit float dtypes
    (fp8 e4m3 activations, e8m0 scales).  Those are all single-byte types, so
    re-index through a uint8 bit view -- element size is unchanged, no
    precision is touched.
    """
    global _ROW_TAKE_8BIT_DTYPES
    if _ROW_TAKE_8BIT_DTYPES is None:
        _ROW_TAKE_8BIT_DTYPES = {
            torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e8m0fnu,
            torch.uint8, torch.int8,
        }
    if t.dtype in _ROW_TAKE_8BIT_DTYPES:
        return t.view(torch.uint8).index_select(0, idx).view(t.dtype)
    return t.index_select(0, idx)


def _mlp_slice(
    mlp_compute_input: MoEMlpComputeInput,
    *,
    row_idx: torch.Tensor,
    hidden_states: torch.Tensor,
    group_list: torch.Tensor,
    weights,
    layer=None,
) -> MoEMlpComputeInput:
    """Row-sliced view of the MLP compute input for one GMM.

    index_select outputs are freshly allocated (contiguous, storage_offset 0)
    -- required because the aclnn grouped-matmul variants mis-handle
    storage_offset != 0 inputs.
    expanded_row_idx/topk_ids/lora are dispatch-side metadata the MLP stage
    must not consume on a partial row set.
    ``layer`` (default: keep) swaps the gmm weight source for the decode-slot
    GMM — main0928 quant methods read weights from ``.layer``.
    """
    dynamic_scale = mlp_compute_input.dynamic_scale
    if dynamic_scale is not None:
        dynamic_scale = _row_take(dynamic_scale, row_idx)
    topk_scales = mlp_compute_input.topk_scales
    if topk_scales is not None:
        topk_scales = _row_take(topk_scales, row_idx)
    return dataclasses.replace(
        mlp_compute_input,
        hidden_states=hidden_states,
        group_list=group_list.contiguous(),
        dynamic_scale=dynamic_scale,
        topk_scales=topk_scales,
        weights=weights,
        layer=mlp_compute_input.layer if layer is None else layer,
        expanded_row_idx=None,
        topk_ids=None,
        lora_context=None,
    )
