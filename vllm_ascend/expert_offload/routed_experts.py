"""Expert paging at the RoutedExperts boundary used by current vLLM.

Quantization schemes consume the current layer weight view. Decode pages the
resident slots before mapping logical IDs; prefill uses a separate view of the
full expert pool, leaving the persistent decode parameters unchanged.
"""

from copy import copy

import torch

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager
from vllm_ascend.ops.fused_moe.moe_comm_method import _map_logical_ids_to_phy


def _build_prefill_split_ctx(mgr, layer, layer_idx: int):
    """Stage the split-GMM context: decode-slot weights per quant scheme.

    The weight conventions mirror what each quant method itself feeds the
    MLP stage (list-wrapped for the int8 dynamic schemes, raw tensors for
    mxfp4/unquantized).  ``decode_layer`` carries the ORIGINAL layer: the
    main0928 quant methods read weights from ``mlp_compute_input.layer``.
    Returns None for schemes without a known convention (split disabled).
    """
    from vllm_ascend.expert_offload.prefill_split import PrefillSplitCtx
    from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights
    from vllm_ascend.quantization.quant_type import QuantType

    qt = getattr(layer.quant_method, "quant_type", QuantType.NONE)
    if qt == QuantType.W4A8MXFP:
        weights = MoEWeights(
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
        )
        num_slots = layer.w13_weight.shape[0]
    elif qt == QuantType.W4A8:
        weights = MoEWeights(
            w1=[layer.w13_weight],
            w2=[layer.w2_weight],
            w1_scale=[layer.w13_weight_scale],
            w2_scale=[layer.w2_weight_scale],
            w1_scale_bias=[layer.w13_scale_bias.detach()] if hasattr(layer, "w13_scale_bias") else None,
            w2_scale_bias=[layer.w2_scale_bias.detach()] if hasattr(layer, "w2_scale_bias") else None,
        )
        num_slots = layer.w13_weight.shape[0]
    elif qt == QuantType.W8A8:
        weights = MoEWeights(
            w1=[layer.w13_weight],
            w2=[layer.w2_weight],
            w1_scale=[layer.w13_weight_scale_fp32],
            w2_scale=[layer.w2_weight_scale],
        )
        num_slots = layer.w13_weight.shape[0]
    elif qt == QuantType.NONE:
        weights = MoEWeights(
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
        )
        num_slots = layer.w13_weight.shape[0]
    else:
        return None
    return PrefillSplitCtx(
        resident_slots=mgr.get_prefill_resident_slots(layer_idx),
        decode_weights=weights,
        decode_layer=layer,
        num_slots=num_slots,
        layer_idx=layer_idx,
    )


def _set_view_tensor(view, name, tensor):
    # A shallow Module view must not mutate the original parameter dictionary.
    view._parameters.pop(name, None)
    view._buffers.pop(name, None)
    object.__setattr__(view, name, tensor)


def _prefill_view(layer, mgr, slot):
    view = copy(layer)
    view._parameters = layer._parameters.copy()
    view._buffers = layer._buffers.copy()
    view.moe_config = copy(layer.moe_config)
    view.moe_config.num_local_experts = mgr.mc_shard_size if layer.enable_multi_card else mgr.num_total_experts
    weights = {
        "w13_weight": "_prefill_w13",
        "w2_weight": "_prefill_w2",
        "w13_weight_scale": "_prefill_w13_scale",
        "w2_weight_scale": "_prefill_w2_scale",
        "w13_weight_scale_fp32": "_prefill_w13_scale_fp32",
        "w13_scale_bias": "_prefill_w13_scale_bias",
        "w2_scale_bias": "_prefill_w2_scale_bias",
    }
    for name, pool_name in weights.items():
        pool = getattr(mgr, pool_name, None)
        if pool:
            _set_view_tensor(view, name, pool[slot])
    view.ascend_expert_map = mgr._get_shard_expert_map() if layer.enable_multi_card else None
    return view


def apply_offloaded_experts(layer, hidden_states, topk_weights, topk_ids, router_logits):
    mgr = ExpertOffloadManager.get_instance()
    comm = _EXTRA_CTX.moe_comm_method
    num_tokens = topk_ids.size(0)
    router = layer.router
    routing = dict(
        router_logits=router_logits,
        renormalize=getattr(router, "renormalize", True),
        scoring_func=getattr(router, "scoring_func", "softmax"),
        e_score_correction_bias=getattr(router, "e_score_correction_bias", None),
        routed_scaling_factor=getattr(router, "routed_scaling_factor", 1.0),
        is_hash_routed=layer.tid2eid is not None,
    )
    if layer.enable_multi_card:
        mgr.update_weights_multi_card(
            layer,
            topk_ids,
            layer.log2phy,
            topk_weights,
            hidden_states=hidden_states,
            mc2_mask=layer.ascend_mc2_mask,
            **routing,
        )
        prefill_regime = _EXTRA_CTX.moe_comm_type != MoECommType.MC2
    else:
        mgr.update_weights(layer, topk_ids, layer.log2phy, topk_weights, hidden_states=hidden_states, **routing)
        prefill_regime = num_tokens > mgr.offload_threshold
    use_prefill = prefill_regime and mgr._prefill_initialized and (layer.enable_multi_card or not mgr._skip_prefill)
    dispatcher = comm.token_dispatcher
    saved = {}
    view = layer
    try:
        if use_prefill:
            slot = mgr.moe_layers.index(layer) % len(mgr._prefill_w13)
            view = _prefill_view(layer, mgr, slot)
            names = (
                "num_experts_local",
                "num_local_experts",
                "num_experts",
                "expert_ids_per_ep_rank",
                "local_expert_indices",
            )
            saved = {name: getattr(dispatcher, name) for name in names if hasattr(dispatcher, name)}
            local_count = view.moe_config.num_local_experts
            for name in ("num_experts_local", "num_local_experts"):
                if name in saved:
                    setattr(dispatcher, name, local_count)
            if "num_experts" in saved:
                dispatcher.num_experts = mgr.num_total_experts
            if layer.enable_multi_card and "local_expert_indices" in saved:
                dispatcher.local_expert_indices = list(
                    range(dispatcher.ep_rank * local_count, (dispatcher.ep_rank + 1) * local_count)
                )
                if saved.get("expert_ids_per_ep_rank") is not None:
                    dispatcher.expert_ids_per_ep_rank = (
                        torch.arange(
                            mgr.num_total_experts, dtype=torch.int32, device=saved["expert_ids_per_ep_rank"].device
                        )
                        % local_count
                    )
        elif layer.log2phy is not None:
            topk_ids = _map_logical_ids_to_phy(
                topk_ids, layer.log2phy, layer.moe_config.num_local_experts * layer.moe_config.ep_size
            )
        if hidden_states.dtype != torch.float8_e4m3fn:
            topk_weights = topk_weights.to(hidden_states.dtype)
        if not use_prefill and num_tokens <= mgr.offload_threshold:
            mgr.trigger_next_layer_prefetch(layer, hidden_states)
        _split_ctx = None
        if use_prefill and mgr.offload_config.prefill_split_gmm is True:
            layer_idx = mgr.moe_layers.index(layer)
            _split_ctx = _build_prefill_split_ctx(mgr, layer, layer_idx)
        if _split_ctx is not None:
            _EXTRA_CTX.expert_offload_prefill_split = _split_ctx
        try:
            result = layer.quant_method.apply(
                layer=view,
                x=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts=None,
                shared_experts_input=None,
            )
        finally:
            if _split_ctx is not None:
                _EXTRA_CTX.expert_offload_prefill_split = None
        if use_prefill:
            # Prefill load pipeline: publish the staged next-layer residency
            # and kick the next load post-GMM-submit so it overlaps the GMM.
            mgr.on_prefill_layer_done(layer)
    finally:
        for name, value in saved.items():
            setattr(dispatcher, name, value)
    return result
