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
    counters = getattr(mgr, "_benchmark_route_counters", None)
    counter_sample = None
    counter_layer = None
    if counters is not None and not layer.enable_multi_card and layer.log2phy is not None:
        counter_layer = mgr.moe_layers.index(layer)
        counter_sample = counters.begin(counter_layer, topk_ids, topk_weights,
                                        layer.log2phy, is_hash=layer.tid2eid is not None)
    hybrid_plan = None
    prefetch_staged = False
    if not layer.enable_multi_card:
        from vllm_ascend.expert_offload.hybrid_executor import begin_hybrid_routes

        # Phase A only: finishes the pending prefetch join and preprocesses
        # routes. The current layer's CPU D2H copies and ready-flag publish
        # are deferred to prepared.finish() so the next-layer prefetch signal
        # can be fired first.
        prepared = begin_hybrid_routes(
            layer,
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            log2phy=layer.log2phy,
            transfer_manager=mgr,
            router_logits=router_logits,
            scoring_func=routing["scoring_func"],
            e_score_correction_bias=routing["e_score_correction_bias"],
            is_hash_routed=routing["is_hash_routed"],
        )
        # The actual current-layer top-k is the authoritative observation for
        # the target layer's NPU-owned LRC table. This does not alter routing;
        # it only updates the device-side eviction state before next-layer
        # prediction is launched.
        if getattr(getattr(mgr.offload_config, "cpu_moe", None), "enabled", False):
            mgr.observe_device_prefetch_route(
                mgr.moe_layers.index(layer), topk_ids, topk_weights
            )
        if prepared is not None:
            # Stage next-layer prediction + selector + prefetch notify BEFORE
            # the current layer's CPU input copies and submit. hidden_states
            # is already available here; the prefetch chain no longer waits
            # behind the current layer's CPU-ready publish (~200us measured).
            mgr.trigger_next_layer_prefetch(layer, hidden_states)
            prefetch_staged = True
            hybrid_plan = prepared.finish()
    if hybrid_plan is not None:
        topk_ids = hybrid_plan.npu_topk_ids
        topk_weights = hybrid_plan.npu_topk_weights
        prefill_regime = False
    elif layer.enable_multi_card:
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
        # Prefill pool transfers are eager-only; preserve the hybrid-plan guard.
        prefill_regime = num_tokens > mgr.offload_threshold and not torch.npu.is_current_stream_capturing()
    use_prefill = (
        hybrid_plan is None
        and prefill_regime
        and mgr._prefill_initialized
        and (layer.enable_multi_card or not mgr._skip_prefill)
    )
    if counter_sample is not None:
        counter_weights = topk_weights
        if hybrid_plan is not None:
            counter_weights = hybrid_plan.npu_topk_weights + hybrid_plan.cpu_topk_weights
        counters.finish(counter_layer, counter_sample, topk_ids, counter_weights,
                        layer.log2phy, use_prefill=use_prefill)
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
        # Fallback trigger site for paths without a hybrid plan (multi-card /
        # paging / prefill regimes): stage next-layer prediction before the
        # current-layer GMM is submitted. The hybrid path already staged the
        # prefetch before its CPU submit above (prefetch_staged).
        if (not prefetch_staged and not use_prefill
                and (hybrid_plan is not None or num_tokens <= mgr.offload_threshold)):
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
    if hybrid_plan is not None:
        result.cpu_expert_task = hybrid_plan.cpu_task
        stats = getattr(mgr, "_stats", None)
        if stats is not None and hybrid_plan.stats is not None:
            stats.record_layer(layer_idx=mgr.moe_layers.index(layer), gsize=hybrid_plan.stats["cpu_experts"] + hybrid_plan.stats["npu_experts"], **hybrid_plan.stats)
    return result
