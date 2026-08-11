"""Expert paging at the RoutedExperts boundary used by current vLLM.

Quantization schemes consume the current layer weight view. Decode pages the
resident slots before mapping logical IDs; prefill uses a separate view of the
full expert pool, leaving the persistent decode parameters unchanged.
"""
from copy import copy

import torch

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager


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
        "w13_weight": "_prefill_w13", "w2_weight": "_prefill_w2",
        "w13_weight_scale": "_prefill_w13_scale", "w2_weight_scale": "_prefill_w2_scale",
        "w13_weight_scale_fp32": "_prefill_w13_scale_fp32",
        "w13_scale_bias": "_prefill_w13_scale_bias", "w2_scale_bias": "_prefill_w2_scale_bias",
    }
    for name, pool_name in weights.items():
        pool = getattr(mgr, pool_name, None)
        if pool:
            _set_view_tensor(view, name, pool[slot])
    view.ascend_expert_map = mgr._get_shard_expert_map() if layer.enable_multi_card else mgr._prefill_log2phy
    return view


def apply_offloaded_experts(layer, hidden_states, topk_weights, topk_ids, router_logits):
    mgr = ExpertOffloadManager.get_instance()
    comm = _EXTRA_CTX.moe_comm_method
    num_tokens = topk_ids.size(0)
    router = layer.router
    routing = dict(router_logits=router_logits,
                   renormalize=getattr(router, "renormalize", True),
                   scoring_func=getattr(router, "scoring_func", "softmax"),
                   e_score_correction_bias=getattr(router, "e_score_correction_bias", None),
                   routed_scaling_factor=getattr(router, "routed_scaling_factor", 1.0),
                   is_hash_routed=layer.tid2eid is not None)
    if layer.enable_multi_card:
        mgr.update_weights_multi_card(layer, topk_ids, layer.log2phy, topk_weights,
                                      hidden_states=hidden_states, mc2_mask=layer.ascend_mc2_mask, **routing)
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
            names = ("num_experts_local", "num_local_experts", "num_experts", "expert_ids_per_ep_rank", "local_expert_indices")
            saved = {name: getattr(dispatcher, name) for name in names if hasattr(dispatcher, name)}
            local_count = view.moe_config.num_local_experts
            for name in ("num_experts_local", "num_local_experts"):
                if name in saved:
                    setattr(dispatcher, name, local_count)
            if "num_experts" in saved:
                dispatcher.num_experts = mgr.num_total_experts
            if layer.enable_multi_card and "local_expert_indices" in saved:
                dispatcher.local_expert_indices = list(range(dispatcher.ep_rank * local_count, (dispatcher.ep_rank + 1) * local_count))
                if saved.get("expert_ids_per_ep_rank") is not None:
                    dispatcher.expert_ids_per_ep_rank = torch.arange(
                        mgr.num_total_experts, dtype=torch.int32,
                        device=saved["expert_ids_per_ep_rank"].device) % local_count
        elif layer.log2phy is not None:
            topk_ids = layer.log2phy[topk_ids].clamp(min=0)
        result = layer.quant_method.apply(
            layer=view, x=hidden_states, topk_weights=topk_weights, topk_ids=topk_ids,
            shared_experts=None, shared_experts_input=None,
        )
    finally:
        for name, value in saved.items():
            setattr(dispatcher, name, value)
    if not use_prefill and num_tokens <= mgr.offload_threshold:
        mgr.trigger_next_layer_prefetch(layer, hidden_states)
    return result
