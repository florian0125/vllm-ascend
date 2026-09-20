"""Regression tests for offload at the current RoutedExperts boundary."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.expert_offload import routed_experts as offload


def setup_case(monkeypatch, *, multi_card=False, tokens=2, hash_routed=False):
    layer = torch.nn.Module()
    layer.moe_config = SimpleNamespace(num_local_experts=2, ep_size=2 if multi_card else 1)
    layer.enable_multi_card = multi_card
    layer.log2phy = torch.tensor([1, 0, -1, -1], dtype=torch.int32)
    layer.ascend_expert_map = torch.arange(4, dtype=torch.int32) if multi_card else None
    layer.ascend_mc2_mask = None
    layer.router = SimpleNamespace(scoring_func="sigmoid", renormalize=False, routed_scaling_factor=2.0)
    layer.tid2eid = torch.tensor([0]) if hash_routed else None
    for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
        layer.register_parameter(name, torch.nn.Parameter(torch.randn(2, 3, 4)))
    layer.quant_method = Mock()
    layer.quant_method.apply.return_value = object()
    mgr = SimpleNamespace(
        offload_threshold=2,
        _prefill_initialized=True,
        _skip_prefill=False,
        moe_layers=[layer],
        num_total_experts=4,
        mc_shard_size=2,
        _prefill_w13=[torch.randn(2 if multi_card else 4, 3, 4)],
        _prefill_w2=[torch.randn(2 if multi_card else 4, 3, 4)],
        _prefill_w13_scale=[torch.randn(2 if multi_card else 4, 3, 4)],
        _prefill_w2_scale=[torch.randn(2 if multi_card else 4, 3, 4)],
        _prefill_log2phy=torch.arange(4, dtype=torch.int32),
        _get_shard_expert_map=Mock(return_value=torch.tensor([-1, -1, 0, 1])),
        update_weights=Mock(),
        update_weights_multi_card=Mock(),
        trigger_next_layer_prefetch=Mock(),
    )
    dispatcher = SimpleNamespace(
        num_experts=4,
        num_local_experts=2,
        num_experts_local=2,
        ep_rank=1 if multi_card else 0,
        local_expert_indices=[0, 1],
        expert_ids_per_ep_rank=torch.tensor([0, 1, 0, 1]),
    )
    comm_type = MoECommType.ALLTOALL if multi_card and tokens > 2 else MoECommType.MC2
    monkeypatch.setattr(
        offload,
        "_EXTRA_CTX",
        SimpleNamespace(moe_comm_type=comm_type, moe_comm_method=SimpleNamespace(token_dispatcher=dispatcher)),
    )
    monkeypatch.setattr(offload.ExpertOffloadManager, "get_instance", lambda: mgr)
    x = torch.randn(tokens, 8)
    ids = torch.tensor([[0, 1]] * tokens, dtype=torch.int32)
    weights = torch.full((tokens, 2), 0.5)
    logits = torch.randn(tokens, 4)
    return layer, mgr, dispatcher, (x, weights, ids, logits)


@pytest.mark.parametrize("multi_card", [False, True])
@pytest.mark.parametrize("hash_routed", [False, True])
def test_decode_updates_logical_routes_then_maps_slots(monkeypatch, multi_card, hash_routed):
    layer, mgr, dispatcher, args = setup_case(monkeypatch, multi_card=multi_card, hash_routed=hash_routed)
    weights_before = args[1].clone()
    result = offload.apply_offloaded_experts(layer, *args)
    update = mgr.update_weights_multi_card if multi_card else mgr.update_weights
    other = mgr.update_weights if multi_card else mgr.update_weights_multi_card
    update.assert_called_once()
    other.assert_not_called()
    assert update.call_args.args[1] is args[2]
    assert update.call_args.kwargs["is_hash_routed"] is hash_routed
    assert update.call_args.kwargs["router_logits"] is args[3]
    assert update.call_args.kwargs["scoring_func"] == "sigmoid"
    assert update.call_args.kwargs["renormalize"] is False
    call = layer.quant_method.apply.call_args.kwargs
    assert call["topk_ids"].tolist() == [[1, 0], [1, 0]]
    assert torch.equal(args[1], weights_before)
    assert result is layer.quant_method.apply.return_value
    mgr.trigger_next_layer_prefetch.assert_called_once_with(layer, args[0])


@pytest.mark.parametrize("multi_card", [False, True])
@pytest.mark.parametrize("raise_in_kernel", [False, True])
def test_prefill_uses_pool_without_mutating_decode_state(monkeypatch, multi_card, raise_in_kernel):
    layer, mgr, dispatcher, args = setup_case(monkeypatch, multi_card=multi_card, tokens=4)
    original_weights = layer.w13_weight
    original_config = layer.moe_config
    original_dispatcher = vars(dispatcher).copy()
    captured = []

    def kernel(**kwargs):
        view = kwargs["layer"]
        captured.append(view)
        assert view is not layer
        assert view.w13_weight is mgr._prefill_w13[0]
        assert view.moe_config.num_local_experts == (2 if multi_card else 4)
        assert layer.w13_weight is original_weights
        assert layer.moe_config is original_config
        if multi_card:
            assert dispatcher.local_expert_indices == [2, 3]
        if not multi_card:
            assert view.ascend_expert_map is None
        if raise_in_kernel:
            raise RuntimeError("kernel failed")
        return 123

    layer.quant_method.apply.side_effect = kernel
    if raise_in_kernel:
        with pytest.raises(RuntimeError, match="kernel failed"):
            offload.apply_offloaded_experts(layer, *args)
    else:
        assert offload.apply_offloaded_experts(layer, *args) == 123
    assert captured
    assert layer.w13_weight is original_weights
    assert layer.moe_config.num_local_experts == 2
    for name, value in original_dispatcher.items():
        assert getattr(dispatcher, name) is value
    mgr.trigger_next_layer_prefetch.assert_not_called()


def test_pruned_route_is_not_mapped_to_last_expert(monkeypatch):
    layer, mgr, dispatcher, args = setup_case(monkeypatch)
    args[2][0, 1] = -1
    offload.apply_offloaded_experts(layer, *args)
    assert layer.quant_method.apply.call_args.kwargs["topk_ids"][0, 1] == 2


def test_prefetch_reads_activations_before_kernel_disposes_them(monkeypatch):
    layer, mgr, dispatcher, args = setup_case(monkeypatch)
    order = []
    mgr.trigger_next_layer_prefetch.side_effect = lambda *_: order.append("prefetch")
    layer.quant_method.apply.side_effect = lambda **_: order.append("kernel")
    offload.apply_offloaded_experts(layer, *args)
    assert order == ["prefetch", "kernel"]
