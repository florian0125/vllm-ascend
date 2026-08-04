import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from vllm_ascend.ascend_config import ExpertOffloadConfig
from vllm_ascend.expert_offload.expert_offload_manager import (
    ExpertOffloadManager,
    _expert_weight,
)


def _manager_for_prediction(next_layer, gate_weight, *, topk=2,
                            prefetch_topk=2):
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.moe_layers = [SimpleNamespace(), next_layer]
    manager._gate_weights_npu = [None, gate_weight]
    manager.topk = topk
    manager.prefetch_topk = prefetch_topk
    return manager


def test_expert_weight_supports_deepseek_and_kimi_k3_layouts():
    deepseek_weight = object()
    kimi_weight = object()
    deepseek_layer = SimpleNamespace(w13_weight=deepseek_weight)
    kimi_layer = SimpleNamespace(
        experts=SimpleNamespace(w13_weight=kimi_weight))

    assert _expert_weight(deepseek_layer, "w13_weight") is deepseek_weight
    assert _expert_weight(kimi_layer, "w13_weight") is kimi_weight


def test_gate_registration_is_model_agnostic_and_index_aligned():
    deepseek_weight = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
    kimi_weight = torch.tensor([[3.0, 4.0]], dtype=torch.float16)
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.moe_layers = [
        SimpleNamespace(gate=SimpleNamespace(weight=deepseek_weight)),
        SimpleNamespace(gate=SimpleNamespace(weight=kimi_weight)),
        SimpleNamespace(),
    ]
    manager._gate_weights_npu = [torch.zeros(1)]

    manager.register_gate_weights(None)

    assert len(manager._gate_weights_npu) == 3
    assert torch.equal(manager._gate_weights_npu[0], deepseek_weight.float())
    assert torch.equal(manager._gate_weights_npu[1], kimi_weight.float())
    assert manager._gate_weights_npu[2] is None


def test_post_finalize_gate_registration_keeps_missing_gate_slot():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.offload_config = ExpertOffloadConfig({
        "expert_prefetch_enabled": True
    })
    manager.topk_ids_h = object()
    manager._gate_weights_npu = []
    manager.moe_layers = [SimpleNamespace()]

    manager._register_layer_gate(manager.moe_layers[0])

    assert manager._gate_weights_npu == [None]


def test_deepseek_hash_router_prefetch_uses_tid2eid():
    tid2eid = torch.tensor([[0, 1], [2, 3], [3, 1]], dtype=torch.int32)
    next_layer = SimpleNamespace(gate=SimpleNamespace(tid2eid=tid2eid))
    manager = _manager_for_prediction(
        next_layer, torch.ones((4, 3)), topk=2, prefetch_topk=1)

    with patch(
        "vllm_ascend.expert_offload.expert_offload_manager.get_forward_context",
        return_value=SimpleNamespace(input_ids=torch.tensor([2])),
    ):
        weights, expert_ids = manager.predict_next_layer_experts_npu(
            0, torch.tensor([[100.0, -100.0, 0.0]]))

    assert torch.equal(expert_ids, tid2eid[2:3])
    assert torch.equal(weights, torch.full((1, 2), 0.5))


def test_kimi_k3_learned_router_prefetch_uses_gate_logits():
    next_layer = SimpleNamespace(gate=SimpleNamespace())
    gate_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    manager = _manager_for_prediction(
        next_layer, gate_weight, topk=2, prefetch_topk=2)

    weights, expert_ids = manager.predict_next_layer_experts_npu(
        0, torch.tensor([[1.0, 2.0]]))

    expected_weights = torch.softmax(torch.tensor([[1.0, 2.0, 3.0]]),
                                     dim=-1).topk(2, dim=-1).values
    assert torch.equal(expert_ids, torch.tensor([[2, 1]]))
    assert torch.allclose(weights, expected_weights)


def test_expert_substitution_log_is_controlled_by_debug():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.offload_config = ExpertOffloadConfig({
        "expert_substitution_enabled": True,
        "expert_substitution_threshold": 0.25,
    })
    original_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    substituted_ids = torch.tensor([[1, 5], [6, 4]], dtype=torch.int32)

    with patch(
        "vllm_ascend.expert_offload.expert_offload_manager.logger.info"
    ) as log_info:
        manager._debug = False
        manager._log_expert_substitution(
            7, original_ids, substituted_ids)
        log_info.assert_not_called()

        manager._debug = True
        manager._log_expert_substitution(
            7, original_ids, substituted_ids)

    log_args = log_info.call_args.args
    assert log_args[1:4] == (7, 2, 0.25)
    assert log_args[4] == [
        {"token": 0, "position": 1, "original": 2, "substitute": 5},
        {"token": 1, "position": 0, "original": 3, "substitute": 6},
    ]


def test_multi_card_substitution_updates_only_active_rows():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.offload_config = ExpertOffloadConfig({
        "expert_substitution_enabled": True,
        "expert_substitution_threshold": 0.30,
    })
    manager.topk = 2
    manager._debug = False
    router_logits_h = torch.log(torch.tensor([
        [0.40, 0.25, 0.20, 0.15],
        [0.40, 0.25, 0.20, 0.15],
    ]))
    topk_ids_h = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
    topk_weights_h = torch.tensor([[0.40, 0.25], [0.40, 0.25]])
    # The full global placement says experts 0 and 3 are resident across EP.
    log2phy_h = torch.tensor([0, -1, -1, 3], dtype=torch.int32)
    mc2_mask_h = torch.tensor([1, 0], dtype=torch.int32)

    manager._apply_multi_card_substitution(
        5,
        topk_ids_h,
        topk_weights_h,
        router_logits_h,
        log2phy_h,
        mc2_mask_h,
        False,
        "softmax",
        None,
        1.0,
    )

    assert topk_ids_h.tolist() == [[0, 3], [0, 1]]
    torch.testing.assert_close(
        topk_weights_h,
        torch.tensor([[0.40, 0.15], [0.40, 0.25]]),
    )


def test_prefetch_dispatch_uses_single_or_per_layer_multi_card_capacity():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.offload_config = ExpertOffloadConfig(
        {"num_device_experts": [8, 12]})
    manager._ep_size = 2
    manager._ep_info_resolved = True
    manager._update_weights = MagicMock()
    manager._update_weights_multi_card = MagicMock()
    inputs = (object(), object(), object(), object(), object(), 1)

    manager.enable_multi_card = False
    single_fn, single_args = manager._build_prefetch_call(*inputs)
    assert single_fn is manager._update_weights
    assert single_args[-1] is True

    manager.enable_multi_card = True
    multi_fn, multi_args = manager._build_prefetch_call(*inputs)
    assert multi_fn is manager._update_weights_multi_card
    assert multi_args[4] == 6
    assert multi_args[-2:] == (True, None)


def test_shared_expert_copy_is_non_blocking_and_refreshes_quant_scale():
    class RecordingStorage:
        def __init__(self):
            self.slice = None
            self.copies = []

        def __getitem__(self, storage_slice):
            self.slice = storage_slice
            return self

        def copy_(self, source, non_blocking=False):
            self.copies.append((self.slice, source, non_blocking))

    class Weight:
        def __init__(self):
            self.data = self
            self.storage = RecordingStorage()

        def untyped_storage(self):
            return self.storage

    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.w13_expert_size_bytes = 4
    manager.w2_expert_size_bytes = 2
    manager._expert_src_storage = MagicMock(
        side_effect=["w13-source", "w2-source"])
    manager._copy_quant_attrs_into_slot = MagicMock()
    layer = SimpleNamespace(
        w13_weight=Weight(),
        w2_weight=Weight(),
        w13_weight_scale=SimpleNamespace(data=torch.tensor([1.25, 2.5])),
        w13_weight_scale_fp32=torch.zeros(2),
    )

    manager._load_expert_weights_into_slot(layer, 0, 7, 1)

    assert layer.w13_weight.storage.copies == [
        (slice(4, 8), "w13-source", True)]
    assert layer.w2_weight.storage.copies == [
        (slice(2, 4), "w2-source", True)]
    manager._copy_quant_attrs_into_slot.assert_called_once_with(
        layer, 0, 7, 1)
    assert layer.w13_weight_scale_fp32[1] == 2.5


def test_single_card_hot_preload_uses_format_aware_shared_copy(tmp_path):
    ranking_path = tmp_path / "hot.json"
    ranking_path.write_text(json.dumps({"0": [[3, 0.9], [1, 0.8],
                                                   [2, 0.7]]}))
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.offload_config = ExpertOffloadConfig({
        "num_device_experts": 2,
        "hot_expert_preload": True,
        "hot_experts_file": str(ranking_path),
    })
    manager.enable_multi_card = False
    manager.cache_policy = None
    manager._debug = False
    manager.load_stream = MagicMock()
    manager.log2phy_h = torch.full((4,), -1, dtype=torch.int32)
    manager.log2phy_np = manager.log2phy_h.numpy()
    layer = SimpleNamespace(log2phy=MagicMock())
    manager.moe_layers = [layer]
    manager._load_expert_weights_into_slot = MagicMock()

    with patch(
        "vllm_ascend.expert_offload.expert_offload_manager.torch_npu.npu.stream",
        return_value=nullcontext(),
    ):
        manager._preload_hot_experts()

    assert manager._load_expert_weights_into_slot.call_args_list == [
        call(layer, 0, 3, 0),
        call(layer, 0, 1, 1),
    ]
    assert manager.log2phy_h.tolist() == [-1, 1, -1, 0]
    layer.log2phy.copy_.assert_called_once_with(manager.log2phy_h)


def test_multi_card_hot_preload_loads_only_rank_owned_experts(tmp_path):
    ranking_path = tmp_path / "hot.json"
    ranking_path.write_text(json.dumps({
        "0": [[0, 0.9], [1, 0.8], [2, 0.7], [3, 0.6],
              [4, 0.5], [5, 0.4], [6, 0.3], [7, 0.2]],
    }))
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.offload_config = ExpertOffloadConfig({
        "num_device_experts": 4,
        "hot_expert_preload": True,
        "hot_experts_file": str(ranking_path),
        "shard_per_rank": True,
    })
    manager.enable_multi_card = True
    manager._ep_size = 2
    manager._ep_rank = 1
    manager._ep_info_resolved = True
    manager._shard_size = 4
    manager.cache_policy = MagicMock()
    manager._mc_lrc = None
    manager._mc_prev_log2phy = {}
    manager._mc_resident = {}
    manager._debug = False
    manager.load_stream = MagicMock()
    manager.num_total_experts = 8
    manager.log2phy_h = torch.full((8,), -1, dtype=torch.int32)
    manager.log2phy_np = manager.log2phy_h.numpy()
    layer = SimpleNamespace(log2phy=MagicMock())
    manager.moe_layers = [layer]
    manager._load_expert_weights_into_slot = MagicMock()

    with patch(
        "vllm_ascend.expert_offload.expert_offload_manager.torch_npu.npu.stream",
        return_value=nullcontext(),
    ):
        manager._preload_hot_experts()

    assert manager._load_expert_weights_into_slot.call_args_list == [
        call(layer, 0, 4, 0),
        call(layer, 0, 5, 1),
    ]
    assert manager.log2phy_h.tolist() == [0, 1, -1, -1, 2, 3, -1, -1]
    assert manager._mc_resident[0] == {0: 4, 1: 5}
    assert torch.equal(manager._mc_prev_log2phy[0], manager.log2phy_h)
    assert manager._mc_lrc is manager.cache_policy
    manager.cache_policy.seed_layer_hotness.assert_called_once_with(
        0, {0: 0.9, 1: 0.8, 4: 0.5, 5: 0.4})
    layer.log2phy.copy_.assert_called_once_with(manager.log2phy_h)
