import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager
from vllm_ascend.expert_offload.h2d_transfer import H2DCopyTask, TorchCopyH2DTransport


def make_manager():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.offload_config = SimpleNamespace(shard_per_rank=False)
    manager.w13_expert_size_bytes = 4
    manager.w2_expert_size_bytes = 2
    manager.w13_weights_cpu = [[torch.arange(4, dtype=torch.uint8) + i for i in range(3)]]
    manager.w2_weights_cpu = [[torch.arange(2, dtype=torch.uint8) + i for i in range(3)]]
    manager.scale_cpu_buffers = {"w13_weight_scale": [[torch.tensor([i + 0.25, i + 0.5]) for i in range(3)]]}
    manager.offset_cpu_buffers = {}
    manager.scale_bias_cpu_buffers = {}
    layer = SimpleNamespace(w13_weight=torch.zeros((2, 4), dtype=torch.uint8),
                            w2_weight=torch.zeros((2, 2), dtype=torch.uint8),
                            w13_weight_scale=torch.zeros((2, 2)))
    return manager, layer


def test_source_and_destination_views_are_separate_and_bounded():
    manager, layer = make_manager()
    a = manager._build_expert_h2d_tasks(layer, 0, 0, 0)
    b = manager._build_expert_h2d_tasks(layer, 0, 1, 0)
    c = manager._build_expert_h2d_tasks(layer, 0, 0, 1)
    assert a[0].destination is b[0].destination
    assert a[0].source is c[0].source
    assert a[2].destination is b[2].destination
    assert a[2].source is c[2].source
    for eid in range(3):
        for slot in range(2):
            manager._build_expert_h2d_tasks(layer, 0, eid, slot)
    assert len(manager._expert_h2d_views) == (3 + 2) * 3


def test_cached_sources_read_inplace_updates_and_cross_slot_loads():
    manager, layer = make_manager()
    first = manager._build_expert_h2d_tasks(layer, 0, 0, 0)
    manager.w13_weights_cpu[0][0].fill_(37)
    manager.scale_cpu_buffers["w13_weight_scale"][0][0].fill_(2.5)
    TorchCopyH2DTransport().copy_batch(first)
    assert torch.equal(layer.w13_weight[0], torch.full((4,), 37, dtype=torch.uint8))
    assert torch.equal(layer.w13_weight_scale[0], torch.full((2,), 2.5))
    TorchCopyH2DTransport().copy_batch(manager._build_expert_h2d_tasks(layer, 0, 1, 0))
    TorchCopyH2DTransport().copy_batch(manager._build_expert_h2d_tasks(layer, 0, 0, 1))
    assert torch.equal(layer.w13_weight[0], manager.w13_weights_cpu[0][1])
    assert torch.equal(layer.w13_weight[1], manager.w13_weights_cpu[0][0])


def test_view_cache_invalidates_owner_and_same_tensor_storage_replacement():
    manager, layer = make_manager()
    first = manager._build_expert_h2d_tasks(layer, 0, 0, 0)
    layer.w13_weight.data = torch.zeros_like(layer.w13_weight)
    manager.w13_weights_cpu[0][0] = torch.full((4,), 19, dtype=torch.uint8)
    second = manager._build_expert_h2d_tasks(layer, 0, 0, 0)
    assert second[0].destination is not first[0].destination
    assert second[0].source is not first[0].source
    TorchCopyH2DTransport().copy_batch(second)
    assert torch.equal(layer.w13_weight[0], torch.full((4,), 19, dtype=torch.uint8))


def test_noncontiguous_quant_reshape_does_not_cache_a_stale_copy():
    manager, layer = make_manager()
    source = torch.arange(4, dtype=torch.float32).reshape(2, 2).t()
    manager.scale_cpu_buffers["w13_weight_scale"][0][0] = source
    layer.w13_weight_scale = torch.zeros((2, 4))
    first = manager._build_quant_attr_h2d_tasks(layer, 0, 0, 0)
    source.add_(10)
    second = manager._build_quant_attr_h2d_tasks(layer, 0, 0, 0)
    assert first[0].destination is second[0].destination
    assert first[0].source.data_ptr() != second[0].source.data_ptr()
    TorchCopyH2DTransport().copy_batch(second)
    assert torch.equal(layer.w13_weight_scale[0], source.reshape(4))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int8])
def test_scale_refresh_direct_copy_preserves_conversion(dtype):
    source = torch.arange(8).to(dtype).reshape(2, 4)
    layer = SimpleNamespace(w13_weight_scale=source, w13_weight_scale_fp32=torch.zeros((2, 4)))
    ExpertOffloadManager._refresh_expert_fp32_scale(layer, 1)
    assert torch.equal(layer.w13_weight_scale_fp32[1], source[1].to(torch.float32))
    assert not layer.w13_weight_scale_fp32[0].count_nonzero()


def test_quant_copy_preserves_original_field_operations():
    sources = [torch.arange(4, dtype=torch.float32) for _ in range(2)]
    destinations = [torch.zeros(4) for _ in range(2)]
    tasks = [H2DCopyTask(s, d, 16) for s, d in zip(sources, destinations)]
    original = torch._foreach_copy_
    with patch("torch._foreach_copy_", wraps=original) as foreach:
        TorchCopyH2DTransport().copy_batch(tasks)
    foreach.assert_not_called()
    assert all(torch.equal(s, d) for s, d in zip(sources, destinations))


def test_trace_and_strided_destination_keep_per_field_fallback():
    sources = [torch.ones(4), torch.ones(4)]
    destinations = [torch.zeros(4), torch.zeros(8)[::2]]
    tasks = [H2DCopyTask(s, d, 16) for s, d in zip(sources, destinations)]
    transport = TorchCopyH2DTransport()
    with patch("torch._foreach_copy_") as foreach:
        transport.copy_batch(tasks)
    foreach.assert_not_called()
    trace = SimpleNamespace(_local=threading.local(), mark=MagicMock())
    trace._local.context = (1, 0, 0)
    transport._cpu_prefetch_trace = trace
    with patch("torch._foreach_copy_") as foreach:
        transport.copy_batch(tasks)
    foreach.assert_not_called()
    assert trace.mark.call_count == 4


def test_replaced_views_retire_after_all_related_h2d_streams_drain():
    manager, layer = make_manager()
    first = manager._build_expert_h2d_tasks(layer, 0, 0, 0)
    manager.w13_weights_cpu[0][0] = torch.full((4,), 91, dtype=torch.uint8)
    manager._build_expert_h2d_tasks(layer, 0, 0, 0)
    assert any(entry[4] is first[0].source for entry in manager._retired_expert_h2d_views)
    concurrent_retirement = object()
    manager.load_stream = SimpleNamespace(synchronize=MagicMock(
        side_effect=lambda: manager._retired_expert_h2d_views.append(concurrent_retirement)))
    manager._cpu_prefetch_dma_stream = SimpleNamespace(synchronize=MagicMock())
    manager._prefetch_stream = SimpleNamespace(synchronize=MagicMock(
        side_effect=AssertionError("handshake stream waits for this callback's done")))
    manager.h2d_transport = SimpleNamespace(
        synchronize=MagicMock(side_effect=lambda stream: stream.synchronize()), close=MagicMock())
    manager._synchronize_h2d(manager._cpu_prefetch_dma_stream)
    assert manager.h2d_transport.synchronize.call_count == 2
    manager.load_stream.synchronize.assert_called_once()
    manager._cpu_prefetch_dma_stream.synchronize.assert_called_once()
    manager._prefetch_stream.synchronize.assert_not_called()
    assert manager._retired_expert_h2d_views == [concurrent_retirement]
    manager.close()
    assert not manager._expert_h2d_views
    assert not manager._retired_expert_h2d_views


def test_prefetch_reads_current_ids_once_and_rebuilds_current_mapping():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.offload_config = SimpleNamespace(experts_pruning_enabled=False)
    manager._stats = None
    manager._debug = False
    manager.cache_policy = None
    manager.num_total_experts = 4
    manager.prefetch_topk = 1
    manager._prefetch_trace = SimpleNamespace(mark=MagicMock())
    manager._cpu_prefetch_dispatcher = None
    manager.load_stream = object()
    manager._load_expert_weights_into_slots = MagicMock()
    manager._synchronize_h2d = MagicMock()
    ids = SimpleNamespace(tolist=MagicMock(return_value=[[2]]))
    mapping = np.array([0, 1, -1, -1], dtype=np.int32)
    with patch("vllm_ascend.expert_offload.expert_offload_manager.torch_npu.npu.stream", return_value=nullcontext()):
        manager._update_weights((ids, mapping, object(), 0, None, True))
    ids.tolist.assert_called_once()
    assert mapping.tolist() == [0, -1, 1, -1]
