import os
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from vllm_ascend.expert_offload.h2d_transfer import (
    CopyDirection,
    H2DCopyTask,
    MemFabricLocalH2DTransport,
    MemFabricSharedH2DTransport,
    ONE_GIB,
    TorchCopyH2DTransport,
    TorchSharedCPUH2DTransport,
    TorchSharedCPUWeightPool,
    create_h2d_transport,
)


def test_torch_transport_preserves_task_order_and_non_blocking_flags():
    first_destination = MagicMock()
    second_destination = MagicMock()
    tasks = [
        H2DCopyTask(
            source="w13-source",
            destination=first_destination,
            nbytes=16,
            name="w13",
        ),
        H2DCopyTask(
            source="scale-source",
            destination=second_destination,
            nbytes=4,
            name="scale",
            non_blocking=False,
        ),
    ]

    TorchCopyH2DTransport().copy_batch(tasks)

    assert first_destination.copy_.call_args_list == [
        call("w13-source", non_blocking=True)]
    assert second_destination.copy_.call_args_list == [
        call("scale-source", non_blocking=False)]


def test_torch_transport_accepts_an_empty_batch():
    TorchCopyH2DTransport().copy_batch([])


def test_torch_transport_accepts_all_expert_copy_directions():
    destinations = [MagicMock(), MagicMock(), MagicMock()]
    tasks = [
        H2DCopyTask(
            source=f"source-{direction.value}",
            destination=destination,
            nbytes=4,
            direction=direction,
        )
        for direction, destination in zip(CopyDirection, destinations)
    ]

    TorchCopyH2DTransport().copy_batch(tasks)

    for direction, destination in zip(CopyDirection, destinations):
        destination.copy_.assert_called_once_with(
            f"source-{direction.value}", non_blocking=True)


def test_torch_transport_allocates_pinned_host_tensor():
    expected = object()
    with patch(
        "vllm_ascend.expert_offload.h2d_transfer.torch.empty",
        return_value=expected,
    ) as empty:
        result = TorchCopyH2DTransport().allocate_host_tensor((2, 3), "dtype")

    assert result is expected
    empty.assert_called_once_with(
        (2, 3), dtype="dtype", device="cpu", pin_memory=True)


def test_torch_shared_transport_stages_until_sync_then_reuses_buffer():
    transport = TorchSharedCPUH2DTransport()
    source = MagicMock()
    destination = MagicMock()
    staging = MagicMock()
    staging_storage = MagicMock()
    staging.untyped_storage.return_value = staging_storage
    task = H2DCopyTask(source, destination, 16)

    with patch(
        "vllm_ascend.expert_offload.h2d_transfer.torch.empty",
        return_value=staging,
    ) as empty:
        transport.copy_batch([task])
        assert transport._inflight_staging[0][1] is staging

        stream = MagicMock()
        transport.synchronize(stream)
        assert transport._inflight_staging == []

        transport.copy_batch([task])

    empty.assert_called_once_with(
        (16,), dtype=torch.uint8, device="cpu", pin_memory=True)
    assert staging_storage.copy_.call_args_list == [call(source), call(source)]
    assert destination.copy_.call_args_list == [
        call(staging_storage, non_blocking=True),
        call(staging_storage, non_blocking=True),
    ]
    assert transport._inflight_staging[0][1] is staging

    stream.synchronize.assert_called_once_with()


class _FakeCollectives:

    def __init__(self, world_size):
        self.world_size = world_size
        self.barriers = 0

    def all_gather_object(self, output, value, group):
        del group
        for rank in range(self.world_size):
            output[rank] = value

    def barrier(self, group):
        del group
        self.barriers += 1


def test_torch_shared_weight_pool_maps_and_unlinks_backing_file(tmp_path):
    collectives = _FakeCollectives(world_size=2)
    scalar = MagicMock()
    scalar.element_size.return_value = 4
    mapped = MagicMock()
    expected = object()
    mapped.reshape.return_value = expected
    with (
        patch(
            "vllm_ascend.expert_offload.h2d_transfer.torch.empty",
            return_value=scalar,
        ),
        patch(
            "vllm_ascend.expert_offload.h2d_transfer.torch.from_file",
            return_value=mapped,
        ) as from_file,
    ):
        pool = TorchSharedCPUWeightPool(
            str(tmp_path), 2, 0, object(), dist_module=collectives)
        result = pool.allocate("layer/0:w13", (2, 3), "dtype")

    assert result is expected
    path = from_file.call_args.args[0]
    from_file.assert_called_once_with(
        path, shared=True, size=6, dtype="dtype")
    assert not os.path.exists(path)
    assert collectives.barriers == 1
    run_dir = pool.run_dir

    pool.close()

    assert not os.path.exists(run_dir)


def test_factory_selects_torch_shared_staging_transport():
    transport = create_h2d_transport(
        "torch",
        device_id=0,
        memfabric_pool_size_gib=0,
        memfabric_log_level=3,
        enable_torch_shared_cpu=True,
    )

    assert isinstance(transport, TorchSharedCPUH2DTransport)


class _FakeDescriptor:

    def __init__(self, values, dtype, device):
        self.values = values
        self.dtype = dtype
        self.device = device

    def data_ptr(self):
        return id(self)


def _make_memfabric_transport(pool_size_gib=2):
    mf_module = MagicMock()
    offload_module = MagicMock()
    offload_module.OffloadConfig.side_effect = SimpleNamespace
    offload_module.initialize.return_value = 0
    offload_module.sparse_copy.return_value = 0
    offload_module.uninitialize.return_value = 0
    with (
        patch(
            "vllm_ascend.expert_offload.h2d_transfer.torch.device",
            return_value="npu:fake",
        ),
        patch(
            "vllm_ascend.expert_offload.h2d_transfer.atexit.register",
        ),
    ):
        transport = MemFabricLocalH2DTransport(
            device_id=3,
            pool_size_gib=pool_size_gib,
            log_level=1,
            mf_module=mf_module,
            offload_module=offload_module,
        )
    return transport, mf_module, offload_module


def test_memfabric_transport_initializes_local_pool_and_allocates_from_it():
    transport, mf_module, offload_module = _make_memfabric_transport()
    expected = object()
    offload_module.empty.return_value = expected

    result = transport.allocate_host_tensor((4, 8), "dtype")

    mf_module.set_log_level.assert_called_once_with(1)
    config = offload_module.initialize.call_args.args[0]
    assert config.device_id == 3
    assert config.reserve_size == 2 * ONE_GIB
    assert config.alloc_size == 2 * ONE_GIB
    assert result is expected
    offload_module.empty.assert_called_once_with((4, 8), dtype="dtype")


def test_memfabric_shared_transport_initializes_one_contribution_per_rank():
    mf_module = MagicMock()
    offload_module = MagicMock()
    offload_module.OffloadConfig.side_effect = SimpleNamespace
    offload_module.Scene.SHARED = "shared"
    offload_module.initialize.return_value = 0
    with (
        patch(
            "vllm_ascend.expert_offload.h2d_transfer.torch.device",
            return_value="npu:fake",
        ),
        patch(
            "vllm_ascend.expert_offload.h2d_transfer.atexit.register",
        ),
    ):
        transport = MemFabricSharedH2DTransport(
            device_id=2,
            pool_size_gib=4,
            world_size=8,
            rank_id=2,
            log_level=1,
            mf_module=mf_module,
            offload_module=offload_module,
        )

    config = offload_module.initialize.call_args.args[0]
    assert config.device_id == 2
    assert config.reserve_size == 4 * ONE_GIB
    assert config.alloc_size == 4 * ONE_GIB
    assert config.world_size == 8
    assert config.rank_id == 2
    assert config.scene == "shared"
    assert transport.supports_remote_sources is True


def test_memfabric_shared_transport_validates_rank_topology():
    with pytest.raises(ValueError, match="world_size"):
        MemFabricSharedH2DTransport(0, 1, 1, 0)
    with pytest.raises(ValueError, match="Invalid.*rank"):
        MemFabricSharedH2DTransport(0, 1, 4, 4)


def test_memfabric_transport_pads_odd_sparse_copy_batch_and_retires_descriptors():
    transport, _, offload_module = _make_memfabric_transport()
    sources = [MagicMock(), MagicMock(), MagicMock()]
    destinations = [MagicMock(), MagicMock(), MagicMock()]
    for index, source in enumerate(sources):
        source.data_ptr.return_value = 100 + index
    for index, destination in enumerate(destinations):
        destination.data_ptr.return_value = 200 + index
    tasks = [
        H2DCopyTask(source, destination, index + 1)
        for index, (source, destination) in enumerate(
            zip(sources, destinations))
    ]
    descriptors = []

    def make_descriptor(values, dtype, device):
        descriptor = _FakeDescriptor(values, dtype, device)
        descriptors.append(descriptor)
        return descriptor

    with patch(
        "vllm_ascend.expert_offload.h2d_transfer.torch.tensor",
        side_effect=make_descriptor,
    ):
        transport.copy_batch(tasks)

    assert descriptors[0].values == [100, 101, 102, 102]
    assert descriptors[1].values == [200, 201, 202, 202]
    assert descriptors[2].values == [1, 2, 3, 0]
    assert descriptors[3].values == 4
    offload_module.sparse_copy.assert_called_once_with(
        *descriptors, "npu:fake")
    assert len(transport._inflight_descriptors) == 1

    stream = MagicMock()
    transport.synchronize(stream)

    stream.synchronize.assert_called_once_with()
    assert transport._inflight_descriptors == []


def test_memfabric_transport_preserves_uint32_length_bits():
    assert MemFabricLocalH2DTransport._int32_bits((1 << 31) - 1) == \
        (1 << 31) - 1
    assert MemFabricLocalH2DTransport._int32_bits(1 << 31) == -(1 << 31)
    assert MemFabricLocalH2DTransport._int32_bits((1 << 32) - 1) == -1
    with pytest.raises(ValueError, match="exceeds uint32"):
        MemFabricLocalH2DTransport._int32_bits(1 << 32)


def test_memfabric_transport_rejects_non_h2d_copy_directions():
    transport, _, offload_module = _make_memfabric_transport()
    task = H2DCopyTask(
        MagicMock(), MagicMock(), 4,
        name="victim-d2h",
        direction=CopyDirection.D2H,
    )

    with pytest.raises(RuntimeError, match="supports H2D only.*d2h"):
        transport.copy_batch([task])

    offload_module.sparse_copy.assert_not_called()


def test_memfabric_transport_reports_sparse_copy_failure():
    transport, _, offload_module = _make_memfabric_transport()
    offload_module.sparse_copy.return_value = 9
    task = H2DCopyTask(MagicMock(), MagicMock(), 4)

    with (
        patch(
            "vllm_ascend.expert_offload.h2d_transfer.torch.tensor",
            side_effect=lambda values, dtype, device: _FakeDescriptor(
                values, dtype, device),
        ),
        pytest.raises(RuntimeError, match="sparse_copy failed.*ret=9"),
    ):
        transport.copy_batch([task])

    assert transport._inflight_descriptors == []


def test_memfabric_transport_close_is_idempotent():
    transport, _, offload_module = _make_memfabric_transport()

    with patch(
        "vllm_ascend.expert_offload.h2d_transfer.atexit.unregister",
    ):
        transport.close()
        transport.close()

    offload_module.uninitialize.assert_called_once_with()
    with pytest.raises(RuntimeError, match="already closed"):
        transport.allocate_host_tensor((1,), "dtype")


def test_memfabric_transport_reports_initialization_failure():
    mf_module = MagicMock()
    offload_module = MagicMock()
    offload_module.OffloadConfig.side_effect = SimpleNamespace
    offload_module.initialize.return_value = 7

    with (
        patch(
            "vllm_ascend.expert_offload.h2d_transfer.torch.device",
            return_value="npu:fake",
        ),
        pytest.raises(RuntimeError, match="initialization failed.*ret=7"),
    ):
        MemFabricLocalH2DTransport(
            device_id=0,
            pool_size_gib=1,
            mf_module=mf_module,
            offload_module=offload_module,
        )


def test_h2d_transport_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported"):
        create_h2d_transport(
            "unknown",
            device_id=0,
            memfabric_pool_size_gib=0,
            memfabric_log_level=3,
        )


@pytest.mark.parametrize("nbytes", [0, -1])
def test_h2d_copy_task_requires_positive_size(nbytes):
    with pytest.raises(ValueError, match="copy size must be positive"):
        H2DCopyTask(source=object(), destination=object(), nbytes=nbytes)
