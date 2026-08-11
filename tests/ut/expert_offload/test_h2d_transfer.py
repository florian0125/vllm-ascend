from unittest.mock import MagicMock, call

import pytest

from vllm_ascend.expert_offload.h2d_transfer import (
    H2DCopyTask,
    TorchCopyH2DTransport,
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


@pytest.mark.parametrize("nbytes", [0, -1])
def test_h2d_copy_task_requires_positive_size(nbytes):
    with pytest.raises(ValueError, match="copy size must be positive"):
        H2DCopyTask(source=object(), destination=object(), nbytes=nbytes)
