import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.distributed.utils import StatelessProcessGroup

from tests.ut.base import TestBase
from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator
from vllm_ascend.distributed.device_communicators.pyhccl_wrapper import (
    hcclComm_t,
    hcclDataTypeEnum,
    hcclSendRecvTypeEnum,
)


class MockHcclLib:
    pass


class MockUniqueId:
    pass


class TestPyHcclCommunicator(TestBase):
    @patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "1"})
    def test_world_size_1_return_early(self):
        comm = PyHcclCommunicator(
            group=StatelessProcessGroup(0, 1, None, None),
            device="npu:0",
        )
        self.assertTrue(comm.disabled)
        self.assertFalse(comm.available)

    @patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "2"})
    def test_load_hccl_fail(self):
        comm = PyHcclCommunicator(
            group=StatelessProcessGroup(0, 2, None, None), device="npu:0", library_path="/not/exist/path/libhccl.so"
        )
        self.assertTrue(comm.disabled)

    @patch("vllm_ascend.distributed.device_communicators.pyhccl_wrapper.HCCLLibrary", MockHcclLib)
    @patch("vllm_ascend.distributed.device_communicators.pyhccl_wrapper.hcclUniqueId", MockUniqueId)
    @patch("torch.npu.device")
    @patch("vllm_ascend.utils.current_stream", return_value=MagicMock(npu_stream=5678))
    def test_stateless_group(self, *_):
        group = StatelessProcessGroup(rank=3, world_size=4, store=None)

        comm = PyHcclCommunicator(group=group, device=3)

        self.assertEqual(comm.rank, 3)
        self.assertEqual(comm.world_size, 4)

    @patch.dict(os.environ, {"RANK": "1", "WORLD_SIZE": "2"})
    @patch("vllm_ascend.distributed.device_communicators.pyhccl_wrapper.HCCLLibrary", MockHcclLib)
    @patch("vllm_ascend.distributed.device_communicators.pyhccl_wrapper.hcclUniqueId", MockUniqueId)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_backend", return_value="nccl")
    @patch("torch.distributed.Backend.HCCL", "hccl", create=True)
    @patch("torch.distributed.get_rank", return_value=1)
    @patch("torch.distributed.get_world_size", return_value=2)
    @patch("torch.distributed.get_process_group_ranks", return_value=[0, 1])
    @patch("torch.distributed.broadcast")
    @patch("torch.npu.device")
    @patch("vllm_ascend.utils.current_stream", return_value=MagicMock(npu_stream=1234))
    def test_multi_gpu_pg_torch(
        self,
        *_,
    ):
        fake_pg = MagicMock()
        comm = PyHcclCommunicator(group=fake_pg, device="npu:1")

        self.assertEqual(comm.rank, 1)
        self.assertEqual(comm.world_size, 2)
        self.assertFalse(comm.available)
        self.assertTrue(comm.disabled)

    def test_batch_send_recv_passes_raw_buffers_and_stream(self):
        comm = PyHcclCommunicator.__new__(PyHcclCommunicator)
        comm.disabled = False
        comm.rank = 0
        comm.world_size = 2
        comm.device = torch.device("cpu")
        comm.comm = hcclComm_t(99)
        comm.hccl = MagicMock()
        comm.hccl.supports_batch_send_recv = True
        send = torch.arange(8, dtype=torch.uint8)
        recv = torch.zeros(12, dtype=torch.uint8)
        stream = SimpleNamespace(npu_stream=77)

        comm.batch_send_recv({1: send}, {1: recv}, stream=stream)

        items, actual_comm, actual_stream = (
            comm.hccl.hcclBatchSendRecv.call_args.args)
        self.assertEqual(len(items), 2)
        self.assertEqual(
            items[0].sendRecvType, hcclSendRecvTypeEnum.hcclSend)
        self.assertEqual(items[0].buf, send.data_ptr())
        self.assertEqual(items[0].count, send.numel())
        self.assertEqual(
            items[0].dataType, hcclDataTypeEnum.hcclUint8)
        self.assertEqual(items[0].remoteRank, 1)
        self.assertEqual(
            items[1].sendRecvType, hcclSendRecvTypeEnum.hcclRecv)
        self.assertEqual(items[1].buf, recv.data_ptr())
        self.assertEqual(actual_comm, comm.comm)
        self.assertEqual(actual_stream.value, stream.npu_stream)

    def test_batch_send_recv_rejects_nonzero_storage_offset(self):
        comm = PyHcclCommunicator.__new__(PyHcclCommunicator)
        comm.disabled = False
        comm.rank = 0
        comm.world_size = 2
        comm.device = torch.device("cpu")
        comm.comm = hcclComm_t(99)
        comm.hccl = MagicMock()
        comm.hccl.supports_batch_send_recv = True
        offset_view = torch.empty(9, dtype=torch.uint8)[1:]

        with self.assertRaisesRegex(ValueError, "zero-offset"):
            comm.batch_send_recv({1: offset_view}, {})

        comm.hccl.hcclBatchSendRecv.assert_not_called()

    def test_close_destroys_communicator_once(self):
        comm = PyHcclCommunicator.__new__(PyHcclCommunicator)
        comm.disabled = False
        comm.comm = hcclComm_t(99)
        comm.hccl = MagicMock()

        comm.close()
        comm.close()

        comm.hccl.hcclCommDestroy.assert_called_once_with(comm.comm)
        self.assertTrue(comm.disabled)
