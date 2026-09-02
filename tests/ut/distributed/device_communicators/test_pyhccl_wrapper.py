import ctypes
from unittest.mock import MagicMock, patch

import torch
from torch.distributed import ReduceOp

from tests.ut.base import TestBase
from vllm_ascend.distributed.device_communicators.pyhccl_wrapper import (
    Function,
    HCCLLibrary,
    aclrtStream_t,
    buffer_type,
    hcclComm_t,
    hcclDataType_t,
    hcclDataTypeEnum,
    hcclRedOp_t,
    hcclRedOpTypeEnum,
    hcclResult_t,
    hcclSendRecvItem,
    hcclSendRecvTypeEnum,
    hcclUniqueId,
)


class TestHcclUniqueId(TestBase):
    def test_construct(self):
        uid = hcclUniqueId()
        uid.internal[0] = 12
        self.assertEqual(len(uid.internal), 4108)
        self.assertEqual(uid.internal[0], 12)


class TestHcclDataTypeEnum(TestBase):
    def test_torch_dtype_mapping(self):
        expected = {
            torch.int8: hcclDataTypeEnum.hcclInt8,
            torch.uint8: hcclDataTypeEnum.hcclUint8,
            torch.int32: hcclDataTypeEnum.hcclInt32,
            torch.int64: hcclDataTypeEnum.hcclInt64,
            torch.float16: hcclDataTypeEnum.hcclFloat16,
            torch.float32: hcclDataTypeEnum.hcclFloat32,
            torch.float64: hcclDataTypeEnum.hcclFloat64,
            torch.bfloat16: hcclDataTypeEnum.hcclBfloat16,
        }

        for torch_dtype, expected_enum in expected.items():
            with self.subTest(torch_dtype=torch_dtype):
                self.assertEqual(hcclDataTypeEnum.from_torch(torch_dtype), expected_enum)

    def test_unsupported_dtype_raises(self):
        with self.assertRaises(ValueError):
            hcclDataTypeEnum.from_torch(torch.complex64)


class TestHcclSendRecvItem(TestBase):
    def test_abi_layout(self):
        self.assertEqual(ctypes.sizeof(hcclSendRecvItem), 32)
        self.assertEqual(hcclSendRecvItem.sendRecvType.offset, 0)
        self.assertEqual(hcclSendRecvItem.buf.offset, 8)
        self.assertEqual(hcclSendRecvItem.count.offset, 16)
        self.assertEqual(hcclSendRecvItem.dataType.offset, 24)
        self.assertEqual(hcclSendRecvItem.remoteRank.offset, 28)

    def test_construct(self):
        item = hcclSendRecvItem(
            sendRecvType=hcclSendRecvTypeEnum.hcclRecv,
            buf=buffer_type(1234),
            count=4096,
            dataType=hcclDataTypeEnum.hcclUint8,
            remoteRank=3,
        )

        self.assertEqual(
            item.sendRecvType, hcclSendRecvTypeEnum.hcclRecv)
        self.assertEqual(item.buf, 1234)
        self.assertEqual(item.count, 4096)
        self.assertEqual(item.dataType, hcclDataTypeEnum.hcclUint8)
        self.assertEqual(item.remoteRank, 3)


class TestHcclRedOpTypeEnum(TestBase):
    def test_torch_reduce_op_mapping(self):
        expected = {
            ReduceOp.SUM: hcclRedOpTypeEnum.hcclSum,
            ReduceOp.PRODUCT: hcclRedOpTypeEnum.hcclProd,
            ReduceOp.MAX: hcclRedOpTypeEnum.hcclMax,
            ReduceOp.MIN: hcclRedOpTypeEnum.hcclMin,
        }

        for torch_op, expected_enum in expected.items():
            with self.subTest(torch_op=torch_op):
                self.assertEqual(hcclRedOpTypeEnum.from_torch(torch_op), expected_enum)

    def test_unsupported_op_raises(self):
        unsupported_op = "NOT_EXIST"
        with self.assertRaises(ValueError):
            hcclRedOpTypeEnum.from_torch(unsupported_op)


class TestFunction(TestBase):
    def test_construct_with_valid_args(self):
        func = Function(name="foo", restype=int, argtypes=[int, str, float])
        self.assertEqual(func.name, "foo")
        self.assertIs(func.restype, int)
        self.assertEqual(func.argtypes, [int, str, float])


class TestHCLLLibrary(TestBase):
    def test_init_with_nonexistent_so(self):
        fake_path = "/definitely/not/exist/libhccl.so"
        with self.assertRaises(OSError):
            HCCLLibrary(fake_path)

    def test_hccl_get_error_string(self):
        lib = MagicMock(sepc=HCCLLibrary)
        mock_fn = MagicMock()
        mock_fn.return_value = "HCCL internal error"
        lib.hcclGetErrorString = mock_fn

        result = hcclResult_t(1)
        msg = lib.hcclGetErrorString(result)
        self.assertEqual(msg, "HCCL internal error")
        mock_fn.assert_called_once()

    def test_hccl_check(self):
        lib = HCCLLibrary.__new__(HCCLLibrary)
        mock_fn = MagicMock()
        mock_fn.return_value = "fake error"
        lib.hcclGetErrorString = mock_fn
        result = hcclResult_t(123)
        with self.assertRaises(RuntimeError) as cm:
            lib.HCCL_CHECK(result)

        self.assertEqual(str(cm.exception), "HCCL error: fake error")

    @patch.object(HCCLLibrary, "HCCL_CHECK")
    def test_hccl_get_uniqueId(self, mock_HCCL_CHECK):
        lib = HCCLLibrary.__new__(HCCLLibrary)
        lib._funcs = {"HcclGetRootInfo": MagicMock(return_value=0)}
        unique_id = lib.hcclGetUniqueId()
        self.assertIsInstance(unique_id, hcclUniqueId)
        lib._funcs["HcclGetRootInfo"].assert_called_once()
        mock_HCCL_CHECK.assert_called_once_with(0)

    @patch.object(HCCLLibrary, "HCCL_CHECK")
    def test_hccl_comm_initRank(self, mock_hccl_check):
        lib = HCCLLibrary.__new__(HCCLLibrary)
        lib._funcs = {"HcclCommInitRootInfo": MagicMock(return_value=0)}

        world_size = 4
        unique_id = hcclUniqueId()
        rank = 1

        comm = lib.hcclCommInitRank(world_size, unique_id, rank)
        self.assertIsInstance(comm, hcclComm_t)
        lib._funcs["HcclCommInitRootInfo"].assert_called_once()
        mock_hccl_check.assert_called_once_with(0)

    @patch.object(HCCLLibrary, "HCCL_CHECK")
    def test_hccl_all_reduce(self, mock_hccl_check):
        lib = HCCLLibrary.__new__(HCCLLibrary)
        lib._funcs = {"HcclAllReduce": MagicMock(return_value=0)}
        sendbuff = buffer_type()
        recvbuff = buffer_type()
        count = 10
        datatype = hcclDataType_t(1)
        op = hcclRedOp_t(0)
        comm = hcclComm_t()
        stream = aclrtStream_t()

        lib.hcclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream)

        lib._funcs["HcclAllReduce"].assert_called_once_with(sendbuff, recvbuff, count, datatype, op, comm, stream)
        mock_hccl_check.assert_called_once_with(0)

    @patch.object(HCCLLibrary, "HCCL_CHECK")
    def test_hccl_broad_cast(self, mock_hccl_check):
        lib = HCCLLibrary.__new__(HCCLLibrary)
        lib._funcs = {"HcclBroadcast": MagicMock(return_value=0)}
        buff = buffer_type()
        count = 10
        datatype = 1
        root = 0
        comm = hcclComm_t()
        stream = aclrtStream_t()

        lib.hcclBroadcast(buff, count, datatype, root, comm, stream)

        lib._funcs["HcclBroadcast"].assert_called_once_with(buff, count, datatype, root, comm, stream)
        mock_hccl_check.assert_called_once_with(0)

    @patch.object(HCCLLibrary, "HCCL_CHECK")
    def test_hccl_batch_send_recv(self, mock_hccl_check):
        lib = HCCLLibrary.__new__(HCCLLibrary)
        batch = MagicMock(return_value=0)
        lib._funcs = {"HcclBatchSendRecv": batch}
        items = [
            hcclSendRecvItem(
                sendRecvType=hcclSendRecvTypeEnum.hcclSend,
                buf=buffer_type(1234),
                count=16,
                dataType=hcclDataTypeEnum.hcclUint8,
                remoteRank=1,
            ),
            hcclSendRecvItem(
                sendRecvType=hcclSendRecvTypeEnum.hcclRecv,
                buf=buffer_type(5678),
                count=32,
                dataType=hcclDataTypeEnum.hcclUint8,
                remoteRank=1,
            ),
        ]
        comm = hcclComm_t(99)
        stream = aclrtStream_t(77)

        lib.hcclBatchSendRecv(items, comm, stream)

        item_array, item_count, actual_comm, actual_stream = (
            batch.call_args.args)
        self.assertEqual(item_count, 2)
        self.assertEqual(item_array[0].buf, 1234)
        self.assertEqual(item_array[1].buf, 5678)
        self.assertEqual(actual_comm, comm)
        self.assertEqual(actual_stream, stream)
        mock_hccl_check.assert_called_once_with(0)

    def test_hccl_batch_send_recv_requires_exported_symbol(self):
        lib = HCCLLibrary.__new__(HCCLLibrary)
        lib._funcs = {}
        self.assertFalse(lib.supports_batch_send_recv)

        with self.assertRaisesRegex(
                RuntimeError, "does not export HcclBatchSendRecv"):
            lib.hcclBatchSendRecv(
                [hcclSendRecvItem()], hcclComm_t(), aclrtStream_t())

    @patch.object(HCCLLibrary, "HCCL_CHECK")
    def test_hcclCommDestroy_success(self, mock_hccl_check):
        lib = HCCLLibrary.__new__(HCCLLibrary)
        lib._funcs = {"HcclCommDestroy": MagicMock(return_value=0)}
        comm = hcclComm_t()
        lib.hcclCommDestroy(comm)
        lib._funcs["HcclCommDestroy"].assert_called_once_with(comm)
        mock_hccl_check.assert_called_once_with(0)
