#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import logger

from vllm_ascend.distributed.device_communicators.pyhccl_wrapper import (
    HCCLLibrary,
    aclrtStream_t,
    buffer_type,
    hcclComm_t,
    hcclDataTypeEnum,
    hcclRedOpTypeEnum,
    hcclSendRecvItem,
    hcclSendRecvTypeEnum,
    hcclUniqueId,
)
from vllm_ascend.utils import current_stream


class PyHcclCommunicator:
    def __init__(
        self,
        group: ProcessGroup | StatelessProcessGroup,
        device: int | str | torch.device,
        library_path: str | None = None,
    ):
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the PyHcclCommunicator to. If None,
                it will be bind to f"npu:{local_rank}".
            library_path: the path to the HCCL library. If None, it will
                use the default library path.
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device.
        """

        if not isinstance(group, StatelessProcessGroup):
            assert dist.is_initialized()
            assert dist.get_backend(group) != dist.Backend.HCCL, (
                "PyHcclCommunicator should be attached to a non-HCCL group."
            )
            # note: this rank is the rank in the group
            self.rank = dist.get_rank(group)
            self.world_size = dist.get_world_size(group)
        else:
            self.rank = group.rank
            self.world_size = group.world_size

        self.group = group

        # if world_size == 1, no need to create communicator
        if self.world_size == 1:
            self.available = False
            self.disabled = True
            return

        try:
            self.hccl = HCCLLibrary(library_path)
        except Exception:
            # disable because of missing HCCL library
            # e.g. in a non-NPU environment
            self.available = False
            self.disabled = True
            return

        self.available = True
        self.disabled = False

        logger.info("vLLM is using pyhccl")

        if isinstance(device, int):
            device = torch.device(f"npu:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device

        if self.rank == 0:
            # get the unique id from HCCL
            with torch.npu.device(device):
                self.unique_id = self.hccl.hcclGetUniqueId()
        else:
            # construct an empty unique id
            self.unique_id = hcclUniqueId()

        if not isinstance(group, StatelessProcessGroup):
            tensor = torch.ByteTensor(list(self.unique_id.internal))
            ranks = dist.get_process_group_ranks(group)
            # arg `src` in `broadcast` is the global rank
            dist.broadcast(tensor, src=ranks[0], group=group)
            byte_list = tensor.tolist()
            for i, byte in enumerate(byte_list):
                self.unique_id.internal[i] = byte
        else:
            self.unique_id = group.broadcast_obj(self.unique_id, src=0)

        # hccl communicator and stream will use this device
        # `torch.npu.device` is a context manager that changes the
        # current npu device to the specified one
        with torch.npu.device(device):
            self.comm: hcclComm_t = self.hccl.hcclCommInitRank(self.world_size, self.unique_id, self.rank)

            stream = current_stream()
            # A small all_reduce for warmup.
            data = torch.zeros(1, device=device)
            self.all_reduce(data)
            stream.synchronize()
            del data

    def all_reduce(self, in_tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM, stream=None) -> torch.Tensor:
        if self.disabled:
            return None
        # hccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert in_tensor.device == self.device, (
            f"this hccl communicator is created to work on {self.device}, but the input tensor is on {in_tensor.device}"
        )

        out_tensor = torch.empty_like(in_tensor)

        if stream is None:
            stream = current_stream()
        self.hccl.hcclAllReduce(
            buffer_type(in_tensor.data_ptr()),
            buffer_type(out_tensor.data_ptr()),
            in_tensor.numel(),
            hcclDataTypeEnum.from_torch(in_tensor.dtype),
            hcclRedOpTypeEnum.from_torch(op),
            self.comm,
            aclrtStream_t(stream.npu_stream),
        )
        return out_tensor

    def broadcast(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this hccl communicator is created to work on {self.device}, but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = current_stream()
        buffer = buffer_type(tensor.data_ptr())
        self.hccl.hcclBroadcast(
            buffer,
            tensor.numel(),
            hcclDataTypeEnum.from_torch(tensor.dtype),
            src,
            self.comm,
            aclrtStream_t(stream.npu_stream),
        )

    def batch_send_recv(
        self,
        send_tensors: dict[int, torch.Tensor],
        recv_tensors: dict[int, torch.Tensor],
        stream=None,
    ) -> None:
        """Enqueue one raw-pointer send and receive per peer on ``stream``.

        HCCL's batch P2P API accepts device pointers directly, so callers do
        not have to expose internal-format expert storage as distributed
        Tensor slices. Each tensor must nevertheless be a zero-offset,
        contiguous communication allocation because HCCL transfers exactly
        ``numel * element_size`` bytes starting at ``data_ptr``.
        """
        if self.disabled:
            raise RuntimeError("PyHCCL communicator is disabled")
        if not self.hccl.supports_batch_send_recv:
            raise RuntimeError(
                "The loaded HCCL library does not support "
                "HcclBatchSendRecv")
        items = []
        for operation, tensors in (
            (hcclSendRecvTypeEnum.hcclSend, send_tensors),
            (hcclSendRecvTypeEnum.hcclRecv, recv_tensors),
        ):
            for peer_rank, tensor in sorted(tensors.items()):
                if (peer_rank == self.rank
                        or not 0 <= peer_rank < self.world_size):
                    raise ValueError(
                        "Invalid PyHCCL P2P peer: "
                        f"rank={self.rank}, peer={peer_rank}, "
                        f"world_size={self.world_size}")
                if tensor.device != self.device:
                    raise ValueError(
                        "PyHCCL P2P tensor is on the wrong device: "
                        f"expected={self.device}, actual={tensor.device}")
                if (not tensor.is_contiguous()
                        or tensor.storage_offset() != 0):
                    raise ValueError(
                        "PyHCCL P2P requires a contiguous zero-offset tensor: "
                        f"peer={peer_rank}, contiguous={tensor.is_contiguous()}, "
                        f"storage_offset={tensor.storage_offset()}")
                if tensor.numel() <= 0:
                    raise ValueError(
                        f"PyHCCL P2P tensor for peer {peer_rank} is empty")
                items.append(hcclSendRecvItem(
                    sendRecvType=operation,
                    buf=buffer_type(tensor.data_ptr()),
                    count=tensor.numel(),
                    dataType=hcclDataTypeEnum.from_torch(tensor.dtype),
                    remoteRank=peer_rank,
                ))
        if not items:
            return
        if stream is None:
            stream = current_stream()
        self.hccl.hcclBatchSendRecv(
            items, self.comm, aclrtStream_t(stream.npu_stream))

    def close(self) -> None:
        """Destroy the owned HCCL communicator; safe to call repeatedly."""
        if self.disabled:
            return
        self.hccl.hcclCommDestroy(self.comm)
        self.disabled = True
