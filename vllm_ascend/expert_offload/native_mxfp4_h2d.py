"""Native host expert storage -> ND staging -> NZ device slots.

All device work is enqueued on the caller's load stream. Existing manager
synchronization/event boundaries therefore cover both upload and conversion.
No converted weight is downloaded or retained on the host.
"""

from dataclasses import dataclass
import threading
from typing import Any

import torch

from vllm_ascend.expert_offload.h2d_transfer import H2DCopyTask, TorchCopyH2DTransport


@dataclass(frozen=True)
class NativeExpertUploadTask:
    owner: Any
    expert: int
    w13_destination: Any
    w2_destination: Any
    scale13_destination: torch.Tensor
    scale2_destination: torch.Tensor
    name: str = ""


class NativeMXFP4H2DTransport:
    """Decorate the selected transport with native expert conversion tasks.

    Keep one expert-sized staging set per load stream, reused for every model
    layer and expert. The original transport still owns host allocations and
    H2D descriptor retirement. NZ writes use raw device byte copies; scales
    preserve the original transport's tensor-versus-byte copy semantics.
    """

    supports_remote_sources = False

    def __init__(self, base):
        if getattr(base, "supports_remote_sources", False):
            raise ValueError("Native MXFP4 shared storage currently supports single-card offload only")
        self.base = base
        self._staging = {}
        self._last_outputs = {}
        self._lock = threading.RLock()

    def allocate_host_tensor(self, shape, dtype):
        return self.base.allocate_host_tensor(shape, dtype)

    @staticmethod
    def _copy_device_bytes(destination, source, stream):
        # Internal NZ tensors cannot be copied through Tensor.copy_ on A5.
        # ACL consumes physical storage addresses without invoking a format cast.
        import acl

        src = source.untyped_storage() if isinstance(source, torch.Tensor) else source
        if isinstance(destination, torch.Tensor):
            # An expert tensor is a slice of a larger layer storage. Copy only
            # that expert's byte span, starting at the tensor's offset pointer.
            dst_pointer = destination.data_ptr()
            dst_bytes = destination.numel() * destination.element_size()
        else:
            dst_pointer = destination.data_ptr()
            dst_bytes = destination.nbytes()
        if src.nbytes() != dst_bytes:
            raise ValueError(f"Native MXFP4 destination size mismatch: {src.nbytes()} != {dst_bytes}")
        ret = acl.rt.memcpy_async(
            dst_pointer, dst_bytes, src.data_ptr(), src.nbytes(), 3, int(stream.npu_stream)
        )
        if ret != 0:
            raise RuntimeError(f"Native MXFP4 device slot copy failed: ACL error {ret}")

    def _upload(self, task):
        import torch_npu
        from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import apply_mxfp4_weight_scale_layout

        owner = task.owner
        if not 0 <= task.expert < owner.gate.shape[0]:
            raise IndexError(f"Invalid native expert id {task.expert}")
        stream = torch.npu.current_stream(task.scale13_destination.device)
        key = (task.scale13_destination.device.index, int(stream.npu_stream))
        sources = tuple(tensor[task.expert] for tensor in owner.tensors)
        shapes = tuple(tuple(tensor.shape) for tensor in sources)
        cached = self._staging.get(key)
        if cached is None or cached[0] != shapes:
            staging = tuple(
                torch.empty(shape, dtype=torch.uint8, device=task.scale13_destination.device) for shape in shapes
            )
            self._staging[key] = (shapes, staging)
        else:
            staging = cached[1]
        self.base.copy_batch([
            H2DCopyTask(source=source, destination=destination, nbytes=source.numel(), name=task.name)
            for source, destination in zip(sources, staging)
        ])
        gate, up, down, gate_scale, up_scale, down_scale = staging
        w13 = torch.cat((gate, up), dim=0)
        scale13 = torch.cat((gate_scale, up_scale), dim=0)
        nz13 = torch_npu.npu_format_cast(
            w13, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
        )
        nz2 = torch_npu.npu_format_cast(
            down, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
        )
        # The slots carry transpose(NZ(original)); a transpose changes metadata,
        # not the physical storage consumed by the following byte copies.
        self._copy_device_bytes(task.w13_destination, nz13, stream)
        self._copy_device_bytes(task.w2_destination, nz2, stream)
        scale13 = apply_mxfp4_weight_scale_layout(scale13).contiguous()
        scale2 = apply_mxfp4_weight_scale_layout(down_scale).contiguous()
        for source, destination in (
            (scale13, task.scale13_destination), (scale2, task.scale2_destination)
        ):
            source = source.reshape(destination.shape)
            if isinstance(self.base, TorchCopyH2DTransport):
                destination.copy_(source, non_blocking=True)
            else:
                # MemFabric's existing attribute copies are byte-preserving,
                # unlike TorchCopy's logical tensor copies.
                self._copy_device_bytes(destination, source, stream)
        for output in (w13, nz13, nz2, scale13, scale2):
            output.record_stream(stream)
        # Allocator reuse is ordered on this same stream. Only the final
        # outputs are retained; prefill must not retain a full converted layer.
        self._last_outputs[key] = (w13, nz13, nz2, scale13, scale2)

    def copy_batch(self, tasks):
        with self._lock:
            plain = []
            for task in tasks:
                if isinstance(task, NativeExpertUploadTask):
                    if plain:
                        self.base.copy_batch(plain)
                        plain = []
                    self._upload(task)
                else:
                    plain.append(task)
            if plain:
                self.base.copy_batch(plain)

    def synchronize(self, stream):
        with self._lock:
            self.base.synchronize(stream)
            for key in list(self._last_outputs):
                if key[1] == int(stream.npu_stream):
                    del self._last_outputs[key]

    def close(self):
        # Manager.close synchronizes its load stream before this is called.
        with self._lock:
            self._last_outputs.clear()
            self._staging.clear()
            self.base.close()
