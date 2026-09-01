"""Host-to-device transfer abstractions for expert offload."""

import atexit
import os
import re
import shutil
import socket
import tempfile
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import torch


ONE_GIB = 1 << 30
UINT32_MAX = (1 << 32) - 1
_SMAPS_HEADER_RE = re.compile(r"^[0-9A-Fa-f]+-[0-9A-Fa-f]+\s")


class CopyDirection(str, Enum):
    """Direction of one byte-preserving expert-weight transfer."""

    H2D = "h2d"
    D2H = "d2h"
    D2D = "d2d"


@dataclass(frozen=True)
class H2DCopyTask:
    """One byte-preserving expert copy.

    ``source`` and ``destination`` intentionally accept both tensors and
    untyped-storage slices. Expert weights use storage slices to preserve their
    packed/NZ byte layout, while quantization attributes use shaped tensors.
    ``nbytes`` is carried explicitly so pointer-based backends can consume the
    same task without inferring sizes from tensor metadata.  The historical
    class name is retained for compatibility; ``direction`` distinguishes H2D,
    D2H and D2D operations used by dynamic exclusive expert ownership.
    """

    source: Any
    destination: Any
    nbytes: int
    name: str = ""
    non_blocking: bool = True
    direction: CopyDirection = CopyDirection.H2D

    def __post_init__(self) -> None:
        if self.nbytes <= 0:
            raise ValueError(
                f"Expert copy size must be positive; got {self.nbytes}")
        if not isinstance(self.direction, CopyDirection):
            try:
                object.__setattr__(self, "direction",
                                   CopyDirection(self.direction))
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported expert copy direction: {self.direction}"
                ) from exc


@dataclass(frozen=True)
class HostPointerSource:
    """Pointer-only view of a peer rank's MemFabric SHARED allocation."""

    pointer: int

    def data_ptr(self) -> int:
        return self.pointer


class ExpertH2DTransport(Protocol):
    """Backend contract for a batch of expert H2D copies."""

    def allocate_host_tensor(self, shape, dtype) -> torch.Tensor:
        """Allocate a CPU tensor suitable as this backend's H2D source."""
        ...

    def copy_batch(self, tasks: list[H2DCopyTask]) -> None:
        """Enqueue all tasks on the caller-selected device stream."""
        ...

    def synchronize(self, stream) -> None:
        """Wait for submitted copies and release descriptor resources."""
        ...

    def close(self) -> None:
        """Release backend resources."""
        ...


class TorchCopyH2DTransport:
    """Existing expert-offload H2D behavior implemented with ``copy_``."""

    supports_remote_sources = False

    def allocate_host_tensor(self, shape, dtype) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)

    def copy_batch(self, tasks: list[H2DCopyTask]) -> None:
        for task in tasks:
            task.destination.copy_(task.source,
                                   non_blocking=task.non_blocking)

    @staticmethod
    def synchronize(stream) -> None:
        stream.synchronize()

    @staticmethod
    def close() -> None:
        return


class TorchSharedCPUH2DTransport(TorchCopyH2DTransport):
    """Torch H2D from pageable shared mappings via pinned staging buffers.

    ``torch.from_file`` mappings cannot be pinned. Keeping the canonical
    expert weights in one shared mapping therefore needs a short-lived,
    rank-local pinned copy before an asynchronous NPU ``copy_``. Staging
    tensors are retained until ``synchronize`` so their storage cannot be
    recycled while the device copy is still in flight.
    """

    supports_shared_cpu_sources = True

    def __init__(self) -> None:
        self._inflight_staging: list[tuple[tuple, torch.Tensor]] = []
        self._free_staging: dict[tuple, list[torch.Tensor]] = {}
        self._staging_lock = threading.Lock()

    def _acquire_staging(self, key: tuple, shape, dtype) -> torch.Tensor:
        available = self._free_staging.get(key)
        if available:
            return available.pop()
        return torch.empty(
            shape, dtype=dtype, device="cpu", pin_memory=True)

    def _stage_source(
        self, task: H2DCopyTask
    ) -> tuple[tuple, torch.Tensor, Any]:
        if isinstance(task.source, torch.Tensor):
            key = ("tensor", tuple(task.source.shape), task.source.dtype)
            staging = self._acquire_staging(
                key, task.source.shape, task.source.dtype)
            staging.copy_(task.source)
            return key, staging, staging

        key = ("storage", task.nbytes, torch.uint8)
        staging = self._acquire_staging(
            key, (task.nbytes,), torch.uint8)
        staging.untyped_storage().copy_(task.source)
        return key, staging, staging.untyped_storage()

    def copy_batch(self, tasks: list[H2DCopyTask]) -> None:
        with self._staging_lock:
            for task in tasks:
                source = task.source
                if task.direction == CopyDirection.H2D:
                    key, staging, source = self._stage_source(task)
                    self._inflight_staging.append((key, staging))
                task.destination.copy_(
                    source, non_blocking=task.non_blocking)

    def synchronize(self, stream) -> None:
        with self._staging_lock:
            stream.synchronize()
            for key, staging in self._inflight_staging:
                self._free_staging.setdefault(key, []).append(staging)
            self._inflight_staging.clear()

    def close(self) -> None:
        with self._staging_lock:
            self._inflight_staging.clear()
            self._free_staging.clear()


class TorchSharedCPUWeightPool:
    """Collectively allocate same-node file-backed CPU tensors.

    Every rank maps the same file with ``MAP_SHARED`` semantics. The backing
    file is unlinked after all ranks have mapped it, so the physical pages
    live exactly as long as the tensors and no large files remain after an
    abnormal process exit. Calls to :meth:`allocate` must occur in identical
    order on every rank in ``cpu_group``.
    """

    def __init__(
        self,
        base_dir: str,
        world_size: int,
        rank_id: int,
        cpu_group,
        *,
        dist_module=None,
    ) -> None:
        if world_size <= 1:
            raise ValueError(
                "Torch shared CPU weights require world_size > 1")
        if not 0 <= rank_id < world_size:
            raise ValueError(
                f"Invalid Torch shared CPU rank: {rank_id}/{world_size}")
        if not base_dir:
            raise ValueError("Torch shared CPU weight directory is empty")
        if dist_module is None:
            from torch import distributed as dist
            dist_module = dist

        self.base_dir = os.path.abspath(base_dir)
        self.world_size = world_size
        self.rank_id = rank_id
        self.cpu_group = cpu_group
        self._dist = dist_module
        self._allocation_index = 0
        self._closed = False

        hostnames = [None] * world_size
        self._dist.all_gather_object(
            hostnames, socket.gethostname(), group=cpu_group)
        if len(set(hostnames)) != 1:
            raise RuntimeError(
                "Torch shared CPU weights are same-node only; EP ranks are "
                f"on different hosts: {hostnames}")

        local_run_dir = None
        local_run_dir_error = None
        if rank_id == 0:
            try:
                os.makedirs(self.base_dir, exist_ok=True)
                local_run_dir = tempfile.mkdtemp(
                    prefix="vllm-ascend-torch-shared-",
                    dir=self.base_dir,
                )
            except OSError as exc:
                local_run_dir_error = f"{type(exc).__name__}: {exc}"
        run_dir_states = [None] * world_size
        self._dist.all_gather_object(
            run_dir_states,
            (local_run_dir, local_run_dir_error),
            group=cpu_group,
        )
        run_dir, run_dir_error = run_dir_states[0]
        if run_dir_error is not None:
            raise RuntimeError(
                "Failed to create Torch shared CPU weight directory: "
                f"{run_dir_error}")
        if not isinstance(run_dir, str) or not os.path.isabs(run_dir):
            raise RuntimeError(
                "Failed to create Torch shared CPU weight directory: "
                f"{run_dir!r}")
        if not os.path.isdir(run_dir):
            raise RuntimeError(
                "Torch shared CPU weight directory is not visible to all "
                f"ranks: {run_dir}")
        self.run_dir = run_dir

    @staticmethod
    def _safe_name(name: str) -> str:
        safe = "".join(
            char if char.isalnum() or char in "-_." else "_"
            for char in name)
        return safe[:96] or "buffer"

    def allocate(self, name: str, shape, dtype) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("Torch shared CPU weight pool is closed")
        shape = tuple(int(dim) for dim in shape)
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError(
                f"Shared CPU tensor shape must be positive; got {shape}")
        element_size = torch.empty((), dtype=dtype).element_size()
        numel = 1
        for dim in shape:
            numel *= dim
        nbytes = numel * element_size
        index = self._allocation_index
        self._allocation_index += 1
        path = os.path.join(
            self.run_dir, f"{index:05d}-{self._safe_name(name)}.bin")

        local_create_error = None
        if self.rank_id == 0:
            try:
                descriptor = os.open(
                    path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                try:
                    os.ftruncate(descriptor, nbytes)
                finally:
                    os.close(descriptor)
            except OSError as exc:
                local_create_error = f"{type(exc).__name__}: {exc}"
        create_errors = [None] * self.world_size
        self._dist.all_gather_object(
            create_errors, local_create_error, group=self.cpu_group)
        create_error = create_errors[0]
        if create_error is not None:
            raise RuntimeError(
                "Failed to create Torch shared CPU weight file "
                f"{path}: {create_error}")

        tensor = None
        map_error = None
        try:
            tensor = torch.from_file(
                path, shared=True, size=numel, dtype=dtype).reshape(shape)
        except Exception as exc:
            map_error = f"rank={self.rank_id} {type(exc).__name__}: {exc}"
        map_errors = [None] * self.world_size
        self._dist.all_gather_object(
            map_errors, map_error, group=self.cpu_group)
        failures = [error for error in map_errors if error is not None]
        if failures:
            if self.rank_id == 0:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
            raise RuntimeError(
                "Failed to map Torch shared CPU weight file: "
                + "; ".join(failures))

        # Ensure every successful mapping exists before rank 0 unlinks the
        # name. The mappings keep the inode/pages alive after this point.
        self._dist.barrier(group=self.cpu_group)
        local_unlink_error = None
        if self.rank_id == 0:
            try:
                os.unlink(path)
            except OSError as exc:
                local_unlink_error = f"{type(exc).__name__}: {exc}"
        unlink_errors = [None] * self.world_size
        self._dist.all_gather_object(
            unlink_errors, local_unlink_error, group=self.cpu_group)
        unlink_error = unlink_errors[0]
        if unlink_error is not None:
            raise RuntimeError(
                "Failed to unlink mapped Torch shared CPU weight file "
                f"{path}: {unlink_error}")
        return tensor

    def mapping_memory_snapshot(self) -> dict[str, int] | None:
        """Return this process's smaps accounting for pool mappings.

        Backing files are unlinked after every rank maps them, but Linux keeps
        the original path with a ``(deleted)`` suffix in ``/proc/self/smaps``.
        Matching the per-run directory therefore excludes unrelated model and
        process memory while retaining the shared expert mappings.
        """
        totals = {
            "size_bytes": 0,
            "rss_bytes": 0,
            "pss_bytes": 0,
            "swap_pss_bytes": 0,
        }
        mapping_count = 0
        in_pool_mapping = False
        mapping_prefix = self.run_dir + os.sep
        smaps_keys = {
            "Size": "size_bytes",
            "Rss": "rss_bytes",
            "Pss": "pss_bytes",
            "SwapPss": "swap_pss_bytes",
        }
        try:
            with open("/proc/self/smaps", encoding="utf-8") as smaps_file:
                for line in smaps_file:
                    if _SMAPS_HEADER_RE.match(line):
                        in_pool_mapping = mapping_prefix in line
                        if in_pool_mapping:
                            mapping_count += 1
                        continue
                    if not in_pool_mapping:
                        continue
                    key, separator, value = line.partition(":")
                    output_key = smaps_keys.get(key)
                    if separator and output_key is not None:
                        totals[output_key] += (
                            int(value.strip().split()[0]) * 1024)
        except (OSError, ValueError, IndexError):
            return None
        totals["mapping_count"] = mapping_count
        return totals

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.rank_id == 0:
            shutil.rmtree(self.run_dir, ignore_errors=True)


class MemFabricLocalH2DTransport:
    """MemFabric LOCAL DRAM allocator and sparse-copy H2D backend.

    MemFabric launches ``sparse_copy`` on torch-npu's current stream. The
    manager selects its load stream before calling ``copy_batch`` and later
    calls ``synchronize`` on that same stream, preserving graph host-callback
    ordering without a device-wide synchronization.
    """

    supports_remote_sources = False

    def __init__(
        self,
        device_id: int,
        pool_size_gib: int,
        log_level: int = 3,
        *,
        mf_module=None,
        offload_module=None,
    ) -> None:
        if pool_size_gib <= 0:
            raise ValueError(
                "MemFabric pool size must be positive; "
                f"got {pool_size_gib} GiB")
        if mf_module is None or offload_module is None:
            try:
                import memfabric_hybrid as mf
                from memfabric_hybrid import offload
            except ImportError as exc:
                raise RuntimeError(
                    "h2d_backend='memfabric' requires memfabric_hybrid; "
                    "install it and source the MemFabric environment first"
                ) from exc
            mf_module = mf
            offload_module = offload

        self.device_id = device_id
        self.device = torch.device(f"npu:{device_id}")
        self._offload = offload_module
        self._closed = False
        self._inflight_descriptors = []
        self._descriptor_lock = threading.Lock()

        mf_module.set_log_level(log_level)
        config = offload_module.OffloadConfig()
        config.device_id = device_id
        config.reserve_size = pool_size_gib * ONE_GIB
        config.alloc_size = config.reserve_size
        ret = offload_module.initialize(config)
        if ret != 0:
            raise RuntimeError(
                "MemFabric LOCAL initialization failed: "
                f"device={device_id}, pool_size_gib={pool_size_gib}, ret={ret}")
        atexit.register(self.close)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MemFabric H2D transport is already closed")

    def allocate_host_tensor(self, shape, dtype) -> torch.Tensor:
        self._ensure_open()
        return self._offload.empty(shape, dtype=dtype)

    @staticmethod
    def _int32_bits(value: int) -> int:
        if not 0 <= value <= UINT32_MAX:
            raise ValueError(
                "MemFabric sparse-copy length exceeds uint32: "
                f"{value} bytes")
        return value if value < (1 << 31) else value - (1 << 32)

    def copy_batch(self, tasks: list[H2DCopyTask]) -> None:
        self._ensure_open()
        if not tasks:
            return
        unsupported = [task for task in tasks
                       if task.direction != CopyDirection.H2D]
        if unsupported:
            raise RuntimeError(
                "MemFabric expert transport currently supports H2D only; "
                f"got direction={unsupported[0].direction.value} "
                f"for task={unsupported[0].name!r}")

        src_values = [task.source.data_ptr() for task in tasks]
        dst_values = [task.destination.data_ptr() for task in tasks]
        length_values = [self._int32_bits(task.nbytes) for task in tasks]

        # Release 1.2's sparse-copy kernel splits the descriptor list into two
        # equal halves. Pad odd batches with a zero-byte no-op descriptor so no
        # real task is dropped.
        if len(tasks) % 2:
            src_values.append(src_values[-1])
            dst_values.append(dst_values[-1])
            length_values.append(0)

        src_ptrs = torch.tensor(src_values,
                                dtype=torch.int64,
                                device=self.device)
        dst_ptrs = torch.tensor(dst_values,
                                dtype=torch.int64,
                                device=self.device)
        lengths = torch.tensor(length_values,
                               dtype=torch.int32,
                               device=self.device)
        count = torch.tensor(len(src_values),
                             dtype=torch.int32,
                             device=self.device)
        with self._descriptor_lock:
            self._ensure_open()
            ret = self._offload.sparse_copy(src_ptrs, dst_ptrs, lengths,
                                            count, self.device)
            if ret != 0:
                raise RuntimeError(
                    "MemFabric sparse_copy failed: "
                    f"device={self.device_id}, tasks={len(tasks)}, ret={ret}")
            self._inflight_descriptors.append(
                (src_ptrs, dst_ptrs, lengths, count))

    def synchronize(self, stream) -> None:
        with self._descriptor_lock:
            stream.synchronize()
            self._inflight_descriptors.clear()

    def close(self) -> None:
        with self._descriptor_lock:
            if self._closed:
                return
            self._offload.uninitialize()
            self._inflight_descriptors.clear()
            self._closed = True
        atexit.unregister(self.close)


class MemFabricSharedH2DTransport(MemFabricLocalH2DTransport):
    """Multi-card SHARED DRAM backend with one contribution per EP rank."""

    supports_remote_sources = True

    def __init__(
        self,
        device_id: int,
        pool_size_gib: int,
        world_size: int,
        rank_id: int,
        log_level: int = 3,
        *,
        mf_module=None,
        offload_module=None,
    ) -> None:
        if world_size <= 1:
            raise ValueError(
                "MemFabric SHARED world_size must be greater than one")
        if not 0 <= rank_id < world_size:
            raise ValueError(
                f"Invalid MemFabric SHARED rank: {rank_id}/{world_size}")
        if pool_size_gib <= 0:
            raise ValueError(
                "MemFabric pool size must be positive; "
                f"got {pool_size_gib} GiB")
        if mf_module is None or offload_module is None:
            try:
                import memfabric_hybrid as mf
                from memfabric_hybrid import offload
            except ImportError as exc:
                raise RuntimeError(
                    "h2d_backend='memfabric' requires memfabric_hybrid; "
                    "install it and source the MemFabric environment first"
                ) from exc
            mf_module = mf
            offload_module = offload

        self.device_id = device_id
        self.device = torch.device(f"npu:{device_id}")
        self._offload = offload_module
        self._closed = False
        self._inflight_descriptors = []
        self._descriptor_lock = threading.Lock()
        self.world_size = world_size
        self.rank_id = rank_id

        mf_module.set_log_level(log_level)
        config = offload_module.OffloadConfig()
        config.device_id = device_id
        config.reserve_size = pool_size_gib * ONE_GIB
        config.alloc_size = config.reserve_size
        config.world_size = world_size
        config.rank_id = rank_id
        config.scene = offload_module.Scene.SHARED
        ret = offload_module.initialize(config)
        if ret != 0:
            raise RuntimeError(
                "MemFabric SHARED initialization failed: "
                f"device={device_id}, rank={rank_id}/{world_size}, "
                f"pool_size_gib={pool_size_gib}, ret={ret}")
        atexit.register(self.close)


def create_h2d_transport(
    backend: str,
    *,
    device_id: int,
    memfabric_pool_size_gib: int,
    memfabric_log_level: int,
    enable_multi_card: bool = False,
    enable_torch_shared_cpu: bool = False,
    world_size: int = 1,
    rank_id: int = 0,
) -> ExpertH2DTransport:
    if backend == "torch":
        if enable_torch_shared_cpu:
            return TorchSharedCPUH2DTransport()
        return TorchCopyH2DTransport()
    if backend == "memfabric":
        if enable_multi_card:
            return MemFabricSharedH2DTransport(
                device_id=device_id,
                pool_size_gib=memfabric_pool_size_gib,
                world_size=world_size,
                rank_id=rank_id,
                log_level=memfabric_log_level,
            )
        return MemFabricLocalH2DTransport(
            device_id=device_id,
            pool_size_gib=memfabric_pool_size_gib,
            log_level=memfabric_log_level,
        )
    raise ValueError(f"Unsupported expert-offload H2D backend: {backend}")
