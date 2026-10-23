"""Device-owned hybrid admission and mapped-DRAM expert uploads.

CPU receives only the final compute routes. Placement stays on device; one
parallel copy kernel converts native bytes directly into the selected HBM slot.
"""

import ctypes
import os
from pathlib import Path

import torch

from vllm_ascend.expert_offload.frequency_transfer_policy import (
    FREQUENCY_DECAY, FREQUENCY_MIN_SCORE, FREQUENCY_REPLACEMENT_MARGIN,
    FREQUENCY_REPLACEMENT_RATIO,
)


class DeviceExpertPolicy:
    """Fixed-shape routing statistics and placement; no device-to-host reads."""

    def __init__(self, experts, slots, device, policy):
        self.policy = policy
        self.experts = torch.arange(experts, device=device, dtype=torch.int64)
        self.slots = torch.arange(slots, device=device, dtype=torch.int64)
        self.scores = torch.zeros(experts, device=device, dtype=torch.float32)
        self.counts = torch.zeros_like(self.scores)

    def select(self, ids, mapping):
        flat = ids.reshape(-1).long()
        valid = (flat >= 0) & (flat < mapping.numel())
        safe = flat.clamp(0, mapping.numel() - 1)
        self.counts.zero_().scatter_add_(0, safe, valid.float())
        self.scores.mul_(FREQUENCY_DECAY).add_(self.counts)
        routed = self.counts > 0
        missing = routed & (mapping < 0)
        owners = mapping[:, None] == self.slots[None, :]
        protected = (owners & routed[:, None]).any(dim=0)
        available = ~protected
        positions = torch.arange(flat.numel(), device=ids.device, dtype=torch.int64)
        first = torch.where(
            (self.experts[:, None] == flat[None, :]) & valid[None, :],
            positions[None, :], flat.numel()).amin(dim=1)
        if self.policy == "frequency":
            best = torch.where(missing, self.scores, -1).amax()
            candidate = torch.where(missing & (self.scores == best), first, flat.numel()).argmin()
            slot_scores = torch.where(owners, self.scores[:, None], 0).amax(dim=0)
            slot = torch.where(available, slot_scores, float("inf")).argmin()
            demand, victim = self.scores[candidate], slot_scores[slot]
            admit = ((demand >= FREQUENCY_MIN_SCORE)
                     & (demand > victim * FREQUENCY_REPLACEMENT_RATIO)
                     & ((demand - victim) > FREQUENCY_REPLACEMENT_MARGIN))
        else:
            candidate = torch.where(missing, first, flat.numel()).argmin()
            slot = torch.where(available, self.slots, self.slots.numel()).argmin()
            admit = torch.ones((), device=ids.device, dtype=torch.bool)
        active = missing.any() & available.any() & admit
        updated = torch.where(active & (mapping == slot), -1, mapping)
        updated = torch.where(active & (self.experts == candidate), slot.to(mapping.dtype), updated)
        mapping.copy_(updated)
        return candidate, slot, active


class DeviceSparseCopy:
    """Keep mapped host owners and launch the MemFabric byte-copy kernel."""

    def __init__(self, library):
        if not Path(library).is_file():
            raise ValueError(f"NPU expert copy kernel not found: {library}")
        self.library = ctypes.CDLL(library)
        self.launch = self.library.AccOffloadSparseCopy
        self.launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint8]
        self.launch.restype = None
        self.native_launch = getattr(self.library, "AccOffloadNativeCopy", None)
        if self.native_launch is not None:
            self.native_launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint8]
            self.native_launch.restype = None
        version = getattr(self.library, "AccOffloadNativeCopyPlanVersion", None)
        if version is not None:
            version.argtypes = []
            version.restype = ctypes.c_uint32
        self.native_plan_version = version() if version is not None else 1
        if self.native_plan_version not in (1, 2):
            raise ValueError("Unsupported native expert-copy recipe version")
        self.acl = ctypes.CDLL("libascendcl.so")
        self.acl.aclrtHostRegister.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.acl.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
        self.owners = []
        self.registered = []

    def map_host(self, tensor):
        return self.map_hosts((tensor,))[0]

    def map_hosts(self, tensors):
        page_size = os.sysconf("SC_PAGE_SIZE")
        ranges = []
        for index, tensor in enumerate(tensors):
            pointer = tensor.data_ptr()
            end = pointer + tensor.numel() * tensor.element_size()
            ranges.append((pointer // page_size * page_size,
                           (end + page_size - 1) // page_size * page_size, index))
        groups = []
        for start, end, index in sorted(ranges):
            if groups and start < groups[-1][1]:
                groups[-1][1] = max(groups[-1][1], end)
                groups[-1][2].append(index)
            else:
                groups.append([start, end, [index]])
        mapped = [None] * len(tensors)
        try:
            for start, end, indices in groups:
                pointer = ctypes.c_void_p()
                ret = self.acl.aclrtHostRegister(start, end - start, 0, ctypes.byref(pointer))
                if ret:
                    raise RuntimeError(f"NPU expert host registration failed: ACL {ret}")
                self.registered.append(start)
                for index in indices:
                    mapped[index] = pointer.value + tensors[index].data_ptr() - start
            self.owners.extend(tensors)
        except Exception:
            self.close()
            raise
        return mapped

    def copy(self, sources, destinations, lengths, count):
        self.launch(sources.data_ptr(), destinations.data_ptr(), lengths.data_ptr(),
                    count.data_ptr(), sources.device.index)

    def close(self):
        for pointer in self.registered:
            ret = self.acl.aclrtHostUnregister(pointer)
            if ret:
                raise RuntimeError(f"NPU expert host unregister failed: ACL {ret}")
        self.registered.clear()
        self.owners.clear()


class DeviceExpertTransfer:
    """One parallel kernel: mapped DRAM -> final NZ weights and scale slots."""

    device_owned = True

    def __init__(self, manager, layer):
        from vllm_ascend.expert_offload.expert_offload_manager import _expert_weight

        if not manager.native_mxfp4_enabled or manager.ep_size != 1:
            raise ValueError("NPU transfer scheduler requires native MXFP4 and EP size one")
        if getattr(manager, "exclusive_dynamic_enabled", False):
            raise ValueError("NPU transfer scheduler requires replicated CPU experts")
        self.manager, self.layer = manager, layer
        self.layer_idx = manager.moe_layers.index(layer)
        owner = manager._native_owners[self.layer_idx]
        self.device = layer.w13_weight_scale.device
        slots = manager.num_device_experts_for_layer(self.layer_idx)
        if slots < 1:
            raise ValueError("NPU transfer scheduler requires at least one device slot")
        config = manager.offload_config.cpu_moe
        self.policy = DeviceExpertPolicy(owner.gate.shape[0], slots, self.device, config.transfer_policy)
        self.copier = DeviceSparseCopy(config.transfer_copy_library)
        if self.copier.native_launch is None:
            raise ValueError("Rebuild the expert copy library: AccOffloadNativeCopy is required")
        w13, w2 = _expert_weight(layer, "w13_weight"), _expert_weight(layer, "w2_weight")
        intermediate = owner.gate.shape[1]
        hidden = owner.down.shape[1]
        destinations = (w13, w13, w2, layer.w13_weight_scale,
                        layer.w13_weight_scale, layer.w2_weight_scale)
        target_rows = (2 * intermediate, 2 * intermediate, hidden,
                       2 * intermediate, 2 * intermediate, hidden)
        row_offsets = (0, intermediate, 0, 0, intermediate, 0)
        slot_bytes = (manager.w13_expert_size_bytes, manager.w13_expert_size_bytes,
                      manager.w2_expert_size_bytes,
                      layer.w13_weight_scale[0].numel(), layer.w13_weight_scale[0].numel(),
                      layer.w2_weight_scale[0].numel())
        recipes = []
        mapped_sources = self.copier.map_hosts(owner.tensors)
        for field, (source, destination, rows, offset, stride) in enumerate(zip(
                owner.tensors, destinations, target_rows, row_offsets, slot_bytes)):
            source_rows, row_bytes = source.shape[1:]
            if source.element_size() != 1 or destination.element_size() != 1:
                raise ValueError("Fused native transfer requires byte-sized weights/scales")
            if rows * row_bytes != stride:
                raise ValueError("Unsupported padded expert slot layout")
            width = 16 if field < 3 else self._scale_width(destination, rows, row_bytes)
            if row_bytes % width or ((width + 31) // 32) * 32 * 256 > 128 * 1024:
                raise ValueError("Unsupported native expert row width")
            recipes.append([mapped_sources[field], destination.data_ptr(),
                            source_rows * row_bytes, stride, source_rows, row_bytes,
                            rows, offset, width])
        self._gather_offsets = []
        if self.copier.native_plan_version == 2:
            for recipe in recipes:
                row_bytes, width = recipe[5], recipe[8]
                if recipe[4] % 16 or row_bytes > 2048:
                    raise ValueError("Tiled native copy requires 16-row tiles and row bytes <= 2048")
                if width == row_bytes:
                    recipe.append(0)
                    continue
                item_bytes = 2 if width == 2 else 4
                chunk_items = width // item_bytes
                index = torch.arange(16 * row_bytes // item_bytes, dtype=torch.int64)
                offset = ((index // (16 * chunk_items)) * chunk_items
                          + (index // chunk_items % 16) * (row_bytes // item_bytes)
                          + index % chunk_items) * item_bytes
                offsets = offset.to(device=self.device, dtype=torch.int32)
                self._gather_offsets.append(offsets)
                recipe.append(offsets.data_ptr())
        self.recipes = torch.tensor(recipes, device=self.device, dtype=torch.int64)
        self._plans = []
        if not hasattr(manager, "_device_expert_transfers"):
            manager._device_expert_transfers = []
        manager._device_expert_transfers.append(self)

    @staticmethod
    def _scale_width(destination, rows, columns):
        shape = tuple(destination[0].shape)
        stride = tuple(destination[0].stride())
        if shape != (columns // 2, rows, 2):
            raise ValueError(f"Unsupported native scale shape: {shape}")
        if stride == (2, columns, 1):
            return columns
        if stride == (rows * 2, 2, 1):
            return 2
        raise ValueError(f"Unsupported native scale strides: {stride}")

    def prepare(self, ids, mapping):
        candidate, slot, active = self.policy.select(ids, mapping)
        task = _DeviceUpload(self, candidate, slot, active)
        if task.graph:
            self._plans.append(task)
        else:
            self._last_eager_plan = task
        return task

    def upload(self, task):
        self.copier.native_launch(
            self.recipes.data_ptr(), task.candidate.data_ptr(), task.slot.data_ptr(),
            task.active.data_ptr(), self.device.index)
        if hasattr(self.layer, "w13_weight_scale_fp32"):
            self.layer.w13_weight_scale_fp32.copy_(self.layer.w13_weight_scale.to(torch.float32))


class _DeviceUpload:
    def __init__(self, transfer, candidate, slot, active):
        from vllm_ascend.expert_offload.cpu_backend.kt_kernel_backend import _graph_capture_active

        self.transfer = transfer
        self.candidate, self.slot, self.active = candidate, slot, active
        self.graph = _graph_capture_active()

    def wait_for_weights(self, cpu_task):
        # No CPU submission event or host callback gates this upload.
        self.transfer.upload(self)
