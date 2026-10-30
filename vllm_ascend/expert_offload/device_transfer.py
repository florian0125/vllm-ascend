"""Device-owned hybrid admission and mapped-DRAM expert uploads.

CPU receives only the final compute routes. Placement stays on device; one
parallel copy kernel moves preconverted NZ bytes into the selected HBM slot.
"""

import ctypes
import os
from pathlib import Path

import torch

from vllm_ascend.expert_offload.frequency_transfer_policy import (
    FREQUENCY_DECAY, FREQUENCY_MIN_SCORE,
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
            demand = self.scores.gather(0, candidate.reshape(1)).squeeze(0)
            victim = slot_scores.gather(0, slot.reshape(1)).squeeze(0)
            admit = ((demand >= FREQUENCY_MIN_SCORE)
                     & (demand > victim * FREQUENCY_REPLACEMENT_RATIO))
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
            end = pointer + (tensor.nbytes() if isinstance(tensor, torch.UntypedStorage) else tensor.numel() * tensor.element_size())
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

    def map_storages(self, tensors):
        """Register exact allocator bounds; never extend a pinned allocation.

        Small torch pinned blocks may occupy part of a page. Registering the
        rounded page crosses the allocator range and CANN rejects it.
        Tensor views share their owner's storage registration and add offsets.
        """
        registered = {}
        mapped = []
        try:
            for tensor in tensors:
                storage = tensor.untyped_storage() if isinstance(tensor, torch.Tensor) else tensor
                key = (storage.data_ptr(), storage.nbytes())
                if key not in registered:
                    pointer = ctypes.c_void_p()
                    ret = self.acl.aclrtHostRegister(key[0], key[1], 0, ctypes.byref(pointer))
                    if ret:
                        raise RuntimeError(f"NPU expert host storage registration failed: ACL {ret}")
                    registered[key] = pointer.value
                    self.registered.append(key[0])
                    self.owners.append(storage)
                mapped.append(registered[key] + tensor.data_ptr() - key[0])
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
    """Device-owned admission and byte-preserving copies of preconverted NZ.

    CPU GGUF stays independent. The device receives a table of mapped expert
    addresses once, then selects and copies without a host weight callback.
    """
    device_owned = True

    def __init__(self, manager, layer, *, initialize_policy=True):
        from vllm_ascend.expert_offload.expert_offload_manager import _expert_weight
        if manager.native_mxfp4_enabled or manager.ep_size != 1:
            raise ValueError("Preconverted NZ transfer requires GGUF and EP size one")
        self.manager, self.layer = manager, layer
        self.layer_idx = manager.moe_layers.index(layer)
        if any(len(buffers[self.layer_idx]) != manager.num_total_experts
               for buffers in (manager.w13_weights_cpu, manager.w2_weights_cpu)):
            raise ValueError("Preconverted NZ transfer requires all host experts")
        self.device = layer.w13_weight_scale.device
        slots = manager.num_device_experts_for_layer(self.layer_idx)
        config = manager.offload_config.cpu_moe
        self.load_balanced = config.transfer_policy == "load_balance"
        self.policy = None
        if initialize_policy:
            if self.load_balanced:
                from vllm_ascend.expert_offload.load_balance_transfer_policy import LoadBalanceTransferPolicy
                self.policy = LoadBalanceTransferPolicy(manager.num_total_experts, slots, self.device, config)
                if manager.offload_config.cache_policy_enabled:
                    seed_state = manager.cache_policy.layer_states[self.layer_idx]
                    if getattr(manager.offload_config, "cache_policy_type", "lrc") == "lru":
                        from vllm_ascend.expert_offload.device_lru_policy import DeviceLRUHotness
                        self.policy.lrc = DeviceLRUHotness(manager.num_total_experts, self.device,
                                                           manager.offload_config, seed_state)
                        from vllm.logger import logger
                        logger.info_once("[HYBRID-STACK] NPU transfer eviction=LRU; admission=load_balance")
                    else:
                        from vllm_ascend.expert_offload.device_lrc_policy import DeviceLRCHotness
                        self.policy.lrc = DeviceLRCHotness(manager.num_total_experts, self.device,
                                                          manager.offload_config, seed_state)
                        from vllm.logger import logger
                        logger.info_once("[HYBRID-STACK] NPU transfer eviction=LRC; admission=load_balance")
                    if not hasattr(manager, "_hybrid_lrc_hotness_h"):
                        manager._hybrid_lrc_hotness_h = {}
                    manager._hybrid_lrc_hotness_h[self.layer_idx] = torch.empty(
                        (2, manager.num_total_experts), dtype=torch.float32, pin_memory=True)

            else:
                self.policy = DeviceExpertPolicy(manager.num_total_experts, slots, self.device, config.transfer_policy)
        self.copier = DeviceSparseCopy(config.transfer_copy_library)
        w13, w2 = _expert_weight(layer, "w13_weight"), _expert_weight(layer, "w2_weight")
        self._packed_scales = []
        self._sources = []
        rows = []
        destinations = [w13.data_ptr(), w2.data_ptr()]
        self._field_names = ["w13_weight_NZ", "w2_weight_NZ"]
        sizes = [manager.w13_expert_size_bytes, manager.w2_expert_size_bytes]
        source_fields = [manager.w13_weights_cpu[self.layer_idx], manager.w2_weights_cpu[self.layer_idx]]
        # Tensor.copy_ writes logical scale values into destination strides.
        # Pack that same physical layout once on CPU, never during replay.
        for buffer_dict in (manager.scale_cpu_buffers, manager.offset_cpu_buffers, manager.scale_bias_cpu_buffers):
            for name, buffers in buffer_dict.items():
                target = getattr(layer, name, None)
                if target is None or self.layer_idx >= len(buffers):
                    continue
                dst = target[0]
                span = (1 + sum((n - 1) * stride for n, stride in zip(dst.shape, dst.stride()))) * dst.element_size()
                slot_stride = target.stride(0) * target.element_size()
                if span != dst.numel() * dst.element_size() or slot_stride != span:
                    raise ValueError(f"Unsupported padded scale slot: {name}")
                packed = torch.empty((manager.num_total_experts, span), dtype=torch.uint8, pin_memory=True)
                views = []
                for eid, src in enumerate(buffers[self.layer_idx]):
                    raw = packed[eid]
                    logical = raw.view(dst.dtype).as_strided(dst.shape, dst.stride())
                    logical.copy_(src.reshape(dst.shape))
                    views.append(raw)
                self._packed_scales.append(packed)
                source_fields.append(views)
                destinations.append(target.data_ptr())
                sizes.append(span)
                self._field_names.append(name)
        # Flatten registration across all experts/fields so shared allocator
        # pages are registered only once for this layer.
        flat = [source_fields[field][eid].untyped_storage() if field < 2 else source_fields[field][eid]
                for eid in range(manager.num_total_experts) for field in range(len(source_fields))]
        mapped = self.copier.map_storages(flat)
        for eid in range(manager.num_total_experts):
            rows.append(mapped[eid * len(source_fields):(eid + 1) * len(source_fields)])
        if len(destinations) % 2:
            destinations.append(destinations[-1]); sizes.append(0)
            self._field_names.append("padding_zero_length")
            for row in rows: row.append(row[-1])
        self.sources = torch.tensor(rows, dtype=torch.int64, device=self.device)
        self.destinations = torch.tensor(destinations, dtype=torch.int64, device=self.device)
        self.strides = torch.tensor(sizes, dtype=torch.int64, device=self.device)
        self.lengths = torch.tensor(sizes, dtype=torch.int32, device=self.device)
        self.count = torch.tensor(len(sizes), dtype=torch.int32, device=self.device)
        self.trace = None
        if getattr(config, "transfer_trace_capacity", 0):
            from vllm_ascend.expert_offload.transfer_trace import TransferTrace
            self.trace = TransferTrace(self, config.transfer_trace_capacity, config.transfer_trace_mode)
        self._plans = []
        self._sources = source_fields
        if not hasattr(manager, "_device_expert_transfers"):
            manager._device_expert_transfers = []
        manager._device_expert_transfers.append(self)

    def prepare(self, ids, mapping, weights=None):
        if self.policy is None:
            raise RuntimeError("Source-only expert transfer has no demand-admission policy")
        if self.trace is not None:
            self.trace.mark(0)
        if self.load_balanced:
            candidate, slot, active = self.policy.select(ids, mapping, weights)
        else:
            candidate, slot, active = self.policy.select(ids, mapping)
        self.trace_transfer_count = active.to(torch.int64).sum()
        if self.trace is not None:
            self.trace.mark(1, candidate, slot, active)
        if self.load_balanced:
            sources = self.sources.index_select(0, candidate).reshape(-1)
            destinations = (self.destinations[None, :] + slot[:, None] * self.strides[None, :]).reshape(-1)
            lengths = (self.lengths[None, :] * active[:, None].to(torch.int32)).reshape(-1)
            copy_count = active.to(torch.int32).sum().to(torch.int32) * self.lengths.numel()
        else:
            sources = self.sources.index_select(0, candidate.reshape(1)).reshape(-1)
            destinations = self.destinations + slot * self.strides
            lengths = self.lengths * active.to(torch.int32)
            copy_count = self.count
        if self.trace is not None:
            self.trace.mark(2)
        task = _DeviceUpload(self, sources, destinations, lengths, candidate, slot)
        task.copy_count = copy_count
        self._plans.append(task)
        return task

    def upload(self, task):
        if self.trace is None:
            self.copier.copy(task.sources, task.destinations, task.lengths, task.copy_count)
        else:
            self.trace.mark(3)
            self.trace.copy(task, self.count)
            self.trace.mark(4)
        if hasattr(self.layer, "w13_weight_scale_fp32"):
            self.layer.w13_weight_scale_fp32.copy_(self.layer.w13_weight_scale.to(torch.float32))


class _DeviceUpload:
    def __init__(self, transfer, sources, destinations, lengths, candidate=None, slot=None):
        from vllm_ascend.expert_offload.cpu_backend.kt_kernel_backend import _graph_capture_active
        self.transfer = transfer
        self.candidate, self.slot = candidate, slot
        self.sources, self.destinations, self.lengths = sources, destinations, lengths
        self.graph = _graph_capture_active()

    def wait_for_weights(self, cpu_task):
        # Device descriptors depend on device admission. No host submission
        # event or host callback is on the weight-transfer critical path.
        self.transfer.upload(self)
