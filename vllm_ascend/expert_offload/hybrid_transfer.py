"""Transfer one CPU miss while the other hybrid experts compute on CPU."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch

from vllm_ascend.expert_offload.cpu_backend.kt_kernel_backend import _eager_trace
from vllm_ascend.expert_offload.frequency_transfer_policy import FrequencyTransferPolicy

from vllm_ascend.expert_offload.cpu_backend.transfer_diagnostics import (
    TransferDiagnostics, inspect_route_buffers,
)

CPU_EXPERT_TRANSFER_MIN_MISSES = 1


def select_transfer_expert(topk_ids, log2phy, slot_count):
    """Pick the first distinct miss and a slot unused by ANY current route.

    Any CPU miss can trigger one transfer. Duplicate routes count once. If every
    slot is needed now, preserve the CPU path rather than evict an active
    resident expert. The incoming expert remains resident for future steps.
    """
    routed = list(dict.fromkeys(int(e) for e in topk_ids.reshape(-1).tolist()
                                if int(e) >= 0))
    misses = [e for e in routed if int(log2phy[e]) < 0]
    if len(misses) < CPU_EXPERT_TRANSFER_MIN_MISSES:
        return None
    protected_slots = {int(log2phy[e]) for e in routed
                       if int(log2phy[e]) >= 0}
    for slot in range(slot_count):
        if slot not in protected_slots:
            return misses[0], slot
    return None

@dataclass
class _TransferState:
    ids_h: torch.Tensor
    map_h: torch.Tensor
    input_ready: object
    decision_ready: object
    stream: object
    graph: bool
    load_and_sync: object
    selection: tuple | None = None

    def wait_for_weights(self, cpu_task):
        # Submit CPU work first. Reuse offload's H2D + synchronize on a host
        # callback in the main stream, so subsequent NPU kernels are gated.
        if self.graph:
            from vllm_ascend.expert_offload.cpu_backend.kt_kernel_backend import _ensure_subscribed
            stream = torch.npu.current_stream()
            stream.wait_event(cpu_task.submitted_event)
            _ensure_subscribed(stream)
            torch.npu._launch_host_func(stream, self.load_and_sync, self)
        else:
            self.load_and_sync(self)


class HybridExpertTransfer:
    """Stage routing on a shared stream; reuse offload's H2D synchronization.

    Graph mode stages full routing and submits CPU-owned routes on the host.
    One readiness event covers the updated mapping and synchronized weights.
    CPU work runs asynchronously during H2D and NPU expert execution. Its
    blocking join is deferred until the NPU expert phase has completed.
    Every replay re-evaluates current routing using the configured policy.
    """

    def __init__(self, manager, layer):
        if getattr(manager, "exclusive_dynamic_enabled", False):
            raise ValueError(
                "cpu_moe.transfer_one_expert requires replicated CPU weights")
        self.manager = manager
        self.layer = layer
        self.layer_idx = manager.moe_layers.index(layer)
        self.transfer_policy = getattr(manager.offload_config.cpu_moe, "transfer_policy", "first_miss")
        self._frequency_policy = FrequencyTransferPolicy() if self.transfer_policy == "frequency" else None
        self._states = []  # Retain pinned addresses/events baked into graphs.
        if not hasattr(manager, "_cpu_transfer_diagnostics"):
            manager._cpu_transfer_diagnostics = TransferDiagnostics.configured()
        self.diagnostics = manager._cpu_transfer_diagnostics

    def _scheduling_stream(self, device):
        # One stream per manager/device; capture buffers/events remain private.
        streams = getattr(self.manager, "_hybrid_transfer_streams", None)
        if streams is None:
            streams = {}
            self.manager._hybrid_transfer_streams = streams
        index = device.index
        if index is None:
            index = torch.npu.current_device()
        if index not in streams:
            streams[index] = torch.npu.Stream(device=index)
        return streams[index]

    def _enqueue(self, state):
        selector = self._frequency_policy.select if self._frequency_policy is not None else select_transfer_expert
        record = getattr(state, "_diagnostic_record", None)
        if record:
            record.mark("selector_begin")
        state.selection = selector(
            state.ids_h, state.map_h,
            self.manager.num_device_experts_for_layer(self.layer_idx))
        if record:
            record.mark("selector_end", selection=state.selection)
        if state.selection is None:
            return
        eid, slot = state.selection
        state.map_h[state.map_h == slot] = -1
        state.map_h[eid] = slot
        if record:
            record.mark("map_update_end")

    def capture_cpu_and_weights(self, backend, hidden_states, log2phy,
                                cpu_stream, weights_ready):
        """Capture one CPU decision/submission/weight phase on the CPU stream.

        The backend already staged full routing and activations. The callback
        masks CPU routes using the updated host map before submitting work.
        The main stream only consumes the map after both map and weights land.
        """
        ids_h, weights_h = backend._pinned_route_buffers(hidden_states)
        with torch.inference_mode(False):
            map_h = torch.empty(log2phy.shape, dtype=log2phy.dtype,
                                device="cpu", pin_memory=True)
        state = _TransferState(ids_h, map_h, None, weights_ready,
                               cpu_stream, True, self._load_and_sync)
        self._states.append(state)
        if self.diagnostics:
            state._diagnostic_capture = dict(
                state_id=id(state), stream_handle=int(cpu_stream.npu_stream),
                load_stream_handle=int(self.manager.load_stream.npu_stream),
                staged_ids_ptr=ids_h.data_ptr(), staged_weights_ptr=weights_h.data_ptr(),
                map_ptr=map_h.data_ptr(), buffers=inspect_route_buffers(backend, hidden_states))
            state._diagnostic_replay = 0
        map_h.copy_(log2phy, non_blocking=True)
        torch.npu._launch_host_func(
            cpu_stream, self._prepare_and_launch_cpu,
            (state, backend, hidden_states, weights_h))
        log2phy.copy_(map_h, non_blocking=True)
        cpu_stream.record_event(weights_ready)

    @torch.inference_mode()
    def _prepare_and_launch_cpu(self, args):
        state, backend, hidden_states, weights_h = args
        record = None
        error = None
        if self.diagnostics:
            state._diagnostic_replay += 1
            record = self.diagnostics.begin(
                self.layer_idx, state._diagnostic_capture, state._diagnostic_replay)
            state._diagnostic_record = record
        try:
            if record:
                record.mark("route_snapshot_before_begin")
                record.data["routes_before"] = dict(
                    ids=state.ids_h.tolist(), weights=weights_h.tolist(),
                    logical_to_physical=state.map_h.tolist(),
                    current_buffers=inspect_route_buffers(backend, hidden_states))
                record.mark("route_snapshot_before_end")
                record.mark("enqueue_begin")
            self._enqueue(state)
            if record:
                record.mark("enqueue_end", selection=state.selection)
                record.mark("cpu_route_mask_begin")
            ids = state.ids_h
            valid = (ids >= 0) & (ids < state.map_h.numel()) & (weights_h != 0)
            safe_ids = ids.clamp(min=0, max=state.map_h.numel() - 1).long()
            cpu_owned = valid & (state.map_h[safe_ids] < 0)
            ids.masked_fill_(~cpu_owned, -1)
            weights_h.masked_fill_(~cpu_owned, 0)
            if record:
                record.mark("cpu_route_mask_end")
                record.mark("route_snapshot_after_begin")
                record.data["routes_after"] = dict(
                    ids=ids.tolist(), weights=weights_h.tolist(),
                    logical_to_physical=state.map_h.tolist())
                record.mark("route_snapshot_after_end")
                backend._launch_pinned_forward(
                    (hidden_states, int(state.stream.npu_stream)), diagnostics=record)
            else:
                backend._launch_pinned_forward(
                    (hidden_states, int(state.stream.npu_stream)))
            self._load_and_sync(state)
        except Exception as exc:
            error = exc
            raise
        finally:
            if record:
                record.finish(error)
                state._diagnostic_record = None

    def _load_and_sync(self, state):
        record = getattr(state, "_diagnostic_record", None)
        if record:
            record.mark("weight_phase_begin")
        tracing = bool(os.environ.get("VLLM_ASCEND_CPU_MOE_TRACE"))
        if tracing:
            callback_begin_ns = time.perf_counter_ns()
            callback_begin_wall_ns = time.time_ns()
        if state.selection is not None:
            eid, slot = state.selection
            if tracing:
                load_begin_ns = time.perf_counter_ns()
            transport = self.manager._get_h2d_transport() if record else None
            if record:
                transport.diagnostic_record = record
                if hasattr(transport, "base"):
                    transport.base.diagnostic_record = record
                record.mark("load_stream_context_begin")
            try:
                with torch.npu.stream(self.manager.load_stream):
                    if record:
                        record.mark("weight_task_build_dispatch_begin", expert=eid, slot=slot)
                    self.manager._load_expert_weights_into_slot(
                        self.layer, self.layer_idx, eid, slot)
                    if record:
                        record.mark("weight_task_build_dispatch_end")
            finally:
                if record:
                    transport.diagnostic_record = None
                    if hasattr(transport, "base"):
                        transport.base.diagnostic_record = None
            if tracing:
                load_end_ns = time.perf_counter_ns()
            if record:
                record.mark("h2d_synchronize_begin")
            self.manager._synchronize_h2d()
            if record:
                record.mark("h2d_synchronize_end")
            if tracing:
                sync_end_ns = time.perf_counter_ns()
                # One record after all measured phases; no file writes are
                # inserted between weight enqueue and synchronization.
                _eager_trace(
                    "cpu_moe.transfer_weights", layer=self.layer_idx,
                    expert=eid, slot=slot,
                    callback_begin_ns=callback_begin_ns,
                    callback_begin_wall_ns=callback_begin_wall_ns,
                    load_begin_ns=load_begin_ns, load_end_ns=load_end_ns,
                    sync_end_ns=sync_end_ns,
                    load_duration_ns=load_end_ns-load_begin_ns,
                    sync_duration_ns=sync_end_ns-load_end_ns,
                    callback_work_duration_ns=sync_end_ns-callback_begin_ns,
                )
        elif tracing:
            _eager_trace("cpu_moe.transfer_skipped", layer=self.layer_idx,
                         callback_begin_ns=callback_begin_ns,
                         callback_begin_wall_ns=callback_begin_wall_ns)

        if record:
            record.mark("weight_phase_end", selection=state.selection)

    def prepare(self, topk_ids, log2phy):
        from vllm_ascend.expert_offload.cpu_backend.kt_kernel_backend import (
            _ensure_subscribed,
            _graph_capture_active,
        )

        if topk_ids.device.type != "npu":
            raise ValueError("hybrid expert transfer requires NPU routing")
        graph = _graph_capture_active()
        compute_stream = torch.npu.current_stream(topk_ids.device)
        transfer_stream = self._scheduling_stream(topk_ids.device)
        # Normal mutable tensors are required for replay callbacks even if
        # model warmup/capture runs under inference_mode.
        with torch.inference_mode(False):
            ids_h = torch.empty(topk_ids.shape, dtype=topk_ids.dtype,
                                device="cpu", pin_memory=True)
            map_h = torch.empty(log2phy.shape, dtype=log2phy.dtype,
                                device="cpu", pin_memory=True)
        state = _TransferState(
            ids_h, map_h, torch.npu.Event(), torch.npu.Event(),
            transfer_stream, graph, self._load_and_sync)
        if not graph:
            ids_h.copy_(topk_ids)
            map_h.copy_(log2phy)
            self._enqueue(state)
            log2phy.copy_(map_h, non_blocking=True)
            return state

        self._states.append(state)
        compute_stream.record_event(state.input_ready)
        with torch.npu.stream(transfer_stream):
            transfer_stream.wait_event(state.input_ready)
            ids_h.copy_(topk_ids, non_blocking=True)
            map_h.copy_(log2phy, non_blocking=True)
            _ensure_subscribed(transfer_stream)
            torch.npu._launch_host_func(transfer_stream, self._enqueue, state)
            log2phy.copy_(map_h, non_blocking=True)
            transfer_stream.record_event(state.decision_ready)
            # The main-stream load/sync callback is registered only after
            # the reduced CPU task has been submitted.
        compute_stream.wait_event(state.decision_ready)
        return state
