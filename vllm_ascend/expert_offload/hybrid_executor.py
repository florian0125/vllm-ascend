"""CPU/NPU hybrid routed-expert execution contracts.

This module owns only the device split and task lifecycle.  CPU weight loading
and CPU kernels live behind :class:`CPUExpertBackend` and can be implemented
independently of the Ascend MoE pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class CPUExpertTask(Protocol):
    """One submitted CPU routed-expert computation.

    ``wait`` returns a token-major, route-weighted and expert-reduced tensor
    with shape ``[num_tokens, hidden_size]``.  The tensor may be on CPU or NPU.
    """

    def wait(self) -> torch.Tensor:
        """Wait for CPU computation and return its routed-expert output."""
        ...


@runtime_checkable
class CPUExpertBackend(Protocol):
    """Backend implemented by a concrete CPU MoE kernel integration."""

    @property
    def max_num_tokens(self) -> int:
        """Largest token batch accepted by one submitted task."""
        ...

    def submit(
        self,
        *,
        layer_idx: int,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        defer_graph_join: bool = False,
        graph_transfer=None,
    ) -> CPUExpertTask:
        """Submit the CPU-owned routes without blocking NPU execution.

        ``topk_ids`` contains global logical expert ids for CPU routes and -1
        for routes owned by NPU.  The backend must skip negative ids.  It owns
        any D2H transfer and must apply each non-zero routing weight exactly
        once before reducing expert outputs into token-major order.
        With ``defer_graph_join``, submit records a submission event but leaves
        graph drain/output copies to wait(), after NPU work has been recorded.
        This must never defer the CPU computation itself.
        ``graph_transfer`` optionally supplies a transfer coordinator and
        device map. That path receives full routes, masks them on CPU using
        the updated map, and records a combined map/weight readiness wait.
        """
        ...


@dataclass(frozen=True, slots=True)
class HybridExpertPlan:
    """NPU-safe routing plus the concurrently submitted CPU task."""

    npu_topk_ids: torch.Tensor
    npu_topk_weights: torch.Tensor
    cpu_topk_ids: torch.Tensor
    cpu_topk_weights: torch.Tensor
    cpu_task: CPUExpertTask
    stats: dict | None = None


class HybridExpertExecutor:
    """Split selected routes according to the current NPU resident map."""

    def __init__(self, backend: CPUExpertBackend, layer_idx: int) -> None:
        if not isinstance(backend, CPUExpertBackend):
            raise TypeError("backend does not implement CPUExpertBackend")
        if layer_idx < 0:
            raise ValueError(f"layer_idx must be non-negative; got {layer_idx}")
        self.backend = backend
        self.layer_idx = layer_idx
        self._weight_transfer = None

    def prepare(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        log2phy: torch.Tensor,
        weight_transfer=None,
        transfer_manager=None,
    ) -> HybridExpertPlan:
        """Submit CPU misses and produce a safe zero-weight NPU route.

        Valid logical ids remain unchanged. Pruned/zero-weight routes are
        marked -1 before any mapping lookup or optional weight transfer.
        The existing dispatcher excludes unmapped routes; their NPU weights
        are also zero. The CPU backend skips -1 routes unconditionally.
        """
        if topk_ids.shape != topk_weights.shape:
            raise ValueError(
                "topk ids and weights must have the same shape; "
                f"got ids={tuple(topk_ids.shape)}, "
                f"weights={tuple(topk_weights.shape)}"
            )
        if topk_ids.ndim != 2:
            raise ValueError(f"topk routing must be rank 2; got rank={topk_ids.ndim}")
        if log2phy.ndim != 1:
            raise ValueError(f"log2phy must be rank 1; got rank={log2phy.ndim}")

        valid = (topk_ids >= 0) & (topk_ids < log2phy.numel()) & (topk_weights != 0)
        topk_ids = torch.where(valid, topk_ids, -1)
        if log2phy.numel() == 0:
            raise ValueError("log2phy must not be empty")
        if topk_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError(
                f"topk ids must use a signed integer dtype so -1 can mark NPU-owned routes; got {topk_ids.dtype}"
            )
        cpu_task = None
        transfer_task = None
        if weight_transfer is not None:
            from vllm_ascend.expert_offload.cpu_backend.kt_kernel_backend import _graph_capture_active

            if getattr(weight_transfer, "device_owned", False):
                transfer_task = weight_transfer.prepare(topk_ids, log2phy, topk_weights)
            elif _graph_capture_active():
                # CPU selects placement and submits the reduced CPU routes.
                # This returns after recording one map+weights dependency;
                # device routing below therefore observes the new mapping.
                cpu_task = self.backend.submit(
                    layer_idx=self.layer_idx,
                    hidden_states=hidden_states,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                    defer_graph_join=True,
                    graph_transfer=(weight_transfer, log2phy),
                )
            else:
                transfer_task = weight_transfer.prepare(topk_ids, log2phy)
        valid_routes = (topk_ids >= 0) & (topk_ids < log2phy.numel()) & (topk_weights != 0)
        safe_ids = topk_ids.clamp(min=0, max=log2phy.numel() - 1).long()
        physical_ids = log2phy[safe_ids]
        npu_route_mask = valid_routes & (physical_ids >= 0)
        cpu_route_mask = valid_routes & (physical_ids < 0)
        raw_hit_mask = valid_routes & (physical_ids >= 0)
        zero = torch.zeros((), dtype=topk_weights.dtype, device=topk_weights.device)
        npu_topk_weights = torch.where(npu_route_mask, topk_weights, zero)
        cpu_topk_weights = torch.where(cpu_route_mask, topk_weights, zero)
        cpu_topk_ids = torch.where(
            cpu_route_mask,
            topk_ids,
            torch.full((), -1, dtype=topk_ids.dtype, device=topk_ids.device),
        )

        # Carry the admission count in already-skipped CPU route sentinels.
        # A newly admitted expert has at least one NPU-owned route, so T1
        # always has a negative CPU route available. No additional D2H.
        import os
        if (os.environ.get("KT_MOE_CPU_TRACE") and weight_transfer is not None
                and getattr(weight_transfer, "device_owned", False)):
            sentinel = -1 - weight_transfer.trace_transfer_count
            cpu_topk_ids = torch.where(cpu_topk_ids < 0, sentinel.to(cpu_topk_ids.dtype), cpu_topk_ids)

        submission_options = {}
        if (transfer_task is not None and transfer_task.graph
                and not getattr(weight_transfer, "device_owned", False)):
            submission_options["defer_graph_join"] = True
        if cpu_task is None:
            cpu_task = self.backend.submit(
                layer_idx=self.layer_idx,
                hidden_states=hidden_states,
                topk_ids=cpu_topk_ids,
                topk_weights=cpu_topk_weights,
                **submission_options,
            )
        if transfer_task is not None:
            transfer_task.wait_for_weights(cpu_task)
        collector = getattr(transfer_manager, "_stats", None)
        stats = None
        if collector is not None and getattr(collector, "collecting", False):
            routed_experts = torch.unique(topk_ids[valid_routes])
            stats = {
                "cpu_experts": float(torch.unique(topk_ids[cpu_route_mask]).numel()),
                "npu_experts": float(torch.unique(topk_ids[npu_route_mask]).numel()),
                "cpu_routes": float(cpu_route_mask.sum()),
                "npu_routes": float(npu_route_mask.sum()),
                "raw_hit": float(torch.unique(topk_ids[raw_hit_mask]).numel()) / max(float(routed_experts.numel()), 1.0),
            }
        return HybridExpertPlan(
            npu_topk_ids=topk_ids,
            npu_topk_weights=npu_topk_weights,
            cpu_topk_ids=cpu_topk_ids,
            cpu_topk_weights=cpu_topk_weights,
            cpu_task=cpu_task,
            stats=stats,
        )

    @staticmethod
    def finish(
        npu_routed_out: torch.Tensor,
        cpu_task: CPUExpertTask,
    ) -> torch.Tensor:
        """Wait for CPU output, validate its contract and merge on NPU."""
        cpu_routed_out = cpu_task.wait()
        if not isinstance(cpu_routed_out, torch.Tensor):
            raise TypeError("CPU expert task must return a torch.Tensor")
        if cpu_routed_out.shape != npu_routed_out.shape:
            raise ValueError(
                "CPU and NPU routed outputs must have the same shape; "
                f"cpu={tuple(cpu_routed_out.shape)}, "
                f"npu={tuple(npu_routed_out.shape)}"
            )
        if getattr(cpu_task, "host_mapped_result", False):
            from .cpu_backend.mapped_combine import mapped_combine
            return mapped_combine(npu_routed_out, cpu_routed_out)
        cpu_routed_out = cpu_routed_out.to(
            device=npu_routed_out.device,
            dtype=npu_routed_out.dtype,
            non_blocking=cpu_routed_out.device.type == "cpu",
        )
        return npu_routed_out + cpu_routed_out


class PreparedHybridRoutes:
    """Phase-A result of hybrid route preparation.

    Phase A (``begin_hybrid_routes``) finishes the pending prefetch join and
    preprocesses routes (substitution / pruning / transfer handle), but does
    NOT submit the current layer's CPU MoE task yet. The caller can therefore
    fire the next-layer prefetch signal (prediction + selector + notify)
    between the two phases, so the prefetch chain starts before the current
    layer's CPU D2H copies and ready-flag publish.

    Phase B (``finish``) performs the CPU input copies and the backend submit
    exactly as the previous single-shot path did.
    """

    def __init__(self, executor, *, hidden_states, topk_ids, topk_weights,
                 log2phy, weight_transfer, transfer_manager):
        self._executor = executor
        self._hidden_states = hidden_states
        self._topk_ids = topk_ids
        self._topk_weights = topk_weights
        self._log2phy = log2phy
        self._weight_transfer = weight_transfer
        self._transfer_manager = transfer_manager

    def finish(self) -> HybridExpertPlan:
        return self._executor.prepare(
            hidden_states=self._hidden_states,
            topk_ids=self._topk_ids,
            topk_weights=self._topk_weights,
            log2phy=self._log2phy,
            weight_transfer=self._weight_transfer,
            transfer_manager=self._transfer_manager,
        )


def begin_hybrid_routes(
    layer: torch.nn.Module,
    *,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    log2phy: torch.Tensor,
    transfer_manager=None,
    router_logits=None,
    scoring_func="softmax",
    e_score_correction_bias=None,
    is_hash_routed=False,
) -> PreparedHybridRoutes | None:
    """Phase A: prefetch join + route preprocessing, CPU submit deferred.

    Returns None exactly where the previous single-shot path returned None
    (no executor, profile run, over token capacity). Otherwise returns a
    ``PreparedHybridRoutes`` whose ``finish()`` submits the CPU MoE task.

    Attached executors handle normal forwards within their token capacity.
    Startup memory profiling alone retains the existing NPU path.
    Larger batches retain the existing NPU paging/layerwise path.
    """
    executor = get_hybrid_expert_executor(layer)
    if executor is None:
        return None
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if is_forward_context_available() and getattr(get_forward_context(), "in_profile_run", False):
        return None
    backend_max_tokens = executor.backend.max_num_tokens
    if backend_max_tokens <= 0:
        raise ValueError(f"CPU expert backend max_num_tokens must be positive; got {backend_max_tokens}")
    if hidden_states.shape[0] > backend_max_tokens or topk_ids.shape[0] > backend_max_tokens:
        return None
    # The hybrid path replaces update_weights, so it must perform the same
    # prefetch join before reading or modifying the device residency map.
    if transfer_manager is not None:
        import torch_npu
        transfer_manager._finish_pending_predict(executor.layer_idx)
        with transfer_manager._prefetch_state_lock:
            pending_event = transfer_manager._prefetch_layer_npu_event.pop(executor.layer_idx, None)
        if pending_event is not None:
            torch_npu.npu.current_stream().wait_event(pending_event)

    # Reuse the paging route substitution before deciding CPU/NPU ownership.
    # Hash-routed layers are intentionally excluded, just as in update_weights.
    if (transfer_manager is not None
            and transfer_manager.offload_config.expert_substitution_enabled
            and not is_hash_routed and router_logits is not None
            and router_logits.shape[-1] == transfer_manager.num_total_experts):
        from vllm.logger import logger
        logger.info_once("[HYBRID-STACK] NPU substitution enabled threshold=%s",
                         transfer_manager.offload_config.expert_substitution_threshold)
        from vllm_ascend.ops.fused_moe.experts_selector import substitute_experts_device
        substituted_ids = substitute_experts_device(
            router_logits, topk_ids, log2phy,
            expert_substitution_threshold=transfer_manager.offload_config.expert_substitution_threshold,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
        )
        topk_ids.copy_(substituted_ids)

    # Threshold pruning belongs to routing, not to the paging operation that
    # hybrid execution replaces. Use the final pruned routes for both devices.
    if transfer_manager is not None and getattr(transfer_manager.offload_config, "experts_pruning_enabled", False):
        valid = (topk_ids >= 0) & (topk_ids < log2phy.numel())
        topk_ids = torch.where(valid, topk_ids, -1)
        topk_weights = torch.where(valid, topk_weights, 0)
        from vllm_ascend.ops.fused_moe.experts_selector import maybe_prune_topk_experts

        topk_weights, topk_ids, _ = maybe_prune_topk_experts(topk_weights, topk_ids, log2phy=log2phy)
    weight_transfer = None
    if transfer_manager is not None and transfer_manager.offload_config.cpu_moe.transfer_one_expert:
        if executor._weight_transfer is None:
            from vllm_ascend.expert_offload.hybrid_transfer import (
                HybridExpertTransfer,
            )

            if getattr(transfer_manager.offload_config.cpu_moe, "transfer_scheduler", "cpu") == "npu":
                from vllm_ascend.expert_offload.device_transfer import DeviceExpertTransfer
                executor._weight_transfer = DeviceExpertTransfer(transfer_manager, layer)
            else:
                executor._weight_transfer = HybridExpertTransfer(transfer_manager, layer)
        weight_transfer = executor._weight_transfer
    return PreparedHybridRoutes(
        executor,
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        log2phy=log2phy,
        weight_transfer=weight_transfer,
        transfer_manager=transfer_manager,
    )


def maybe_prepare_hybrid_routes(
    layer: torch.nn.Module,
    *,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    log2phy: torch.Tensor,
    transfer_manager=None,
    router_logits=None,
    scoring_func="softmax",
    e_score_correction_bias=None,
    is_hash_routed=False,
) -> HybridExpertPlan | None:
    """Backward-compatible single-shot path: begin + immediate finish."""
    prepared = begin_hybrid_routes(
        layer,
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        log2phy=log2phy,
        transfer_manager=transfer_manager,
        router_logits=router_logits,
        scoring_func=scoring_func,
        e_score_correction_bias=e_score_correction_bias,
        is_hash_routed=is_hash_routed,
    )
    if prepared is None:
        return None
    return prepared.finish()


def attach_hybrid_expert_executor(
    layer: torch.nn.Module,
    backend: CPUExpertBackend,
    layer_idx: int,
) -> HybridExpertExecutor:
    """Attach a CPU backend to one routed-expert layer.

    A future CPU kernel integration calls this after its layer weights are
    ready.  Until then the layer has no executor and existing offload behavior
    remains unchanged.
    """
    executor = HybridExpertExecutor(backend, layer_idx)
    layer._hybrid_expert_executor = executor
    return executor


def get_hybrid_expert_executor(
    layer: torch.nn.Module,
) -> HybridExpertExecutor | None:
    """Return a previously attached executor, if CPU compute is available."""
    executor = getattr(layer, "_hybrid_expert_executor", None)
    if executor is not None and not isinstance(executor, HybridExpertExecutor):
        raise TypeError("layer._hybrid_expert_executor has an invalid type")
    return executor


__all__ = [
    "CPUExpertBackend",
    "CPUExpertTask",
    "HybridExpertExecutor",
    "HybridExpertPlan",
    "PreparedHybridRoutes",
    "attach_hybrid_expert_executor",
    "begin_hybrid_routes",
    "get_hybrid_expert_executor",
    "maybe_prepare_hybrid_routes",
]
