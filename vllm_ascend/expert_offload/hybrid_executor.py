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
                transfer_task = weight_transfer.prepare(topk_ids, log2phy)
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
        zero = torch.zeros((), dtype=topk_weights.dtype, device=topk_weights.device)
        npu_topk_weights = torch.where(npu_route_mask, topk_weights, zero)
        cpu_topk_weights = torch.where(cpu_route_mask, topk_weights, zero)
        cpu_topk_ids = torch.where(
            cpu_route_mask,
            topk_ids,
            torch.full((), -1, dtype=topk_ids.dtype, device=topk_ids.device),
        )

        submission_options = {}
        # Only host-driven weight transfers need the callback dispatcher free
        # until their weight callback has run. Device-owned copies have no host
        # weight callback: drain CPU work and return its output concurrently with
        # NPU work instead of scheduling a second callback after npu_done.
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
        return HybridExpertPlan(
            npu_topk_ids=topk_ids,
            npu_topk_weights=npu_topk_weights,
            cpu_topk_ids=cpu_topk_ids,
            cpu_topk_weights=cpu_topk_weights,
            cpu_task=cpu_task,
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
        cpu_routed_out = cpu_routed_out.to(
            device=npu_routed_out.device,
            dtype=npu_routed_out.dtype,
            non_blocking=cpu_routed_out.device.type == "cpu",
        )
        return npu_routed_out + cpu_routed_out


def maybe_prepare_hybrid_routes(
    layer: torch.nn.Module,
    *,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    log2phy: torch.Tensor,
    transfer_manager=None,
) -> HybridExpertPlan | None:
    """Split this layer's routes onto CPU and NPU, if a backend owns it.

    Attached executors handle normal forwards within their token capacity.
    Startup memory profiling alone retains the existing NPU path.
    Larger batches retain the existing NPU paging/layerwise path.
    A returned plan replaces paging, which must not rewrite the residency map
    after CPU/NPU ownership has been determined.
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
    return executor.prepare(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        log2phy=log2phy,
        weight_transfer=weight_transfer,
    )


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
    "attach_hybrid_expert_executor",
    "get_hybrid_expert_executor",
    "maybe_prepare_hybrid_routes",
]
