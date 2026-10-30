"""Review candidate: one fate-prefetched expert with NPU LRC and a direct-copy host callback.

Scope is deliberately narrow: single-card hybrid GGUF, LRC, fate prediction,
expert_prefetch_num=1 and transfer_one_expert=False. The hybrid decode route
does not observe the existing host LRC state. Its exact victim order is frozen
after hot-expert preload, rather than adding a new observation policy as part
of this scheduling change. If a non-hybrid route observes that host state
(for example, a 9..32-token small prefill with backend max_num_tokens=8), the
caller must refresh the existing priority buffer before the next prefetch.
Transfer-off alone does not make state immutable for the server lifetime.

The caller owns the ordinary graph-captured fork/join edges::

    prediction on compute stream -> ready.record(compute)
    prefetch.wait_event(ready)
    with stream(prefetch):
        candidate.enqueue(predicted_ids, predicted_weights)
        done.record(prefetch)
    # At the target layer, before CPU/NPU ownership is computed:
    compute.wait_event(done)

The selector and LRC remain device-owned. Local Torch H2D uses pinned scalar
staging plus a stream-ordered host callback to submit direct Tensor.copy_
tasks on the independent load stream before committing the device mapping.
MemFabric and remote-source configurations retain sparse-copy. Construct after
expert weights/hot preload are final, before graph capture.
Unsupported configurations return None from try_create; the caller retains
the original host prefetch path. This module does not change serving code.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Sequence

try:
    import triton
    import triton.language as tl
except ImportError:  # The semantic reference/tests also run on the Windows host.
    triton = None
    tl = None


MAX_EXPERTS = 512
MAX_PREDICTED_ROUTES = 64


@dataclass(frozen=True)
class PrefetchDecision:
    candidate: int = 0
    victim: int = 0
    slot: int = 0
    active: bool = False


def frozen_victim_priority(heat: Sequence[float], last_used: Sequence[float]) -> list[int]:
    """Encode the host's exact (hotness, last_used, expert ID) ordering.

    Integer priorities avoid changing nearly tied host float scores through
    an fp32 conversion. They are valid only while that LRC state is frozen.
    """
    if len(heat) != len(last_used) or not heat:
        raise ValueError("heat and last_used need equal, nonzero lengths")
    if not all(isfinite(float(value)) for value in (*heat, *last_used)):
        raise ValueError("LRC seed scores must be finite")
    order = sorted(range(len(heat)), key=lambda eid: (heat[eid], last_used[eid], eid))
    priority = [0] * len(order)
    for rank, eid in enumerate(order):
        priority[eid] = rank
    return priority


def select_prefetch_one_reference(
    ids: Sequence[Sequence[int]],
    weights: Sequence[Sequence[float]] | None,
    mapping: Sequence[int],
    priority: Sequence[int],
    slots: int,
) -> PrefetchDecision:
    """Small CPU oracle matching the existing host prefetch selection.

    Zero scores are real predictions. Invalid IDs are ignored. A candidate's
    score is its maximum over rows; score ties retain first insertion order.
    All predicted IDs are protected, including candidates not selected for
    loading. Empty slots are intentionally not newly admitted: the existing
    host path chooses a resident victim, and this candidate preserves it.
    """
    if len(mapping) != len(priority) or not mapping or slots <= 0:
        raise ValueError("invalid mapping, priority or slot count")
    if sorted(priority) != list(range(len(mapping))):
        raise ValueError("priority must be a permutation of expert ranks")
    if not ids:
        return PrefetchDecision()
    k = len(ids[0])
    if k == 0 or any(len(row) != k for row in ids):
        raise ValueError("IDs must be a nonempty rectangular matrix")
    if weights is not None and (
        len(weights) != len(ids) or any(len(row) != k for row in weights)
    ):
        raise ValueError("weights must match IDs")
    predicted = {int(eid) for row in ids for eid in row if 0 <= int(eid) < len(mapping)}
    best_score: dict[int, float] = {}
    for row_index, row in enumerate(ids):
        for column, raw_eid in enumerate(row):
            eid = int(raw_eid)
            if not 0 <= eid < len(mapping) or mapping[eid] >= 0:
                continue
            score = float(weights[row_index][column]) if weights is not None else -float(column * len(ids) + row_index)
            if score > best_score.get(eid, float("-inf")):
                best_score[eid] = score
    if not best_score:
        return PrefetchDecision()
    candidate = max(best_score, key=best_score.get)
    eligible = [eid for eid, slot in enumerate(mapping) if 0 <= slot < slots and eid not in predicted]
    if not eligible:
        return PrefetchDecision()
    victim = min(eligible, key=priority.__getitem__)
    return PrefetchDecision(candidate, victim, int(mapping[victim]), True)


if triton is not None:

    @triton.jit
    def _select_and_describe(
        Ids, Weights, Mapping, Priority, Hotness,
        SourceTable, DestinationBases, Strides, FieldLengths,
        Candidate, Victim, Slot, Active,
        CopySources, CopyDestinations, CopyLengths, CopyCount,
        N: tl.constexpr, K: tl.constexpr, E: tl.constexpr,
        S: tl.constexpr, F: tl.constexpr, HAS_WEIGHTS: tl.constexpr,
        USE_HOTNESS: tl.constexpr,
        B_R: tl.constexpr, B_E: tl.constexpr, B_F: tl.constexpr,
    ):
        # One CTA owns every reduction. No inter-CTA ordering or atomics.
        r = tl.arange(0, B_R)
        e = tl.arange(0, B_E)
        route_ids = tl.load(Ids + r, r < N * K, other=-1).to(tl.int32)
        valid = (r < N * K) & (route_ids >= 0) & (route_ids < E)
        if HAS_WEIGHTS:
            route_scores = tl.load(Weights + r, r < N * K, other=float('-inf')).to(tl.float32)
        else:
            route_scores = -((r % K) * N + r // K).to(tl.float32)
        match = valid[:, None] & (route_ids[:, None] == e[None, :]) & (e[None, :] < E)
        predicted = tl.sum(match.to(tl.int32), axis=0) > 0
        # The host's `score > -inf` skips NaN/-inf insertion. This also keeps
        # a candidate's first successful insertion stable if a later row wins.
        scored_match = match & (route_scores[:, None] > float('-inf'))
        score = tl.max(tl.where(scored_match, route_scores[:, None], float('-inf')), axis=0)
        first = tl.min(tl.where(scored_match, r[:, None], N * K), axis=0)
        mapping = tl.load(Mapping + e, e < E, other=-1).to(tl.int32)
        missing = (e < E) & predicted & (mapping < 0) & (score > float('-inf'))
        best_score = tl.max(tl.where(missing, score, float('-inf')), axis=0)
        best_first = tl.min(tl.where(missing & (score == best_score), first, N * K), axis=0)
        candidate = tl.min(tl.where(missing & (score == best_score) & (first == best_first), e, E), axis=0)

        eligible = (e < E) & (mapping >= 0) & (mapping < S) & ~predicted
        if USE_HOTNESS:
            heat = tl.load(Hotness + e, e < E, other=float("inf")).to(tl.float32)
            best_heat = tl.min(tl.where(eligible, heat, float("inf")), axis=0)
            victim = tl.min(tl.where(eligible & (heat == best_heat), e, E), axis=0)
        else:
            priority = tl.load(Priority + e, e < E, other=E).to(tl.int32)
            best_priority = tl.min(tl.where(eligible, priority, E), axis=0)
            victim = tl.min(tl.where(eligible & (priority == best_priority), e, E), axis=0)
        active = (candidate < E) & (victim < E)
        safe_candidate = tl.where(active, candidate, 0)
        safe_victim = tl.where(active, victim, 0)
        slot = tl.load(Mapping + safe_victim).to(tl.int64)
        slot = tl.where(active, slot, 0)
        tl.store(Candidate, safe_candidate.to(tl.int64))
        tl.store(Victim, safe_victim.to(tl.int64))
        tl.store(Slot, slot)
        tl.store(Active, active)

        field = tl.arange(0, B_F)
        source = tl.load(SourceTable + safe_candidate * F + field, field < F, other=0)
        base = tl.load(DestinationBases + field, field < F, other=0)
        stride = tl.load(Strides + field, field < F, other=0)
        length = tl.load(FieldLengths + field, field < F, other=0)
        tl.store(CopySources + field, source, field < F)
        tl.store(CopyDestinations + field, base + slot * stride, field < F)
        tl.store(CopyLengths + field, tl.where(active, length, 0), field < F)
        # Crucial: an inactive replay does not reuse a previous positive count.
        tl.store(CopyCount, tl.where(active, F, 0).to(tl.int32))


    @triton.jit
    def _commit_mapping(Mapping, Candidate, Victim, Slot, Active, E: tl.constexpr, B_E: tl.constexpr):
        e = tl.arange(0, B_E)
        old = tl.load(Mapping + e, e < E, other=-1)
        active = tl.load(Active).to(tl.int1)
        candidate = tl.load(Candidate).to(tl.int32)
        victim = tl.load(Victim).to(tl.int32)
        slot = tl.load(Slot)
        updated = tl.where(active & (e == candidate), slot, tl.where(active & (e == victim), -1, old))
        tl.store(Mapping + e, updated, e < E)


class DevicePrefetchSelector:
    """Persistent device plan buffers; selection and descriptors in one CTA."""

    def __init__(self, source_table, destination_bases, strides, field_lengths,
                 priority=None, slots: int = 0, hotness=None):
        if triton is None:
            raise RuntimeError("Triton is required for device prefetch")
        import torch

        if priority is None and hotness is None:
            raise ValueError("device prefetch selector needs priority or hotness")
        self.sources = source_table
        self.destination_bases = destination_bases
        self.strides = strides
        self.field_lengths = field_lengths
        self.priority = priority
        self.hotness = hotness
        self.use_hotness = hotness is not None
        self.experts, self.fields = source_table.shape
        self.slots = slots
        if not (0 < self.experts <= MAX_EXPERTS and 0 < slots <= self.experts):
            raise ValueError("unsupported expert/slot count")
        if self.fields < 1 or self.fields > 16:
            raise ValueError("unsupported transfer field count")
        if any(t.numel() != self.fields for t in (destination_bases, strides, field_lengths)):
            raise ValueError("descriptor fields must match source-table width")
        rank = hotness if self.use_hotness else priority
        if rank.numel() != self.experts:
            raise ValueError("victim ranking must have one entry per expert")
        device = source_table.device
        self.candidate = torch.zeros(1, dtype=torch.int64, device=device)
        self.victim = torch.zeros(1, dtype=torch.int64, device=device)
        self.slot = torch.zeros(1, dtype=torch.int64, device=device)
        self.active = torch.zeros(1, dtype=torch.bool, device=device)
        self.copy_sources = torch.empty(self.fields, dtype=torch.int64, device=device)
        self.copy_destinations = torch.empty_like(self.copy_sources)
        self.copy_lengths = torch.empty(self.fields, dtype=torch.int32, device=device)
        self.copy_count = torch.zeros((), dtype=torch.int32, device=device)

    def prepare(self, ids, mapping, weights=None, hotness=None) -> None:
        import torch

        if ids.ndim != 2 or not ids.is_contiguous() or ids.numel() < 1 or ids.numel() > MAX_PREDICTED_ROUTES:
            raise ValueError("IDs must be contiguous rank two within the static route limit")
        if mapping.ndim != 1 or mapping.numel() != self.experts or not mapping.is_contiguous():
            raise ValueError("mapping must be a contiguous expert vector")
        if weights is not None and (
            weights.shape != ids.shape or not weights.is_contiguous() or weights.dtype != torch.float32
        ):
            raise ValueError("prediction weights must be matching contiguous fp32")
        active_hotness = self.hotness if hotness is None else hotness
        use_hotness = self.use_hotness and active_hotness is not None
        priority = self.priority if self.priority is not None else active_hotness
        ranking = active_hotness if use_hotness else self.priority
        if ranking is None:
            raise ValueError("victim ranking is unavailable")
        _select_and_describe[(1,)](
            ids, weights if weights is not None else priority, mapping, priority, ranking,
            self.sources, self.destination_bases, self.strides, self.field_lengths,
            self.candidate, self.victim, self.slot, self.active,
            self.copy_sources, self.copy_destinations, self.copy_lengths, self.copy_count,
            ids.shape[0], ids.shape[1], self.experts, self.slots, self.fields, weights is not None,
            use_hotness,
            triton.next_power_of_2(ids.numel()), triton.next_power_of_2(self.experts),
            triton.next_power_of_2(self.fields),
        )

    def commit(self, mapping) -> None:
        """Enqueue only AFTER byte copy and scale refresh on the same stream."""
        _commit_mapping[(1,)](
            mapping, self.candidate, self.victim, self.slot, self.active,
            self.experts, triton.next_power_of_2(self.experts),
        )


def unsupported_reason(manager, layer) -> str | None:
    """Static guard only: no device reads or side effects before fallback."""
    config = manager.offload_config
    cpu = config.cpu_moe
    if not config.expert_prefetch_enabled or config.expert_prefetch_num != 1:
        return "requires exactly one prefetched expert"
    if config.enable_multi_card or manager.ep_size != 1:
        return "requires single card"
    if not cpu.enabled or cpu.weight_format != "gguf" or manager.native_mxfp4_enabled:
        return "requires the GGUF hybrid CPU backend"
    if cpu.transfer_one_expert:
        return "requires frozen host LRC state (on-demand transfer disabled)"
    if cpu.transfer_scheduler != "npu" or cpu.transfer_policy != "load_balance":
        return "requires the existing NPU load-balance source/copy setup"
    library = getattr(cpu, "transfer_copy_library", "")
    if not library or not Path(library).is_file():
        return "requires the existing mapped-copy dispatch library"
    if not config.cache_policy_enabled or getattr(config, "cache_policy_type", "lrc") != "lrc":
        return "requires LRC cache policy"
    if getattr(config, "expert_predictor", "fate") != "fate":
        return "requires fate prediction"
    if not 0 < manager.num_total_experts <= MAX_EXPERTS:
        return "expert count exceeds one-CTA scope"
    if manager.prefetch_tokens * manager.topk > MAX_PREDICTED_ROUTES:
        return "prediction route count exceeds one-CTA scope"
    if manager.cache_policy is None or layer.log2phy is None:
        return "requires finalized cache state and residency mapping"
    if getattr(layer, "_hybrid_expert_executor", None) is None:
        return "requires a hybrid executor on the target layer"
    return None


class DevicePrefetchCandidate:
    """Reuse mapped GGUF/NZ sources, while preserving host prefetch policy."""

    @classmethod
    def try_create(cls, manager, layer):
        if unsupported_reason(manager, layer) is not None:
            return None
        return cls(manager, layer)

    def __init__(self, manager, layer):
        reason = unsupported_reason(manager, layer)
        if reason is not None:
            raise ValueError(reason)
        import torch
        from vllm_ascend.expert_offload.device_transfer import DeviceExpertTransfer

        layer_idx = manager.moe_layers.index(layer)
        state = manager.cache_policy.layer_states[layer_idx]
        from vllm_ascend.expert_offload.device_lrc_policy import DeviceLRCHotness
        # Seed once at construction; all subsequent LRC observations and
        # victim ranking stay on the NPU. The host priority permutation is no
        # longer part of the replay path.
        self.lrc = DeviceLRCHotness(
            manager.num_total_experts,
            layer.w13_weight_scale.device,
            manager.offload_config,
            state,
        )
        # Source registration, physical NZ/scale packing, library ownership and
        # pointer tables are identical to the proven device transfer path.
        # Do NOT attach this object as executor._weight_transfer: doing so
        # would enable demand admission despite transfer_one_expert=False.
        self.transport = DeviceExpertTransfer(manager, layer, initialize_policy=False)
        self.manager = manager
        self.layer_idx = layer_idx
        self.layer = layer
        self.selector = DevicePrefetchSelector(
            self.transport.sources, self.transport.destinations, self.transport.strides,
            self.transport.lengths,
            hotness=self.lrc.hotness(),
            slots=manager.num_device_experts_for_layer(layer_idx),
        )
        # NPU keeps the LRC decision; the selected scalar is copied to
        # pinned host memory before the stream callback performs the existing
        # direct H2D copy_ path.
        import torch
        self._candidate_h = torch.zeros((1,), dtype=torch.int64, device="cpu", pin_memory=True)
        self._slot_h = torch.zeros((1,), dtype=torch.int64, device="cpu", pin_memory=True)
        self._active_h = torch.zeros((1,), dtype=torch.int32, device="cpu", pin_memory=True)
        self._capture_scalar_buffers = []
        self._pending_scalar_buffers = None

    def observe_actual(self, ids, weights=None) -> None:
        """Observe the target layer's real top-k routes on the NPU."""
        self.lrc.observe(ids, weights)

    def refresh_priority_from_host(self) -> None:
        """Compatibility hook; ranking is now owned by the NPU LRC state."""
        return None

    def _direct_copy_callback(self, buffers, static_id=0, allow_async=False) -> None:
        """Run the old direct H2D path after the NPU selector has settled."""
        import torch_npu
        candidate_h, slot_h, active_h = buffers
        def copy_selected(_unused):
            if int(active_h.item()) == 0:
                return
            eid = int(candidate_h.item())
            slot = int(slot_h.item())
            from vllm_ascend.expert_offload.h2d_transfer import TorchCopyH2DTransport

            manager = self.manager
            transport = manager._get_h2d_transport()
            if not isinstance(transport, TorchCopyH2DTransport):
                raise RuntimeError(
                    "direct prefetch requires h2d_backend=\"torch\"; "
                    f"got {type(transport).__name__}"
                )
            tasks = manager._build_expert_h2d_tasks(self.layer, self.layer_idx, eid, slot)
            dma_stream = getattr(manager, "_cpu_prefetch_dma_stream", None)
            if dma_stream is None:
                dma_stream = manager.load_stream
            dma_done = torch_npu.npu.Event()
            with torch_npu.npu.stream(dma_stream):
                transport.copy_batch(tasks)
                manager._refresh_expert_fp32_scale(self.layer, slot)
                dma_stream.record_event(dma_done)
            # Runtime prefetch has no retired allocations, so the event is a
            # sufficient completion token.  Cold reconfiguration still uses
            # the original host synchronization before releasing old views.
            # The capture dispatcher consumes the returned event and waits on
            # it before publishing channel.done.  Eager/ACL host callbacks do
            # not consume a return value, so retain the synchronous fallback
            # there; otherwise the callback could return before DMA finishes.
            if (allow_async
                    and not getattr(manager, "_retired_expert_h2d_views", None)):
                return dma_done
            manager._synchronize_h2d(stream=dma_stream)
            return None

        return self.manager._run_prefetch_callback(
            copy_selected, buffers, None, trace_layer=self.layer_idx, static_id=static_id
        )

    def enqueue(self, predicted_ids, predicted_weights, mapping=None) -> None:
        """Run NPU LRC selection and stage the chosen scalar for the event worker."""
        mapping = self.layer.log2phy if mapping is None else mapping
        self._pending_mapping = mapping
        self.selector.prepare(
            predicted_ids,
            mapping,
            predicted_weights,
            hotness=self.lrc.hotness(),
        )
        from vllm_ascend.expert_offload.h2d_transfer import TorchCopyH2DTransport
        if not isinstance(self.manager._get_h2d_transport(), TorchCopyH2DTransport):
            # MemFabric/remote sources retain the original device sparse-copy.
            self.transport.copier.copy(
                self.selector.copy_sources, self.selector.copy_destinations,
                self.selector.copy_lengths, self.selector.copy_count,
            )
            if hasattr(self.layer, "w13_weight_scale_fp32"):
                import torch
                self.layer.w13_weight_scale_fp32.copy_(
                    self.layer.w13_weight_scale.to(torch.float32)
                )
            self.selector.commit(mapping)
            self._pending_direct = False
            return

        # NPU owns candidate/slot/LRC. Only the three scalar decisions cross
        # to pinned host memory; the old event worker will perform copy_().
        import torch
        if torch.npu.is_current_stream_capturing():
            buffers = (
                torch.zeros((1,), dtype=self._candidate_h.dtype, device="cpu", pin_memory=True),
                torch.zeros((1,), dtype=self._slot_h.dtype, device="cpu", pin_memory=True),
                torch.zeros((1,), dtype=self._active_h.dtype, device="cpu", pin_memory=True),
            )
            self._capture_scalar_buffers.append(buffers)
        else:
            buffers = (self._candidate_h, self._slot_h, self._active_h)
        candidate_h, slot_h, active_h = buffers
        candidate_h.copy_(self.selector.candidate, non_blocking=True)
        slot_h.copy_(self.selector.slot, non_blocking=True)
        active_h.copy_(
            self.selector.active.to(dtype=active_h.dtype),
            non_blocking=True,
        )
        self._pending_scalar_buffers = buffers
        self._pending_direct = True

    def submit_event_copy(self, stream, *, capturing: bool, static_id: int = 0) -> None:
        """Publish native-poll/Event handshake, then commit mapping."""
        if not getattr(self, "_pending_direct", False):
            return
        dispatcher = getattr(self.manager, "_cpu_prefetch_dispatcher", None)
        if dispatcher is not None and capturing:
            dispatcher.record_submission(
                stream=stream,
                callback=lambda buffers=self._pending_scalar_buffers, sid=static_id: self._direct_copy_callback(
                    buffers, sid, allow_async=True),
                layer=self.layer_idx,
                static_id=static_id,
            )
        else:
            import torch_npu
            from vllm_ascend.expert_offload.expert_offload_manager import (
                get_subscribed_compute_streams,
            )
            subscribed_streams = get_subscribed_compute_streams()
            if stream not in subscribed_streams:
                torch_npu.npu._subscribe_report(stream)
                subscribed_streams.add(stream)
            torch_npu.npu._launch_host_func(
                stream, lambda _unused: self._direct_copy_callback(
                    self._pending_scalar_buffers, static_id, allow_async=False), None
            )
        # record_submission inserts stream.wait_event(done); commit follows
        # the CPU copy_ DMA and its completion ExternalEvent.
        self.selector.commit(self._pending_mapping)
        self._pending_direct = False
