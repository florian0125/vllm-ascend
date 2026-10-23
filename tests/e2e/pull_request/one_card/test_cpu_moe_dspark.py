"""Actual NPU H2D/graph regression with a small synthetic CPU worker.

This checks scheduling and route ownership, not the real kt-kernel's numerics
or model-level performance. Run on an idle NPU, outside serving benchmarks.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.expert_offload.cpu_backend.kt_kernel_backend import KtKernelCPUExpertBackend
from vllm_ascend.expert_offload.hybrid_executor import HybridExpertExecutor
from vllm_ascend.expert_offload.hybrid_transfer import (
    HybridExpertTransfer,
    select_transfer_expert,
)


@pytest.mark.parametrize("num_tokens", (1, 6, 10))
@pytest.mark.parametrize("execution", ("eager", "graph"))
@pytest.mark.parametrize("layer_count", (1, 2))
@pytest.mark.parametrize("transfer_enabled", (False, True))
def test_verify_changed_routes_and_cpu_h2d_overlap(num_tokens, execution, layer_count, transfer_enabled):
    device = torch.device("npu", torch.npu.current_device())
    pool = ThreadPoolExecutor(max_workers=1)
    load_stream = torch.npu.Stream(device=device)
    # Empty SimpleNamespace instances compare equal; the real layers use
    # identity. Distinct objects keep manager.moe_layers.index(layer) correct.
    layers = [object() for _ in range(layer_count)]
    slots = [torch.tensor([1.0, 2.0], device=device) for _ in layers]
    source = torch.arange(1.0, 17.0, pin_memory=True)
    loads = []
    overlaps = []
    wrappers = []

    class Wrapper:
        def __init__(self):
            self.x = torch.empty((num_tokens, 32), pin_memory=True)
            self.ids = torch.empty((num_tokens, 6), dtype=torch.int64, pin_memory=True)
            self.weights = torch.empty((num_tokens, 6), pin_memory=True)
            self.out = torch.empty((num_tokens, 32), pin_memory=True)
            self.device_out = torch.empty((num_tokens, 32), device=device)
            self.started = threading.Event()
            self.loaded = threading.Event()
            self.future = None

        def copy_inputs_to_cpu_buffers(self, x, ids, weights):
            self.x.copy_(x, non_blocking=True)
            self.ids.copy_(ids, non_blocking=True)
            self.weights.copy_(weights, non_blocking=True)

        def forward_on_pinned_buffers(self, x, stream):
            self.started.clear()
            self.loaded.clear()
            inputs, ids, weights = self.x.clone(), self.ids.clone(), self.weights.clone()

            def compute():
                self.started.set()
                # A selected expert now owns an NPU route. Keep this synthetic
                # worker active through that expert's H2D to verify the fork.
                # With no selection, there is no weight callback to release it.
                if bool((ids < 0).any()):
                    # NPU routes also occur without a transfer, so completion
                    # is released by the drain in that case below.
                    self.loaded.wait(5)
                gains = (ids.clamp_min(0) + 1) * (ids >= 0)
                self.out.copy_(inputs * (weights * gains).sum(dim=1, keepdim=True))

            self.future = pool.submit(compute)

        def submit_forward(self, x, ids, weights, stream):
            self.copy_inputs_to_cpu_buffers(x, ids, weights)
            # Synthetic wrapper only: ensure its staged inputs are readable.
            # The production wrapper is unchanged and owns its stream protocol.
            torch.npu.current_stream().synchronize()
            self.forward_on_pinned_buffers(x, stream)

        def sync_forward(self, x, stream):
            self.drain_pinned_forward()
            return self.copy_forward_output_to_device(x)

        def drain_pinned_forward(self):
            self.loaded.set()
            self.future.result(timeout=10)

        def copy_forward_output_to_device(self, x):
            self.device_out.copy_(self.out, non_blocking=True)
            return self.device_out

    def load(layer, idx, eid, slot):
        wrapper = wrappers[idx]
        # Record callback evidence and assert on the test thread. Exceptions
        # escaping a runtime host callback can obscure the original failure.
        started = wrapper.started.wait(5)
        overlaps.append((idx, started, wrapper.future is not None and not wrapper.future.done()))
        slots[idx][slot : slot + 1].copy_(source[eid : eid + 1], non_blocking=True)
        loads.append((idx, eid, slot))

    # No blocking CPU join is queued until after the main-stream weight
    # callback and NPU work, so these callbacks can complete on first replay.
    manager = SimpleNamespace(
        exclusive_dynamic_enabled=False,
        moe_layers=layers,
        load_stream=load_stream,
        num_device_experts_for_layer=lambda _: 2,
        _load_expert_weights_into_slot=load,
        _synchronize_h2d=load_stream.synchronize,
    )
    executors, transfers, mappings = [], [], []
    for idx, layer in enumerate(layers):
        wrapper = Wrapper()
        wrappers.append(wrapper)
        backend = KtKernelCPUExpertBackend.__new__(KtKernelCPUExpertBackend)
        backend.config = SimpleNamespace(layer_idx=idx, top_k=6, max_num_tokens=num_tokens, hidden_size=32)
        backend._stream_handle_provider = lambda _: 0
        backend._wrapper = wrapper
        backend._in_flight = None
        backend._graph_tasks = []
        backend._graph_task_created_ns = None
        backend._require_graph_methods = lambda: None
        backend._pin_graph_batch_size = lambda x: None
        backend._pinned_expert_ids = lambda x, w=wrapper: sorted(set(w.ids.reshape(-1).tolist()))
        executors.append(HybridExpertExecutor(backend, idx))
        transfers.append(HybridExpertTransfer(manager, layer))
        mapping = torch.full((16,), -1, dtype=torch.int32, device=device)
        mapping[:2].copy_(torch.tensor([0, 1], dtype=torch.int32, device=device))
        mappings.append(mapping)

    ids = torch.tensor([[2, 3, 4, 5, 6, 7]], device=device).repeat(num_tokens, 1)
    weights = torch.full((num_tokens, 6), 1.0 / 6, device=device)
    x = torch.ones((num_tokens, 32), device=device)

    def forward():
        outputs = []
        for idx, executor in enumerate(executors):
            plan = executor.prepare(
                hidden_states=x,
                topk_ids=ids,
                topk_weights=weights,
                log2phy=mappings[idx],
                weight_transfer=transfers[idx] if transfer_enabled else None,
            )
            physical = mappings[idx][plan.npu_topk_ids].clamp_min(0).long()
            npu = x * (slots[idx][physical] * plan.npu_topk_weights).sum(dim=1, keepdim=True)
            outputs.append(executor.finish(npu, plan.cpu_task))
        return outputs

    try:
        # Compile the small kernels before capture. No model is loaded.
        (x * (slots[0][ids.clamp_max(1)] * weights).sum()).sum()
        torch.npu.synchronize()
        graph = None
        if execution == "graph":
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                outputs = forward()
            torch.npu.synchronize()
        else:
            outputs = forward()
            torch.npu.synchronize()
        cases = [
            [0, 1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6, 7],
            [2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13],
            [8, 9, 10, 11, 12, 13],
            [0, 0, 1, 1, 2, 2],
            [14, 14, 14, 14, 14, 14],
            [0, 0, 1, 1, 0, 1],
            [-1, -1, -1, -1, -1, -1],
            [-1, 0, 2, -1, 3, 1],
        ]
        for step in range(24):
            routes = [cases[(step + row) % len(cases)] for row in range(num_tokens)]
            ids.copy_(torch.tensor(routes, device=device))
            weights.copy_(torch.where(ids >= 0, 1.0 / 6, 0.0))
            row_scale = torch.arange(1, num_tokens + 1, device=device).view(-1, 1) * (step + 1)
            x.copy_(row_scale.expand_as(x))
            expected_loads = []
            for idx, mapping in enumerate(mappings):
                selected = select_transfer_expert(torch.tensor(routes), mapping.cpu(), 2) if transfer_enabled else None
                if selected is not None:
                    expected_loads.append((idx, *selected))
            before = len(loads)
            before_overlap = len(overlaps)
            if graph is not None:
                graph.replay()
            else:
                outputs = forward()
            torch.npu.synchronize()
            assert loads[before:] == expected_loads
            assert all(started and active for _, started, active in overlaps[before_overlap:])
            expected = (
                torch.tensor(
                    [
                        (step + 1) * (row + 1) * sum(eid + 1 for eid in route if eid >= 0) / 6
                        for row, route in enumerate(routes)
                    ]
                )
                .view(-1, 1)
                .expand(num_tokens, 32)
            )
            for output in outputs:
                torch.testing.assert_close(output.cpu(), expected)
    finally:
        for wrapper in wrappers:
            wrapper.loaded.set()
        pool.shutdown(wait=True)
