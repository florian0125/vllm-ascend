"""Run on a free NPU: validates runtime row masks under real ACL graph replay."""
import json
import time

import torch
import torch_npu  # noqa: F401

from vllm_ascend.expert_offload.benchmark_counters import BenchmarkRouteCounters


def test_benchmark_route_counters_graph(tmp_path):
    device = torch.device('npu', torch.npu.current_device())
    counter = BenchmarkRouteCounters(device, tmp_path)
    try:
        ids = torch.tensor([[0, 1], [1, 2], [0, 2]], device=device)
        weights = torch.ones((3, 2), device=device)
        initial = torch.tensor([0, -1, -1], device=device)
        after = torch.tensor([0, 1, -1], device=device)
        mapping = initial.clone()
        def forward():
            mapping.copy_(initial)
            sample = counter.begin(0, ids, weights, mapping)
            mapping.copy_(after)
            counter.finish(0, sample, ids, weights, mapping)
        for _ in range(3):
            forward()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            forward()
        counter.values.zero_()
        counter.set_forward(1, True)
        graph.replay()
        counter.set_forward(2, True)
        graph.replay()
        counter.record_spec(7, 14, 2)
        counter.publish()
        deadline = time.monotonic() + 10
        while not counter.path.exists() and time.monotonic() < deadline:
            time.sleep(.05)
        assert counter.writer_error is None
        data = json.loads(counter.path.read_text())
        totals = data['totals']
        assert totals['layer_calls'] == 2
        assert totals['pre_routes'] == 6 and totals['pre_hit_routes'] == 2
        assert totals['npu_routes'] == 5 and totals['cpu_routes'] == 1
        assert totals['admitted_experts'] == 2
        assert data['speculative'] == dict(accepted=7, proposed=14, drafts=2)
        counter.set_forward(0, False)
        graph.replay()
        torch.npu.synchronize()
        assert counter.values[0].cpu().tolist() == list(totals.values())
    finally:
        counter.close()
