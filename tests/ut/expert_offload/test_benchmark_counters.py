"""Counters: duplicates, padding, replacement timing, replay and interval deltas."""
import importlib.util
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import torch

from vllm_ascend.expert_offload.benchmark_counters import (
    BenchmarkRouteCounters, COLUMNS, route_counts,
)


def test_route_counts_ignore_padding_zero_weights_and_invalid_ids():
    ids = torch.tensor([[0, 0, 2], [1, -1, 99], [2, 2, 2]])
    weights = torch.tensor([[1., 1., 1.], [0., 1., 1.], [1., 1., 1.]])
    counts = route_counts(ids, weights, torch.tensor([True, False, False]), torch.tensor(2))
    assert counts.tolist() == [3, 2, 2, 1]


def bare_counter(device='cpu'):
    counter = BenchmarkRouteCounters.__new__(BenchmarkRouteCounters)
    counter.actual_rows = torch.tensor(1, device=device)
    counter.values = torch.zeros((2, len(COLUMNS)), dtype=torch.int64, device=device)
    counter.hash_layers = set()
    return counter


def test_pre_and_post_use_old_residency_final_uses_new_residency():
    counter = bare_counter()
    mapping = torch.tensor([0, -1, -1])
    raw = torch.tensor([[0, 1, 2]])
    weights = torch.ones_like(raw, dtype=torch.float32)
    sample = counter.begin(0, raw, weights, mapping)
    # Substitute expert 2 by resident expert 0; then admit expert 1.
    post = torch.tensor([[0, 1, 0]])
    mapping[1] = 1
    counter.finish(0, sample, post, weights, mapping)
    values = dict(zip(COLUMNS, counter.values[0].tolist()))
    assert values['pre_hit_routes'] == 1
    assert values['pre_experts'] == 3
    assert values['post_hit_routes'] == 2
    assert values['post_experts'] == 2
    assert values['npu_routes'] == 3 and values['cpu_routes'] == 0
    assert values['admitted_experts'] == 1


def test_prefill_disabled_gate_and_repeated_accumulation():
    counter = bare_counter()
    ids = torch.tensor([[0, 1]])
    weights = torch.ones_like(ids, dtype=torch.float32)
    mapping = torch.tensor([0, -1])
    for rows in [0, 1, 1]:
        counter.actual_rows.fill_(rows)
        sample = counter.begin(0, ids, weights, mapping)
        counter.finish(0, sample, ids, weights, mapping)
    values = dict(zip(COLUMNS, counter.values[0].tolist()))
    assert values['layer_calls'] == 2
    assert values['pre_routes'] == 4
    assert values['cpu_routes'] == 2


def load_wrapper():
    path = Path(__file__).resolve().parents[3] / 'tools/aisbench_with_stats.py'
    spec = importlib.util.spec_from_file_location('aisbench_stats_tool', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_interval_uses_weighted_counts_and_rejects_restarts():
    module = load_wrapper()
    zero = dict.fromkeys(COLUMNS, 0)
    before = dict(session_id='a', sequence=1, totals=zero, per_layer={},
                  speculative=dict(accepted=0, proposed=0, drafts=0))
    totals = dict(zero, pre_routes=100, pre_hit_routes=80, pre_experts=20,
                  pre_hit_experts=10, post_routes=100, post_hit_routes=90,
                  post_experts=20, post_hit_experts=18, npu_routes=95, cpu_routes=5)
    after = dict(session_id='a', sequence=2, totals=totals, per_layer={},
                 speculative=dict(accepted=7, proposed=14, drafts=2), scope='test', excluded_hash_layers=[])
    report = module.calculate(before, after)
    assert report['rates']['pre_route_hit_rate'] == .8
    assert report['rates']['draft_token_acceptance_rate'] == .5
    assert report['rates']['mean_acceptance_length'] == 4.5
    after['session_id'] = 'b'
    with pytest.raises(ValueError, match='restarted'):
        module.calculate(before, after)


def test_wrapper_cli_saves_one_interval(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'healthy')
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    zero = dict.fromkeys(COLUMNS, 0)
    before = dict(session_id='cli-test', sequence=1, totals=zero, per_layer={},
                  speculative=dict(accepted=0, proposed=0, drafts=0))
    after = dict(before, sequence=2, scope='test', excluded_hash_layers=[],
                 totals=dict(zero, pre_routes=10, pre_hit_routes=8,
                             post_routes=10, npu_routes=9, cpu_routes=1),
                 speculative=dict(accepted=7, proposed=14, drafts=2))
    counter = tmp_path / 'counters.json'
    counter.write_text(json.dumps(before))
    tool = Path(__file__).resolve().parents[3] / 'tools/aisbench_with_stats.py'
    out = tmp_path / 'run'
    command = [sys.executable, str(tool), '--counter-file', str(counter),
               '--out', str(out), '--url', f'http://127.0.0.1:{server.server_port}',
               '--settle-seconds', '0.5', '--', sys.executable, '-c',
               'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2])',
               str(counter), json.dumps(after)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads((out / 'summary.json').read_text())
        assert report['rates']['pre_route_hit_rate'] == .8
        assert report['rates']['draft_token_acceptance_rate'] == .5
        assert (out / 'metrics_before.txt').exists()
        assert (out / 'after.json').exists()
    finally:
        server.shutdown()
        server.server_close()
