"""Opt-in, graph-replay-safe single-card route counters.

Only fixed-shape tensor operations run inside the graph. One asynchronous
snapshot is queued per real forward; a host writer publishes cumulative totals.
No device synchronize, item(), dynamic unique(), or per-layer logging is used.
"""
import atexit
import json
import os
import queue
import threading
import time
import uuid
from pathlib import Path

import torch


MAX_TRACKED_LAYERS = 256
COLUMNS = (
    "layer_calls", "pre_routes", "pre_hit_routes", "pre_experts", "pre_hit_experts",
    "post_routes", "post_hit_routes", "post_experts", "post_hit_experts",
    "npu_routes", "cpu_routes", "admitted_experts",
)


def route_counts(ids, weights, resident, actual_rows):
    """Counts are per layer-call; repeated experts count once within that call."""
    n_experts = resident.numel()
    rows = torch.arange(ids.shape[0], device=ids.device)[:, None]
    valid = (rows < actual_rows) & (ids >= 0) & (ids < n_experts) & (weights != 0)
    safe = ids.clamp(0, n_experts - 1).long()
    hit = valid & resident[safe]
    counts = torch.zeros(n_experts, dtype=torch.int32, device=ids.device)
    counts.scatter_add_(0, safe.reshape(-1), valid.reshape(-1).to(torch.int32))
    present = counts > 0
    return torch.stack((valid.sum(), hit.sum(), present.sum(), (present & resident).sum()))


class BenchmarkRouteCounters:
    def __init__(self, device, output_dir, rank=0):
        self.device = torch.device(device)
        self.values = torch.zeros((MAX_TRACKED_LAYERS, len(COLUMNS)), dtype=torch.int64, device=device)
        # Runtime inputs, written outside capture before EACH forward. Captured
        # operations read these addresses on replay, including for padded graphs.
        self.actual_rows = torch.zeros((), dtype=torch.int64, device=device)
        self.hash_layers = set()
        self.spec = dict(accepted=0, proposed=0, drafts=0)
        self.sequence = 0
        self.configuration = {}
        self.session_id = uuid.uuid4().hex
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / f"benchmark_counters_rank{rank}.json"
        self.pending = queue.Queue()
        self.free = []
        self.free_lock = threading.Lock()
        self.writer_error = None
        self.thread = threading.Thread(target=self._writer, name="route_stats_writer", daemon=True)
        self.thread.start()
        atexit.register(self.close)

    def set_forward(self, actual_rows, is_decode):
        if self.writer_error is not None:
            raise RuntimeError("Benchmark counter writer failed") from self.writer_error
        self.actual_rows.fill_(int(actual_rows) if is_decode else 0)

    def begin(self, layer_idx, ids, weights, mapping, is_hash=False):
        if not 0 <= layer_idx < MAX_TRACKED_LAYERS:
            raise ValueError("Too many MoE layers for benchmark counters")
        if is_hash:
            self.hash_layers.add(layer_idx)
        resident = (mapping >= 0).clone()
        pre = route_counts(ids, weights, resident, self.actual_rows)
        return resident, pre

    def finish(self, layer_idx, sample, ids, weights, mapping, use_prefill=False):
        resident, pre = sample
        post = route_counts(ids, weights, resident, self.actual_rows)
        after = mapping >= 0
        rows = torch.arange(ids.shape[0], device=ids.device)[:, None]
        valid = (rows < self.actual_rows) & (ids >= 0) & (ids < mapping.numel()) & (weights != 0)
        safe = ids.clamp(0, mapping.numel() - 1).long()
        npu = post[0] if use_prefill else (valid & after[safe]).sum()
        enabled = (self.actual_rows > 0).to(torch.int64)
        loaded = ((~resident) & after).sum()
        values = torch.cat((enabled.reshape(1), pre, post, npu.reshape(1),
                            (post[0] - npu).reshape(1), loaded.reshape(1)))
        self.values[layer_idx].add_(values * enabled)

    def record_spec(self, accepted, proposed, drafts):
        self.spec["accepted"] += int(accepted)
        self.spec["proposed"] += int(proposed)
        self.spec["drafts"] += int(drafts)

    def publish(self):
        """Called after each real forward, outside graph capture, on its stream."""
        import torch_npu
        with self.free_lock:
            host = self.free.pop() if self.free else None
        if host is None:
            host = torch.empty(self.values.shape, dtype=torch.int64, device="cpu", pin_memory=True)
        host.copy_(self.values, non_blocking=True)
        event = torch_npu.npu.Event()
        event.record(torch_npu.npu.current_stream())
        self.sequence += 1
        self.pending.put((event, host, self.sequence, dict(self.spec), time.time_ns(), tuple(self.hash_layers)))

    def _save(self, item):
        event, host, sequence, spec, timestamp, hashes = item
        layers = {str(i): dict(zip(COLUMNS, map(int, row)))
                  for i, row in enumerate(host.tolist()) if row[0]}
        total = {key: sum(row[key] for i, row in layers.items() if int(i) not in hashes) for key in COLUMNS}
        data = dict(schema_version=1, session_id=self.session_id, pid=os.getpid(),
                    sequence=sequence, wall_ns=timestamp, published_wall_ns=time.time_ns(),
                    columns=COLUMNS, configuration=self.configuration, excluded_hash_layers=sorted(hashes), totals=total,
                    per_layer=layers, speculative=spec,
                    scope="decode/spec-decode actual rows; single-card; totals exclude hash layers",
                    semantics="pre=before substitution and migration; post=after substitution/pruning before migration; final=after migration; experts deduplicated per layer-call",
                    overhead="opt-in tensor reductions per layer, one asynchronous D2H snapshot per forward")
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, separators=(",", ":")))
        os.replace(tmp, self.path)

    def _writer(self):
        latest = None
        last_save = 0.0
        try:
            torch.npu.set_device(self.device)
            while True:
                try:
                    item = self.pending.get(timeout=0.1)
                except queue.Empty:
                    if latest is not None:
                        self._save(latest)
                        with self.free_lock:
                            self.free.append(latest[1])
                        latest = None
                    continue
                if item is None:
                    if latest is not None:
                        self._save(latest)
                    return
                # Query only: no blocking NPU sync on the inference thread.
                while not item[0].query():
                    time.sleep(0.001)
                if latest is not None:
                    with self.free_lock:
                        self.free.append(latest[1])
                latest = item
                if time.monotonic() - last_save >= 0.25:
                    self._save(latest)
                    last_save = time.monotonic()
        except Exception as exc:
            self.writer_error = exc

    def close(self):
        self.pending.put(None)
        self.thread.join(timeout=5)


def get_route_counters():
    # Manager owns the state; avoid a second process-global singleton.
    from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager
    manager = ExpertOffloadManager._instance
    return getattr(manager, "_benchmark_route_counters", None)
