"""Opt-in host phase diagnostics; no device synchronization or callback file I/O."""

import atexit
import itertools
import json
import os
import queue
import threading
import time
from pathlib import Path

from vllm_ascend import envs


class TransferDiagnostics:
    def __init__(self, directory):
        # Keep writes off shared NFS and off the runtime callback thread.
        directory = Path(directory).resolve()
        if directory != Path('/tmp') and Path('/tmp') not in directory.parents:
            raise ValueError('CPU transfer diagnostics must use a local /tmp directory')
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f'transfer_diagnostics_{os.getpid()}.jsonl'
        self.queue = queue.SimpleQueue()
        self.sequence = itertools.count(1)
        self.error = None
        self.thread = threading.Thread(target=self._write, daemon=True,
                                       name='cpu-transfer-diagnostics')
        self.thread.start()
        atexit.register(self.close)

    @classmethod
    def configured(cls):
        directory = envs.VLLM_ASCEND_CPU_MOE_DIAGNOSTICS
        return cls(directory) if directory else None

    def begin(self, layer, capture, replay):
        return TransferRecord(self, layer, next(self.sequence), capture, replay)

    def _write(self):
        try:
            with self.path.open('a', encoding='utf8') as stream:
                pending = []
                while True:
                    try:
                        record = self.queue.get(timeout=0.05)
                    except queue.Empty:
                        record = False
                    if record is not False and record is not None:
                        pending.append(record)
                        continue
                    # Serialize after a quiet interval, not between layer callbacks.
                    for saved in pending:
                        stream.write(json.dumps(saved, separators=(',', ':')) + '\n')
                    if pending:
                        stream.flush()
                        pending.clear()
                    if record is None:
                        break
        except Exception as exc:
            self.error = repr(exc)

    def close(self):
        if self.thread.is_alive():
            self.queue.put(None)
            self.thread.join(timeout=5)


class TransferRecord:
    def __init__(self, writer, layer, sequence, capture, replay):
        # First stamp is taken immediately on entering the Python callback.
        mono = time.perf_counter_ns()
        wall = time.time_ns()
        cpu = time.thread_time_ns()
        self.writer = writer
        self.data = dict(event='cpu_moe.transfer_diagnostics', pid=os.getpid(),
                         tid=threading.get_native_id(), layer=layer,
                         sequence=sequence, capture=capture, replay=replay,
                         wall_ns=wall, mono_ns=mono, thread_cpu_ns=cpu,
                         phases=[dict(name='callback_enter', mono_ns=mono,
                                      wall_ns=wall, thread_cpu_ns=cpu)])

    def mark(self, name, **fields):
        mono = time.perf_counter_ns()
        self.data['phases'].append(dict(
            name=name, mono_ns=mono,
            wall_ns=self.data['wall_ns'] + mono - self.data['mono_ns'],
            thread_cpu_ns=time.thread_time_ns(), **fields))

    def finish(self, error=None):
        self.mark('callback_exit')
        if error is not None:
            self.data['error'] = repr(error)
        self.data['writer_error'] = self.writer.error
        self.writer.queue.put(self.data)


def inspect_route_buffers(backend, hidden_states):
    """Inspect existing buffers only: never allocate or repair a missing cache."""
    from kt_kernel.experts_base import KExpertsCPUBuffer

    batch = int(hidden_states.numel() // hidden_states.shape[-1])
    slot = backend.config.layer_idx % KExpertsCPUBuffer.buffer_depth
    captured = KExpertsCPUBuffer.capture_buffers.get(batch)
    temporary = (KExpertsCPUBuffer.temp_buffer
                 if KExpertsCPUBuffer.temp_bs == batch else None)
    buffers = captured or temporary
    result = dict(batch=batch, slot=slot, cache_hit=captured is not None,
                  temp_hit=temporary is not None,
                  cached_batches=sorted(KExpertsCPUBuffer.capture_buffers),
                  capture_batches=list(KExpertsCPUBuffer.capture_bs),
                  temp_batch=KExpertsCPUBuffer.temp_bs)
    if buffers:
        result.update(input_ptr=buffers[0][slot].data_ptr(),
                      ids_ptr=buffers[1][slot].data_ptr(),
                      deferred_ids_ptr=buffers[2][slot].data_ptr(),
                      weights_ptr=buffers[3][slot].data_ptr(),
                      output_ptr=buffers[4][slot].data_ptr(),
                      ids=buffers[1][slot].tolist(),
                      weights=buffers[3][slot].tolist())
    return result
