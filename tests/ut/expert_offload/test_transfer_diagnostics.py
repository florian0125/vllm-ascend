import json
import sys
from types import SimpleNamespace

import torch

from vllm_ascend.expert_offload.cpu_backend.transfer_diagnostics import (
    TransferDiagnostics, inspect_route_buffers,
)


def test_writer_preserves_phase_order_and_reports_failures(tmp_path):
    writer = TransferDiagnostics(tmp_path)
    record = writer.begin(3, {'staged_ids_ptr': 123}, 7)
    record.mark('cpu_submit_begin')
    record.mark('cpu_submit_end')
    record.mark('weight_h2d_copy_begin', nbytes=1024)
    record.finish(ValueError('test exception'))
    writer.close()
    saved = json.loads(writer.path.read_text())
    assert saved['layer'] == 3 and saved['replay'] == 7
    assert saved['capture']['staged_ids_ptr'] == 123
    assert 'test exception' in saved['error']
    phases = saved['phases']
    assert [p['name'] for p in phases] == [
        'callback_enter', 'cpu_submit_begin', 'cpu_submit_end',
        'weight_h2d_copy_begin', 'callback_exit']
    assert all(a['mono_ns'] <= b['mono_ns'] for a, b in zip(phases, phases[1:]))
    assert all(a['thread_cpu_ns'] <= b['thread_cpu_ns'] for a, b in zip(phases, phases[1:]))
    assert all(p['wall_ns'] - saved['wall_ns'] == p['mono_ns'] - saved['mono_ns'] for p in phases)


def test_writer_rejects_shared_disk():
    import pytest
    with pytest.raises(ValueError, match='local /tmp'):
        TransferDiagnostics('/mnt/share/diagnostics')


def test_buffer_inspection_does_not_allocate_and_exposes_pointer_change(monkeypatch):
    def make_buffers():
        return tuple([torch.zeros(1, 6) for _ in range(2)] for _ in range(7))

    captured = make_buffers()
    current = make_buffers()
    current[1][0].fill_(0)
    buffers = SimpleNamespace(buffer_depth=2, capture_buffers={}, capture_bs=[1],
                              temp_bs=2, temp_buffer=current)
    monkeypatch.setitem(sys.modules, 'kt_kernel', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'kt_kernel.experts_base',
                        SimpleNamespace(KExpertsCPUBuffer=buffers))
    backend = SimpleNamespace(config=SimpleNamespace(layer_idx=0))
    hidden = torch.zeros(1, 4)
    missing = inspect_route_buffers(backend, hidden)
    assert 'ids_ptr' not in missing
    assert buffers.capture_buffers == {}
    buffers.capture_buffers[1] = captured
    old = inspect_route_buffers(backend, hidden)
    buffers.capture_buffers[1] = current
    new = inspect_route_buffers(backend, hidden)
    assert old['ids_ptr'] != new['ids_ptr']
    assert new['ids_ptr'] == current[1][0].data_ptr()
    assert new['ids'] == [[0.0] * 6]
