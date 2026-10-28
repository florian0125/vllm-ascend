"""Report-thread callback must fail visibly without leaking the GIL."""
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def callback(exits):
    source = Path(__file__).parents[3] / 'vllm_ascend/expert_offload/expert_offload_manager.py'
    tree = ast.parse(source.read_text())
    manager = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ExpertOffloadManager')
    method = next(n for n in manager.body if isinstance(n, ast.FunctionDef) and n.name == '_run_prefetch_callback')
    namespace = {'torch': torch, 'os': SimpleNamespace(_exit=exits.append)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['_run_prefetch_callback']


def test_report_thread_mutates_inference_tensor():
    with torch.inference_mode():
        mapping = torch.tensor([0, -1, 1])
    exits, failures = [], []
    manager = SimpleNamespace(_note_cb_failure=failures.append)
    def planner(args):
        assert torch.is_inference_mode_enabled()
        args.copy_(torch.tensor([-1, 0, 1]))
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(callback(exits), manager, planner, mapping, mapping).result(timeout=5)
    assert mapping.tolist() == [-1, 0, 1]
    assert exits == failures == []


@pytest.mark.parametrize('error', [RuntimeError('partial copy'), SystemExit('callback failed')])
def test_planner_failure_terminates_instead_of_reusing_partial_weights(error):
    exits, failures = [], []
    manager = SimpleNamespace(_note_cb_failure=failures.append)
    def planner(args):
        raise error
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(callback(exits), manager, planner, None, None).result(timeout=5)
    assert failures == ['prefetch']
    assert exits == [1]


def test_logger_failure_cannot_escape_callback_bridge():
    exits = []
    def fail(*args):
        raise RuntimeError('failed')
    manager = SimpleNamespace(_note_cb_failure=fail)
    callback(exits)(manager, fail, None, None)
    assert exits == [1]
