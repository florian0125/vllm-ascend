import importlib
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.expert_offload.hybrid_executor import (
    HybridExpertExecutor,
    attach_hybrid_expert_executor,
    maybe_prepare_hybrid_routes,
)


class Task:
    def __init__(self, value):
        self.value = value

    def wait(self):
        return self.value


def reference(x, ids, weights, experts):
    result = torch.zeros_like(x)
    for row in range(ids.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[row, slot])
            if 0 <= expert < len(experts) and weights[row, slot] != 0:
                # A real gated expert MLP, independent of the splitting code.
                gate, up, down = experts[expert]
                result[row] += (torch.nn.functional.silu(x[row] @ gate) * (x[row] @ up)) @ down * weights[row, slot]
    return result


class Backend:
    max_num_tokens = 16

    def __init__(self, experts):
        self.experts = experts
        self.calls = []

    def submit(self, **kwargs):
        self.calls.append(kwargs)
        return Task(reference(kwargs["hidden_states"], kwargs["topk_ids"], kwargs["topk_weights"], self.experts))


@pytest.fixture
def inputs():
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(6, 4, generator=generator)
    experts = [
        (
            torch.randn(4, 8, generator=generator),
            torch.randn(4, 8, generator=generator),
            torch.randn(8, 4, generator=generator),
        )
        for _ in range(4)
    ]
    return x, experts


@pytest.mark.parametrize("mapping", [[0, 1, 2, 3], [-1] * 4, [0, -1, 1, -1]])
@pytest.mark.parametrize("strategy", ["anchor_union", "threshold"])
def test_verify_final_routes_match_reference(inputs, mapping, strategy):
    x, experts = inputs
    ids = torch.tensor([[0, 1, 1], [2, 3, 0], [1, 3, 2], [3, 0, 1], [2, 2, 3], [-1, -1, -1]])
    weights = torch.tensor(
        [[0.5, 0.25, 0.25], [0.3, 0.2, 0.5], [0.2, 0.7, 0.1], [0.2, 0.5, 0.3], [0.4, 0.4, 0.2], [0.0, 0.0, 0.0]]
    )
    if strategy == "anchor_union":
        # Protected token/affiliate selection has already produced this pool.
        ids[:5] %= 2
    else:
        ids[2, 1] = -1
        weights[2, 1] = 0
    backend = Backend(experts)
    executor = HybridExpertExecutor(backend, 0)
    original_ids, original_weights = ids.clone(), weights.clone()
    plan = executor.prepare(hidden_states=x, topk_ids=ids, topk_weights=weights, log2phy=torch.tensor(mapping))
    npu = reference(x, plan.npu_topk_ids, plan.npu_topk_weights, experts)
    out = executor.finish(npu, plan.cpu_task)
    torch.testing.assert_close(out, reference(x, ids, weights, experts))
    torch.testing.assert_close(ids, original_ids)
    torch.testing.assert_close(weights, original_weights)
    assert ((plan.cpu_topk_weights != 0) & (plan.npu_topk_weights != 0)).sum() == 0
    assert (plan.cpu_topk_ids[-1] == -1).all()


def test_invalid_and_zero_routes_are_never_indexed_or_transferred(inputs):
    x, experts = inputs
    ids = torch.tensor([[-1, 999, 0], [1, 2, 3]])
    weights = torch.tensor([[0.8, 0.2, 0.0], [0.5, 0.0, 0.5]])
    observed = []

    class Transfer:
        def prepare(self, routes, mapping):
            observed.append(routes.clone())
            return SimpleNamespace(graph=False, wait_for_weights=lambda _: None)

    backend = Backend(experts)
    plan = HybridExpertExecutor(backend, 0).prepare(
        hidden_states=x[:2],
        topk_ids=ids,
        topk_weights=weights,
        log2phy=torch.tensor([-1, 0, -1, -1]),
        weight_transfer=Transfer(),
    )
    torch.testing.assert_close(observed[0], torch.tensor([[-1, -1, -1], [1, -1, 3]]))
    assert (plan.cpu_topk_ids[0] == -1).all()
    assert (plan.cpu_topk_weights[0] == 0).all()


@pytest.mark.parametrize("phase, expected_calls", [(True, 1), (False, 1), (None, 1)])
def test_enabled_cpu_runs_regardless_of_forward_phase(inputs, monkeypatch, phase, expected_calls):
    x, experts = inputs
    context = importlib.import_module("vllm.forward_context")
    monkeypatch.setattr(context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(context, "get_forward_context", lambda: SimpleNamespace(cpu_moe_decode=phase))
    layer = torch.nn.Module()
    backend = Backend(experts)
    attach_hybrid_expert_executor(layer, backend, 0)
    plan = maybe_prepare_hybrid_routes(
        layer,
        hidden_states=x,
        topk_ids=torch.zeros(6, 2, dtype=torch.long),
        topk_weights=torch.full((6, 2), 0.5),
        log2phy=torch.tensor([-1, 0, -1, 1]),
    )
    assert len(backend.calls) == expected_calls
    assert plan is not None


def test_padded_verify_capacity_overflow_uses_npu_before_submit(inputs, monkeypatch):
    x, experts = inputs
    context = importlib.import_module("vllm.forward_context")
    monkeypatch.setattr(context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(context, "get_forward_context", lambda: SimpleNamespace(cpu_moe_decode=True))
    layer = torch.nn.Module()
    backend = Backend(experts)
    backend.max_num_tokens = 5
    attach_hybrid_expert_executor(layer, backend, 0)
    plan = maybe_prepare_hybrid_routes(
        layer,
        hidden_states=x,
        topk_ids=torch.zeros(6, 2, dtype=torch.long),
        topk_weights=torch.ones(6, 2),
        log2phy=torch.tensor([-1, 0, -1, 1]),
    )
    assert plan is None
    assert not backend.calls


def test_threshold_pruning_precedes_cpu_submission(inputs, monkeypatch):
    """Use the production pruning functions without unrelated operator imports."""
    import ast
    import sys
    import types
    from pathlib import Path

    import vllm_ascend.expert_offload.hybrid_executor as module

    context = importlib.import_module("vllm.forward_context")
    monkeypatch.setattr(context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(context, "get_forward_context", lambda: SimpleNamespace(cpu_moe_decode=True))
    tree = ast.parse((Path(module.__file__).parents[1] / "ops/fused_moe/experts_selector.py").read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("dynamic_pruning_unsorted", "maybe_prune_topk_experts")
    ]
    selector = types.ModuleType("vllm_ascend.ops.fused_moe.experts_selector")
    config = SimpleNamespace(
        experts_pruning_enabled=True, experts_pruning_threshold=[0.0, 0.4], experts_pruning_debug=False
    )
    selector.__dict__.update(
        torch=torch, _prune_thr_npu=None, get_ascend_config=lambda: SimpleNamespace(expert_offload_config=config)
    )
    exec(compile(ast.Module(body=functions, type_ignores=[]), "production_pruning", "exec"), selector.__dict__)
    monkeypatch.setitem(sys.modules, selector.__name__, selector)
    x, experts = inputs
    layer = torch.nn.Module()
    backend = Backend(experts)
    attach_hybrid_expert_executor(layer, backend, 0)
    ids = torch.tensor([[0, 1], [2, 3], [1, 0], [3, 2], [0, -1], [999, 1]])
    weights = torch.tensor([[0.8, 0.2], [0.9, 0.1], [0.8, 0.2], [0.9, 0.1], [1.0, 0.0], [0.0, 1.0]])
    mapping = torch.tensor([0, -1, 1, -1])
    manager = SimpleNamespace(
        offload_config=SimpleNamespace(experts_pruning_enabled=True, cpu_moe=SimpleNamespace(transfer_one_expert=False))
    )
    plan = maybe_prepare_hybrid_routes(
        layer,
        hidden_states=x,
        topk_ids=ids,
        topk_weights=weights,
        log2phy=mapping,
        transfer_manager=manager,
    )
    # Weak misses were removed; strong misses stayed on CPU. Weak hits remain.
    torch.testing.assert_close(
        plan.cpu_topk_ids, torch.tensor([[-1, -1], [-1, -1], [1, -1], [3, -1], [-1, -1], [1, -1]])
    )
    valid = (ids >= 0) & (ids < 4)
    final_weights, final_ids, _ = selector.maybe_prune_topk_experts(
        torch.where(valid, weights, 0), torch.where(valid, ids, -1), log2phy=mapping
    )
    out = HybridExpertExecutor.finish(reference(x, plan.npu_topk_ids, plan.npu_topk_weights, experts), plan.cpu_task)
    torch.testing.assert_close(out, reference(x, final_ids, final_weights, experts))


def test_disabled_cpu_keeps_existing_path(inputs):
    x, _ = inputs
    layer = torch.nn.Module()
    plan = maybe_prepare_hybrid_routes(
        layer,
        hidden_states=x,
        topk_ids=torch.zeros(6, 2, dtype=torch.long),
        topk_weights=torch.ones(6, 2),
        log2phy=torch.tensor([-1, 0, -1, 1]),
    )
    assert plan is None


def test_startup_profile_skips_cpu_before_capacity_check(inputs, monkeypatch):
    x, experts = inputs
    context = importlib.import_module("vllm.forward_context")
    monkeypatch.setattr(context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(context, "get_forward_context", lambda: SimpleNamespace(in_profile_run=True))
    layer = torch.nn.Module()
    backend = Backend(experts)
    backend.max_num_tokens = 1
    attach_hybrid_expert_executor(layer, backend, 0)
    plan = maybe_prepare_hybrid_routes(
        layer,
        hidden_states=x,
        topk_ids=torch.zeros(6, 2, dtype=torch.long),
        topk_weights=torch.ones(6, 2),
        log2phy=torch.tensor([-1, 0, -1, 1]),
    )
    assert plan is None
    assert not backend.calls


@pytest.mark.parametrize("rows,capacity,expected_pages", [(2, 2, 0), (3, 2, 1)])
def test_new_offload_boundary_hybrid_and_capacity_fallback(rows, capacity, expected_pages):
    import ast
    from pathlib import Path

    source = Path(__file__).parents[3] / "vllm_ascend/expert_offload/routed_experts.py"
    tree = ast.parse(source.read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name == "apply_offloaded_experts")

    class RemoveImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None

    fn = RemoveImports().visit(fn)
    pages = []
    submissions = []
    task = SimpleNamespace(wait=lambda: torch.zeros(rows, 4))
    backend = SimpleNamespace(max_num_tokens=capacity,
                              submit=lambda **kw: submissions.append(kw) or task)
    layer = torch.nn.Module()
    layer.enable_multi_card = False
    layer.router = SimpleNamespace()
    layer.tid2eid = None
    layer.log2phy = torch.tensor([0, -1])
    layer.moe_config = SimpleNamespace(num_local_experts=1, ep_size=1)
    layer.quant_method = SimpleNamespace(apply=lambda **kw: SimpleNamespace(
        routed_out=torch.zeros(rows, 4), cpu_expert_task=None))
    attach_hybrid_expert_executor(layer, backend, 0)
    mgr = SimpleNamespace(offload_threshold=100, _prefill_initialized=False,
                          offload_config=SimpleNamespace(experts_pruning_enabled=False,
                              cpu_moe=SimpleNamespace(transfer_one_expert=False)),
                          update_weights=lambda *a, **kw: pages.append(kw),
                          trigger_next_layer_prefetch=lambda *a: None)
    namespace = dict(torch=torch, ExpertOffloadManager=SimpleNamespace(get_instance=lambda: mgr),
                     _EXTRA_CTX=SimpleNamespace(moe_comm_method=SimpleNamespace(
                         token_dispatcher=SimpleNamespace())),
                     maybe_prepare_hybrid_routes=maybe_prepare_hybrid_routes,
                     _map_logical_ids_to_phy=lambda ids, mapping, count: ids)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 str(source), "exec"), namespace)
    result = namespace["apply_offloaded_experts"](
        layer, torch.ones(rows, 4), torch.ones(rows, 1),
        torch.ones(rows, 1, dtype=torch.long), torch.ones(rows, 2))
    assert len(pages) == expected_pages
    assert len(submissions) == 1 - expected_pages
    assert (result.cpu_expert_task is not None) == (expected_pages == 0)
