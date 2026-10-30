import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.expert_offload.cpu_backend.wiring as wiring


@pytest.fixture
def cpu_config_class():
    # Exercise the production class without importing the unrelated full
    # Ascend configuration/worker dependency graph.
    path = Path(wiring.__file__).parents[2] / "ascend_config.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CpuMoeConfig")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["CpuMoeConfig"]


def test_cpu_config_disabled_by_default(cpu_config_class):
    cfg = cpu_config_class()
    assert not cfg.enabled
    assert not cfg.includes(0)


@pytest.mark.parametrize("capacity", [0, -1, True, "6"])
def test_cpu_config_rejects_invalid_capacity(cpu_config_class, capacity):
    with pytest.raises((TypeError, ValueError)):
        cpu_config_class({"max_num_tokens": capacity})


@pytest.mark.parametrize("capacity", [10, 12, 16])
def test_wiring_checks_verify_capacity_before_backend_creation(cpu_config_class, monkeypatch, capacity):
    config_module = types.ModuleType("vllm.config")
    config_module.get_current_vllm_config = lambda: SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1024, max_num_seqs=2),
    )
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)
    monkeypatch.setattr(wiring, "model_layer_index", lambda _: 3)
    created = []

    def factory(config):
        created.append(config)
        return SimpleNamespace(max_num_tokens=config.max_num_tokens, submit=lambda **_: None)

    monkeypatch.setattr(wiring, "KtKernelCPUExpertBackend", factory)
    cfg = cpu_config_class(
        {"enabled": True, "layers": [3], "weight_path": "/unused/layer3.gguf", "max_num_tokens": capacity}
    )
    args = dict(
        routed_experts=torch.nn.Module(),
        layer_name="model.layers.3.mlp",
        moe_config=SimpleNamespace(
            ep_size=1, num_logical_experts=4, experts_per_token=2, hidden_dim=4, intermediate_size_per_partition=8
        ),
        cpu_moe_config=cfg,
        enable_multi_card=False,
        tp_size=1,
    )
    assert wiring.maybe_attach_cpu_expert_backend(**args).max_num_tokens == capacity
    assert len(created) == 1


@pytest.mark.parametrize("tp, ep, multi_card", [(2, 1, False), (1, 2, False), (1, 1, True)])
def test_wiring_retains_single_card_constraint(cpu_config_class, monkeypatch, tp, ep, multi_card):
    monkeypatch.setattr(wiring, "model_layer_index", lambda _: 3)
    cfg = cpu_config_class({"enabled": True, "layers": [3], "weight_path": "/unused/layer3.gguf"})
    with pytest.raises(ValueError, match="single-card"):
        wiring.maybe_attach_cpu_expert_backend(
            routed_experts=torch.nn.Module(),
            layer_name="model.layers.3.mlp",
            moe_config=SimpleNamespace(ep_size=ep),
            cpu_moe_config=cfg,
            enable_multi_card=multi_card,
            tp_size=tp,
        )


def test_npu_transfer_resolves_checkout_copy_library(cpu_config_class, monkeypatch):
    import vllm_ascend.expert_offload.copy_library as library

    monkeypatch.setattr(library, "get_default_copy_library", lambda: "/checkout/default-copy.so")
    cfg = cpu_config_class({"transfer_scheduler": "npu"})
    assert cfg.transfer_copy_library == "/checkout/default-copy.so"


def test_npu_transfer_preserves_explicit_library(cpu_config_class, monkeypatch):
    import vllm_ascend.expert_offload.copy_library as library

    def unexpected_default():
        raise AssertionError("Explicit copy library must bypass default resolution")

    monkeypatch.setattr(library, "get_default_copy_library", unexpected_default)
    cfg = cpu_config_class({"transfer_scheduler": "npu", "transfer_copy_library": "/custom/copy.so"})
    assert cfg.transfer_copy_library == "/custom/copy.so"


def test_missing_checkout_copy_library_has_build_instruction(monkeypatch, tmp_path):
    import vllm_ascend.expert_offload.copy_library as library

    monkeypatch.setattr(library, "__file__", str(tmp_path / "copy_library.py"))
    with pytest.raises(FileNotFoundError, match="csrc/expert_transfer_copy/build.sh"):
        library.get_default_copy_library()
