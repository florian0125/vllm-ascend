"""Opt-in native mode must preserve borrowing and legacy GGUF defaults."""
import ast
from pathlib import Path
import pytest
import sys
import types
import torch
from vllm_ascend.expert_offload.cpu_backend.kt_kernel_backend import KtKernelBackendConfig, KtKernelCPUExpertBackend


def test_native_directory_resolves_without_layer_template(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    config = KtKernelBackendConfig(0, 3, 3, 256, 256, str(tmp_path), weight_format="native_mxfp4")
    assert config.resolved_weight_path() == str(tmp_path)


def test_native_owner_is_forwarded_without_copy(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    owner = object()
    calls = []

    class Wrapper:

        def load_weights(self, mapping):
            assert mapping is None

        def submit_forward(self, *args):
            pass

        def sync_forward(self, *args):
            pass

    def factory(**kwargs):
        calls.append(kwargs)
        return Wrapper()

    config = KtKernelBackendConfig(0, 3, 3, 256, 256, str(tmp_path), weight_format="native_mxfp4")
    KtKernelCPUExpertBackend(config, native_weights=owner, wrapper_factory=factory)
    assert calls[0]["native_weights"] is owner
    assert calls[0]["weight_format"] == "native_mxfp4"


def test_native_rejects_unsupported_numa_split():
    with pytest.raises(ValueError, match="one NUMA"):
        KtKernelBackendConfig(0, 3, 3, 256, 256, "/unused", threadpool_count=2, weight_format="native_mxfp4")


def test_invalid_weight_format_rejected():
    with pytest.raises(ValueError, match="weight_format"):
        KtKernelBackendConfig(0, 3, 3, 256, 256, "/unused", weight_format="invalid")


def test_native_service_config_accepts_checkpoint_directory():
    path = Path(__file__).parents[3] / "vllm_ascend/ascend_config.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "CpuMoeConfig")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    config = namespace["CpuMoeConfig"]({
        "enabled": True,
        "layers": "all",
        "weight_path": "/model",
        "weight_format": "native_mxfp4"
    })
    assert config.weight_format == "native_mxfp4"
    assert config.weight_path == "/model"


def test_graph_registration_retains_warmup_buffers_across_sizes(monkeypatch):
    from vllm_ascend.expert_offload.cpu_backend import kt_kernel_backend as module

    warmup4 = (object(),)
    warmup1 = (object(),)
    buffers = types.SimpleNamespace(temp_bs=4, temp_buffer=warmup4,
                                    capture_buffers={})
    fake = types.ModuleType("kt_kernel.experts_base")
    fake.KExpertsCPUBuffer = buffers
    monkeypatch.setitem(sys.modules, "kt_kernel.experts_base", fake)
    monkeypatch.setattr(module, "_GRAPH_PINNED_BATCH_SIZES", set())
    monkeypatch.setattr(module, "_GRAPH_BATCH_SIGNATURES", {})
    backend = object.__new__(KtKernelCPUExpertBackend)
    backend.config = types.SimpleNamespace(top_k=3)
    backend._wrapper = types.SimpleNamespace(set_capture_batch_sizes=lambda sizes: None)
    backend._pin_graph_batch_size(torch.empty(4, 256))
    buffers.temp_bs, buffers.temp_buffer = 1, warmup1
    backend._pin_graph_batch_size(torch.empty(1, 256))
    assert buffers.capture_buffers[4] is warmup4
    assert buffers.capture_buffers[1] is warmup1
    # An already captured allocation must survive later temporary warmups.
    buffers.temp_bs, buffers.temp_buffer = 4, (object(),)
    backend._pin_graph_batch_size(torch.empty(4, 256))
    assert buffers.capture_buffers[4] is warmup4
