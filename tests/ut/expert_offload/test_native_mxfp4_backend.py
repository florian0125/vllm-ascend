"""Opt-in native mode must preserve borrowing and legacy GGUF defaults."""
import ast
from pathlib import Path
import pytest
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
