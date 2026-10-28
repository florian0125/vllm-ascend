"""Hardware regression for direct mapped-host CPU result combine."""

import pytest


def test_mapped_cpu_result_across_graph_replays(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is required")

    from vllm_ascend.expert_offload.cpu_backend.kt_kernel_backend import (
        KtKernelBackendConfig,
        KtKernelCPUExpertBackend,
    )

    from vllm_ascend.expert_offload.hybrid_executor import HybridExpertExecutor

    torch.npu.set_device(0)
    torch.set_num_threads(1)
    weight_path = tmp_path / "dummy.gguf"
    weight_path.touch()

    class Wrapper:
        def __init__(self, **kwargs):
            self.input = torch.zeros(8, 4096, pin_memory=True,
                                     dtype=torch.bfloat16)
            self.output = torch.zeros_like(self.input, pin_memory=True)
            self.device_output = torch.zeros(8, 4096, device="npu",
                                             dtype=torch.bfloat16)

        def load_weights(self, path):
            pass

        def set_capture_batch_sizes(self, sizes):
            pass

        def submit_forward(self, *args):
            raise AssertionError("Unexpected eager submission")

        def sync_forward(self, *args):
            raise AssertionError("Unexpected eager join")

        def copy_inputs_to_cpu_buffers(self, hidden, ids, weights):
            self.input.copy_(hidden, non_blocking=True)

        def forward_on_pinned_buffers(self, *args, **kwargs):
            pass

        def drain_pinned_forward(self):
            self.output.copy_(self.input * 2 + 3)

        def get_forward_output_buffers(self, hidden):
            return self.output, self.device_output

        def get_forward_output_device(self, hidden):
            return self.device_output

        def copy_forward_output_to_device(self, hidden):
            raise AssertionError("Mapped combine must not stage CPU results through H2D")

    with torch.inference_mode():
        backends = [
            KtKernelCPUExpertBackend(
                KtKernelBackendConfig(layer, 2, 1, 4096, 8,
                                      str(weight_path), max_num_tokens=8),
                wrapper_factory=Wrapper,
            )
            for layer in range(2)
        ]
        hidden = torch.zeros(8, 4096, device="npu", dtype=torch.bfloat16)
        ids = torch.zeros(8, 1, device="npu", dtype=torch.int64)
        weights = torch.ones(8, 1, device="npu")
        torch.npu.synchronize()
        graphs = []
        outputs = []
        for _ in range(4):
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                result = hidden
                for layer, backend in enumerate(backends):
                    task = backend.submit(
                        layer_idx=layer, hidden_states=result,
                        topk_ids=ids, topk_weights=weights,
                    )
                    result = HybridExpertExecutor.finish(torch.zeros_like(result), task)
            graphs.append(graph)
            outputs.append(result)

        for step in range(300):
            hidden.fill_(step % 50)
            for backend in backends:
                backend._wrapper.output.fill_(-777)
                backend._wrapper.device_output.fill_(-999)
            graphs[step % len(graphs)].replay()
            torch.npu.synchronize()
            expected = torch.full((8, 4096), 4 * (step % 50) + 9,
                                  dtype=torch.bfloat16)
            assert torch.equal(outputs[step % len(graphs)].cpu(), expected), step
