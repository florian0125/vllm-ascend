"""Exact numeric and repeated-host-write checks for mapped result reads."""
import pytest


@pytest.mark.parametrize("tokens", [1, 3, 7, 32, 100])
@pytest.mark.parametrize("dtype_name", ["bfloat16", "float32"])
def test_mapped_combine_exact_and_host_visibility(tokens, dtype_name):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is required")
    from vllm_ascend.expert_offload.cpu_backend.mapped_combine import mapped_combine

    torch.npu.set_device(0)
    dtype = getattr(torch, dtype_name)
    with torch.inference_mode():
        host = torch.randn(tokens, 4096, dtype=torch.bfloat16, pin_memory=True)
        device = torch.randn(tokens, 4096, dtype=dtype, device="npu")
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = mapped_combine(device, host)
        for step in range(20):
            host.copy_(torch.randn_like(host) * (step + 1))
            expected = device + host.to(device=device.device, dtype=dtype)
            torch.npu.synchronize()
            graph.replay()
            torch.npu.synchronize()
            assert torch.equal(output.cpu(), expected.cpu()), step
