#include <cstdint>
#include <cstdio>
#include <exception>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/framework/OpCommand.h"
extern "C" void LaunchCpuMoeMappedCombine(void*, void*, void*, uint64_t, uint32_t, void*);
extern "C" int CpuMoeMappedCombine(void* host, void* input, void* output, uint64_t count, uint32_t dtype, uint8_t device) {
    try {
        c10_npu::OptionalNPUGuard guard; guard.set_index(device);
        auto stream = c10_npu::getCurrentNPUStream(device).stream(false);
        auto launch = [=]() -> int { LaunchCpuMoeMappedCombine(host, input, output, count, dtype, stream); return 0; };
        at_npu::native::OpCommand::RunOpApiV2("cpu_moe_mapped_combine", launch);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "CpuMoeMappedCombine failed: %s\n", error.what()); return -1;
    }
}
