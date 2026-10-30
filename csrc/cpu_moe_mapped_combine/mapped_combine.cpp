#include "kernel_operator.h"
#include <type_traits>

// CPU BF16 results are read directly from mapped DRAM into UB. No result
// staging tensor in HBM and no standalone H2D operation are used.
template <typename DeviceT>
class MappedCombine {
public:
    __aicore__ inline void Run(GM_ADDR host, GM_ADDR device, GM_ADDR output, uint64_t count) {
        host_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(host), count);
        host_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        device_.SetGlobalBuffer(reinterpret_cast<__gm__ DeviceT*>(device), count);
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ DeviceT*>(output), count);
        pipe_.InitBuffer(hostQueue_, 1, TILE * sizeof(bfloat16_t));
        pipe_.InitBuffer(deviceQueue_, 1, TILE * sizeof(DeviceT));
        pipe_.InitBuffer(outQueue_, 1, TILE * sizeof(DeviceT));
        pipe_.InitBuffer(hostFloat_, TILE * sizeof(float));
        pipe_.InitBuffer(deviceFloat_, TILE * sizeof(float));
        pipe_.InitBuffer(sum_, TILE * sizeof(float));
        const uint64_t step = TILE * AscendC::GetBlockNum();
        for (uint64_t offset = TILE * AscendC::GetBlockIdx(); offset < count; offset += step) {
            const uint32_t length = count - offset < TILE ? count - offset : TILE;
            auto h = hostQueue_.AllocTensor<bfloat16_t>();
            auto d = deviceQueue_.AllocTensor<DeviceT>();
            AscendC::DataCopyExtParams hp(1, length * sizeof(bfloat16_t), 0, 0, 0);
            AscendC::DataCopyExtParams dp(1, length * sizeof(DeviceT), 0, 0, 0);
            AscendC::DataCopyPadExtParams<bfloat16_t> hpad(false, 0, 0, 0);
            AscendC::DataCopyPadExtParams<DeviceT> dpad(false, 0, 0, 0);
            AscendC::DataCopyPad(h, host_[offset], hp, hpad);
            AscendC::DataCopyPad(d, device_[offset], dp, dpad);
            hostQueue_.EnQue(h); deviceQueue_.EnQue(d);
            h = hostQueue_.DeQue<bfloat16_t>(); d = deviceQueue_.DeQue<DeviceT>();
            auto hf = hostFloat_.Get<float>(); auto df = deviceFloat_.Get<float>();
            auto sum = sum_.Get<float>(); auto out = outQueue_.AllocTensor<DeviceT>();
            AscendC::Cast(hf, h, AscendC::RoundMode::CAST_NONE, length);
            if constexpr (std::is_same<DeviceT, float>::value) {
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(out, hf, d, length);
            } else {
                AscendC::Cast(df, d, AscendC::RoundMode::CAST_NONE, length);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(sum, hf, df, length);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(out, sum, AscendC::RoundMode::CAST_RINT, length);
            }
            outQueue_.EnQue(out); out = outQueue_.DeQue<DeviceT>();
            AscendC::DataCopyPad(output_[offset], out, dp);
            hostQueue_.FreeTensor(h); deviceQueue_.FreeTensor(d); outQueue_.FreeTensor(out);
        }
    }
private:
    static constexpr uint32_t TILE = 4096;
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> hostQueue_, deviceQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> hostFloat_, deviceFloat_, sum_;
    AscendC::GlobalTensor<bfloat16_t> host_;
    AscendC::GlobalTensor<DeviceT> device_, output_;
};
extern "C" __global__ __aicore__ void CpuMoeMappedCombineBf16(GM_ADDR host, GM_ADDR device, GM_ADDR output, uint64_t count) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    MappedCombine<bfloat16_t> op; op.Run(host, device, output, count);
}
extern "C" __global__ __aicore__ void CpuMoeMappedCombineFloat(GM_ADDR host, GM_ADDR device, GM_ADDR output, uint64_t count) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    MappedCombine<float> op; op.Run(host, device, output, count);
}
extern "C" void LaunchCpuMoeMappedCombine(void* host, void* device, void* output, uint64_t count, uint32_t dtype, void* stream) {
    if (dtype == 0) CpuMoeMappedCombineBf16<<<2, nullptr, stream>>>(host, device, output, count);
    else CpuMoeMappedCombineFloat<<<2, nullptr, stream>>>(host, device, output, count);
}
