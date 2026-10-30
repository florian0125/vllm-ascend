/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
*/

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "acc_offload_operators.h"

extern "C" {
void AccOffloadSparseCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs, uint32_t *sizePtr, uint8_t devIdx)
{
    c10_npu::OptionalNPUGuard npuGuard;
    npuGuard.set_index(devIdx);

    auto stream = c10_npu::getCurrentNPUStream(devIdx);
    void *npuStream = stream.stream(false);

    auto callback = [srcPtrs, dstPtrs, lenPtrs, sizePtr, npuStream]() -> int {
        OffloadOpsSparseCopy(srcPtrs, dstPtrs, lenPtrs, sizePtr, npuStream);
        return 0;
    };

    at_npu::native::OpCommand::RunOpApiV2("acc_sparse_copy", callback);
}

}

extern "C" void AccOffloadNativeCopy(uint64_t *recipes, int64_t *expert, int64_t *slot,
                                      uint8_t *active, uint8_t device)
{
    c10_npu::OptionalNPUGuard guard;
    guard.set_index(device);
    auto stream = c10_npu::getCurrentNPUStream(device).stream(false);
    auto callback = [recipes, expert, slot, active, stream]() -> int {
        OffloadOpsNativeCopy(recipes, expert, slot, active, stream);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("fused_native_expert_copy", callback);
}

extern "C" uint32_t AccOffloadNativeCopyPlanVersion() { return 2; }

extern "C" void OffloadOpsTransferMark(uint64_t*,uint64_t*,uint32_t,uint32_t,int64_t*,int64_t*,uint8_t*,void*);
extern "C" void OffloadOpsSparseCopyTrace(uint64_t*,uint64_t*,uint32_t*,uint32_t*,uint64_t*,uint64_t*,int64_t*,int64_t*,uint32_t,uint32_t,uint32_t,void*);
extern "C" void AccExpertTransferMark(uint64_t* records,uint64_t* counter,uint32_t capacity,uint32_t phase,
 int64_t* expert,int64_t* slot,uint8_t* active,uint8_t device) {
 c10_npu::OptionalNPUGuard guard;guard.set_index(device);
 auto stream=c10_npu::getCurrentNPUStream(device).stream(false);
 auto callback=[=]()->int{OffloadOpsTransferMark(records,counter,capacity,phase,expert,slot,active,stream);return 0;};
 at_npu::native::OpCommand::RunOpApiV2("expert_transfer_trace_mark",callback);
}
extern "C" void AccOffloadSparseCopyTrace(uint64_t* src,uint64_t* dst,uint32_t* lens,uint32_t* count,
 uint64_t* records,uint64_t* counters,int64_t* expert,int64_t* slot,
 uint32_t capacity,uint32_t layer,uint32_t mode,uint8_t device) {
 c10_npu::OptionalNPUGuard guard;guard.set_index(device);
 auto stream=c10_npu::getCurrentNPUStream(device).stream(false);
 auto callback=[=]()->int{OffloadOpsSparseCopyTrace(src,dst,lens,count,records,counters,expert,slot,capacity,layer,mode,stream);return 0;};
 at_npu::native::OpCommand::RunOpApiV2("acc_sparse_copy_trace",callback);
}

extern "C" uint32_t AccOffloadSparseCopyTraceCoreCount() { return 2; }
