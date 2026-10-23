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

#include "acc_offload_sparse_copy.h"

extern "C" __global__ __aicore__ void OffloadSparseCopyOps(GM_ADDR inputs, GM_ADDR outputs, GM_ADDR lens, GM_ADDR size)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    OffloadSparseCopyKernel op;
    op.Init(inputs, outputs, lens, size);
    op.Process();
}

extern "C" void OffloadOpsSparseCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs, uint32_t *sizePtr,
                                     void *stream)
{
    constexpr uint32_t blockDim = 32;
    uint8_t *inputs = reinterpret_cast<uint8_t *>(srcPtrs);
    uint8_t *outputs = reinterpret_cast<uint8_t *>(dstPtrs);
    uint8_t *lens = reinterpret_cast<uint8_t *>(lenPtrs);
    uint8_t *size = reinterpret_cast<uint8_t *>(sizePtr);

    OffloadSparseCopyOps<<<blockDim, nullptr, stream>>>(inputs, outputs, lens, size);
}

extern "C" __global__ __aicore__ void FusedNativeExpertCopyOps(
    GM_ADDR recipes, GM_ADDR expert, GM_ADDR slot, GM_ADDR active)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    NativeExpertCopyKernel op;
    op.Init(recipes, expert, slot, active);
    op.Process();
}

extern "C" void OffloadOpsNativeCopy(uint64_t *recipes, int64_t *expert, int64_t *slot,
                                    uint8_t *active, void *stream)
{
    FusedNativeExpertCopyOps<<<32, nullptr, stream>>>(
        reinterpret_cast<GM_ADDR>(recipes), reinterpret_cast<GM_ADDR>(expert),
        reinterpret_cast<GM_ADDR>(slot), reinterpret_cast<GM_ADDR>(active));
}
