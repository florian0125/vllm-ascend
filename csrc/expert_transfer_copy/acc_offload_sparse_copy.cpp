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

extern "C" __global__ __aicore__ void OffloadSparseCopyOps(GM_ADDR inputs, GM_ADDR outputs, GM_ADDR lens, GM_ADDR size, uint32_t wave)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    OffloadSparseCopyKernel op;
    op.Init(inputs, outputs, lens, size, wave);
    op.Process();
}

extern "C" void OffloadOpsSparseCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs, uint32_t *sizePtr,
                                     void *stream)
{
    constexpr uint32_t blockDim = 2;
    uint8_t *inputs = reinterpret_cast<uint8_t *>(srcPtrs);
    uint8_t *outputs = reinterpret_cast<uint8_t *>(dstPtrs);
    uint8_t *lens = reinterpret_cast<uint8_t *>(lenPtrs);
    uint8_t *size = reinterpret_cast<uint8_t *>(sizePtr);

    // Capture a fixed launch topology; descriptors and count stay device-owned.
    for (uint32_t wave = 0; wave < SPARSE_COPY_WAVES; ++wave) {
        OffloadSparseCopyOps<<<blockDim, nullptr, stream>>>(inputs, outputs, lens, size, wave);
    }
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

#include "expert_transfer_trace.h"

extern "C" __global__ __aicore__ void ExpertTransferMarkOps(
 GM_ADDR records,GM_ADDR counter,uint32_t capacity,uint32_t phase,
 GM_ADDR expert,GM_ADDR slot,GM_ADDR active) {
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 auto seqptr=reinterpret_cast<__gm__ uint64_t*>(counter);
 TraceFlush(seqptr);
 uint64_t seq;
 if(phase==0){seq=*seqptr;*seqptr=seq+1;}else{seq=*seqptr-1;}
 auto rec=reinterpret_cast<__gm__ uint64_t*>(records)+(seq%capacity)*8;
 TraceFlush(rec);
 if(phase==1){TraceFlush(reinterpret_cast<__gm__ uint64_t*>(expert));TraceFlush(reinterpret_cast<__gm__ uint64_t*>(slot));TraceFlush(reinterpret_cast<__gm__ uint64_t*>(active));}
 rec[phase]=AscendC::GetSystemCycle();
 if(phase==1){rec[5]=*reinterpret_cast<__gm__ uint64_t*>(expert);
             rec[6]=*reinterpret_cast<__gm__ uint64_t*>(slot);
             rec[7]=*reinterpret_cast<__gm__ uint8_t*>(active);}
 if(phase==1){seqptr[1]+=rec[7];seqptr[2]+=rec[1]-rec[0];}
 if(phase==2)seqptr[3]+=rec[2]-rec[1];
 if(phase==3)seqptr[4]+=rec[3]-rec[2];
 if(phase==4)seqptr[5]+=rec[4]-rec[3];
 TraceFlush(rec);TraceFlush(seqptr);
}
extern "C" void OffloadOpsTransferMark(uint64_t* records,uint64_t* counter,
 uint32_t capacity,uint32_t phase,int64_t* expert,int64_t* slot,uint8_t* active,void* stream) {
 ExpertTransferMarkOps<<<1,nullptr,stream>>>(reinterpret_cast<GM_ADDR>(records),reinterpret_cast<GM_ADDR>(counter),
 capacity,phase,reinterpret_cast<GM_ADDR>(expert),reinterpret_cast<GM_ADDR>(slot),reinterpret_cast<GM_ADDR>(active));
}
extern "C" __global__ __aicore__ void OffloadSparseCopyTraceOps(
 GM_ADDR inputs,GM_ADDR outputs,GM_ADDR lens,GM_ADDR size,GM_ADDR records,
 GM_ADDR counters,GM_ADDR expert,GM_ADDR slot,uint32_t capacity,uint32_t layer,uint32_t mode) {
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 SparseCopyTraceKernel op;op.Run(inputs,outputs,lens,size,records,counters,expert,slot,capacity,layer,mode);
}
extern "C" void OffloadOpsSparseCopyTrace(uint64_t* src,uint64_t* dst,uint32_t* lens,uint32_t* count,
 uint64_t* records,uint64_t* counters,int64_t* expert,int64_t* slot,
 uint32_t capacity,uint32_t layer,uint32_t mode,void* stream) {
 OffloadSparseCopyTraceOps<<<2,nullptr,stream>>>(reinterpret_cast<GM_ADDR>(src),reinterpret_cast<GM_ADDR>(dst),
 reinterpret_cast<GM_ADDR>(lens),reinterpret_cast<GM_ADDR>(count),reinterpret_cast<GM_ADDR>(records),
 reinterpret_cast<GM_ADDR>(counters),reinterpret_cast<GM_ADDR>(expert),reinterpret_cast<GM_ADDR>(slot),capacity,layer,mode);
}
