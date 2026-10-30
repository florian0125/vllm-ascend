
#ifndef EXPERT_TRANSFER_TRACE_H
#define EXPERT_TRANSFER_TRACE_H
#include "kernel_operator.h"
constexpr uint32_t TRACE_CORES = 2;
constexpr uint32_t TRACE_COPY_BYTES = 65536;
constexpr uint32_t TRACE_FIELDS = 8;
constexpr uint32_t TRACE_HEADER = 16;
constexpr uint32_t TRACE_FIELD_WORDS = 12;
constexpr uint32_t TRACE_WORDS = TRACE_HEADER + TRACE_FIELDS * TRACE_FIELD_WORDS;

__aicore__ inline void TraceFlush(__gm__ uint64_t* pointer) {
 AscendC::GlobalTensor<uint64_t> global;global.SetGlobalBuffer(pointer);
 AscendC::DataCacheCleanAndInvalid<uint64_t,AscendC::CacheLine::SINGLE_CACHE_LINE,
                                  AscendC::DcciDst::CACHELINE_OUT>(global);
}

// Separate diagnostic kernel: the normal kernel and ABI are unchanged.
// Mode 1 adds only field-boundary fences. Mode 2 additionally serializes each
// load/store for attribution; its durations are not production throughput.
class SparseCopyTraceKernel {
public:
 __aicore__ inline void Run(GM_ADDR inputs, GM_ADDR outputs, GM_ADDR lens, GM_ADDR size,
                           GM_ADDR records, GM_ADDR counters, GM_ADDR expert, GM_ADDR slot,
                           uint32_t capacity, uint32_t layer, uint32_t mode) {
  uint64_t start = AscendC::GetSystemCycle();
  uint32_t block = AscendC::GetBlockIdx();
  auto counter = reinterpret_cast<__gm__ uint64_t*>(counters);
  TraceFlush(counter+block*8);
  uint64_t seq = counter[block*8]; counter[block*8] = seq + 1;
  auto rec = reinterpret_cast<__gm__ uint64_t*>(records)
      + ((seq % capacity) * TRACE_CORES + block) * TRACE_WORDS;
  auto src = reinterpret_cast<__gm__ uint64_t*>(inputs);
  auto dst = reinterpret_cast<__gm__ uint64_t*>(outputs);
  auto lengths = reinterpret_cast<__gm__ uint32_t*>(lens);
  uint32_t count = *reinterpret_cast<__gm__ uint32_t*>(size);
  pipe_.InitBuffer(queue_,1,TRACE_COPY_BYTES);
  uint64_t initialized = AscendC::GetSystemCycle();
  uint64_t totalBytes=0,totalTiles=0,totalDesc=0,totalLoad=0,totalStore=0;
  for(uint32_t field=0;field<count && field<TRACE_FIELDS;++field) {
   uint64_t md0=AscendC::GetSystemCycle();
   uint32_t length=lengths[field]; uint64_t sourcePtr=src[field],destPtr=dst[field];
   uint64_t md1=AscendC::GetSystemCycle();
   uint64_t bytesDone=0,tiles=0,loadIssue=0,loadTotal=0,storeIssue=0,storeTotal=0,queueCost=0;
   uint64_t fieldStart=AscendC::GetSystemCycle();
   for(uint64_t offset=block*TRACE_COPY_BYTES;offset<length;offset+=TRACE_CORES*TRACE_COPY_BYTES) {
    uint32_t bytes=static_cast<uint32_t>(length-offset);
    bytes=bytes>TRACE_COPY_BYTES?TRACE_COPY_BYTES:bytes;
    AscendC::GlobalTensor<uint8_t> source,destination;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(sourcePtr)+offset,bytes);
    destination.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(destPtr)+offset,bytes);
    auto local=queue_.AllocTensor<uint8_t>();
    AscendC::DataCopyExtParams params(1,bytes,0,0,0);
    AscendC::DataCopyPadExtParams<uint8_t> pad(false,0,0,0);
    uint64_t a=0,b=0,c=0,d=0,e=0;
    if(mode==2) { AscendC::PipeBarrier<PIPE_ALL>(); a=AscendC::GetSystemCycle(); }
    AscendC::DataCopyPad(local,source,params,pad);
    if(mode==2) b=AscendC::GetSystemCycle();
    queue_.EnQue(local); local=queue_.DeQue<uint8_t>();
    if(mode==2) { AscendC::PipeBarrier<PIPE_ALL>(); c=AscendC::GetSystemCycle(); }
    AscendC::DataCopyPad(destination,local,params);
    if(mode==2) { d=AscendC::GetSystemCycle(); AscendC::PipeBarrier<PIPE_ALL>(); e=AscendC::GetSystemCycle(); }
    queue_.FreeTensor(local);
    if(mode==2) { loadIssue+=b-a;loadTotal+=c-a;storeIssue+=d-c;storeTotal+=e-c;queueCost+=c-b; }
    bytesDone+=bytes; ++tiles;
   }
   AscendC::PipeBarrier<PIPE_ALL>();
   uint64_t fieldEnd=AscendC::GetSystemCycle();
   auto f=rec+TRACE_HEADER+field*TRACE_FIELD_WORDS;
   f[0]=bytesDone;f[1]=tiles;f[2]=md1-md0;f[3]=fieldStart;f[4]=fieldEnd;
   f[5]=loadIssue;f[6]=loadTotal;f[7]=storeIssue;f[8]=storeTotal;f[9]=queueCost;
   f[10]=sourcePtr;f[11]=destPtr;
   totalBytes+=bytesDone;totalTiles+=tiles;totalDesc+=md1-md0;totalLoad+=loadTotal;totalStore+=storeTotal;
  }
  AscendC::PipeBarrier<PIPE_ALL>(); uint64_t end=AscendC::GetSystemCycle();
  rec[0]=seq;rec[1]=layer;rec[2]=block;
  rec[3]=*reinterpret_cast<__gm__ uint64_t*>(expert);
  rec[4]=*reinterpret_cast<__gm__ uint64_t*>(slot);
  rec[5]=totalBytes>0;rec[6]=count;rec[7]=totalBytes;rec[8]=totalTiles;
  rec[9]=start;rec[10]=initialized;rec[11]=end;rec[12]=totalDesc;
  rec[13]=end-initialized;rec[14]=mode;rec[15]=0x4d4f4531;
  counter[block*8+1]+=totalBytes;counter[block*8+2]+=totalTiles;
  counter[block*8+3]+=(totalBytes>0);counter[block*8+4]+=end-start;
  counter[block*8+5]+=totalLoad;counter[block*8+6]+=totalStore;counter[block*8+7]+=totalDesc;
  for(uint32_t word=0;word<TRACE_WORDS;word+=8)TraceFlush(rec+word);
  TraceFlush(counter+block*8);
 }
private:
 AscendC::TPipe pipe_;
 AscendC::TQueBind<AscendC::TPosition::VECIN,AscendC::TPosition::VECOUT,1> queue_;
};


#endif
