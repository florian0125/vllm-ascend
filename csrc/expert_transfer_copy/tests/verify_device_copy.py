import os,time,json,statistics,sys
from pathlib import Path
import torch,torch_npu
from vllm_ascend.expert_offload.device_transfer import DeviceSparseCopy
torch.set_num_threads(1);torch.npu.set_device(0)
sizes=[8388608,4194304,524288,262144];total=sum(sizes);rings=8
from vllm_ascend.expert_offload.copy_library import get_default_copy_library
lib=get_default_copy_library()
copier=DeviceSparseCopy(lib)
sources=[[torch.full((n,),17+i*7+j,dtype=torch.uint8,pin_memory=True) for j,n in enumerate(sizes)] for i in range(rings)]
dest=[torch.empty(n,dtype=torch.uint8,device='npu') for n in sizes]
ptrs=copier.map_storages([s for rr in sources for s in rr]);srcptr=[torch.tensor(ptrs[i*4:(i+1)*4],dtype=torch.int64,device='npu') for i in range(rings)]
dstptr=torch.tensor([d.data_ptr() for d in dest],dtype=torch.int64,device='npu');lengths=torch.tensor(sizes,dtype=torch.int32,device='npu');count=torch.tensor(4,dtype=torch.int32,device='npu')
def raw(i):copier.copy(srcptr[i],dstptr,lengths,count)
def native(i):
 for d,s in zip(dest,sources[i]):d.untyped_storage().copy_(s.untyped_storage(),non_blocking=True)
def verify(i):
 torch.npu.synchronize()
 for d,s in zip(dest,sources[i]):assert torch.equal(d.cpu(),s)
def stat(v):
 v=sorted(v);return dict(n=len(v),mean_us=statistics.mean(v),p50_us=statistics.median(v),p90_us=v[int(len(v)*.9)],min_us=v[0],max_us=v[-1])

stream=torch.npu.Stream();graphs={};results={};cases=[('native',native),('fixed_2core',raw)]
with torch.npu.stream(stream):
 for name,fn in cases:
  for i in range(rings):fn(i);verify(i)
  g=torch.npu.NPUGraph()
  with torch.npu.graph(g,stream=stream):
   for i in range(rings):fn(i)
  graphs[name]=g;g.replay();stream.synchronize();verify(rings-1)
 measurements={name:[] for name,_ in cases}
 for repeat in range(120):
  names=list(graphs);names=names[repeat%len(names):]+names[:repeat%len(names)]
  for name in names:
   a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record(stream);graphs[name].replay();b.record(stream);b.synchronize();measurements[name].append(a.elapsed_time(b)*1000/rings)
 for name,values in measurements.items():results[name]=stat(values);results[name]['gbps']=total/(statistics.mean(values)*1000)
 # Dynamic descriptor updates run inside the captured graph, not in a host callback.
 table=torch.stack(srcptr);slots=[[torch.full((n,),211,dtype=torch.uint8,device='npu') for n in sizes] for _ in range(2)]
 dtable=torch.tensor([[d.data_ptr() for d in row] for row in slots],dtype=torch.int64,device='npu')
 eid=torch.tensor([0],dtype=torch.int64,device='npu');sid=torch.tensor([0],dtype=torch.int64,device='npu');active=torch.tensor(1,dtype=torch.int32,device='npu');base=torch.tensor(sizes,dtype=torch.int32,device='npu')
 sp=torch.empty((1,4),dtype=torch.int64,device='npu');dp=torch.empty_like(sp);ls=torch.empty(4,dtype=torch.int32,device='npu')
 dynamic=torch.npu.NPUGraph()
 with torch.npu.graph(dynamic,stream=stream):
  torch.index_select(table,0,eid,out=sp);torch.index_select(dtable,0,sid,out=dp);torch.mul(base,active,out=ls);copier.copy(sp,dp,ls,count)
 expected=[[211]*4 for _ in range(2)]
 for i in range(12):
  e=i%rings;sl=i%2;on=i%3!=0;eid.fill_(e);sid.fill_(sl);active.fill_(int(on));dynamic.replay();stream.synchronize()
  if on:expected[sl]=[17+e*7+j for j in range(4)]
  for slot in range(2):
   for j,d in enumerate(slots[slot]):assert bool((d.cpu()==expected[slot][j]).all())
 print('DYNAMIC_EXPERT_SLOT_AND_ZERO_LENGTH_GRAPH_PASS',flush=True)
 # Byte tails and guard regions, including zero-size descriptors and count=0.
 ns=[0,1,31,32,33,65535,65536,65537,131073]
 ss=[torch.full((max(1,n),),i+3,dtype=torch.uint8,pin_memory=True) for i,n in enumerate(ns)]
 dd=[torch.full((n+64,),199,dtype=torch.uint8,device='npu') for n in ns];pp=copier.map_storages(ss)
 a=torch.tensor(pp,dtype=torch.int64,device='npu');b=torch.tensor([d.data_ptr()+32 for d in dd],dtype=torch.int64,device='npu');l=torch.tensor(ns,dtype=torch.int32,device='npu');c=torch.tensor(len(ns),dtype=torch.int32,device='npu');copier.copy(a,b,l,c);stream.synchronize()
 for i,(d,n) in enumerate(zip(dd,ns)):
  host=d.cpu();assert bool((host[:32]==199).all()) and bool((host[32+n:]==199).all());assert bool((host[32:32+n]==i+3).all())
 c.zero_();copier.copy(a,b,l,c);stream.synchronize();print('TAIL_GUARDS_ZERO_FIELDS_AND_ZERO_COUNT_PASS',flush=True)
 del dynamic;graphs.clear();stream.synchronize()
out={'bytes':total,'results':results,'validation':['all 8 source experts and 4 fields byte equality','12 graph replays with device-side expert and slot selection; inactive transfers preserve both slots','9 tail sizes with guard checks; zero field lengths and zero descriptor count'],'note':'120 alternating rounds, each graph copies 8 experts; no profiler or conversion; pinned node0 sources; host system load is not controlled'}
(Path(sys.argv[1]) if len(sys.argv)>1 else Path('copy_verification.json')).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2),flush=True);copier.close()
