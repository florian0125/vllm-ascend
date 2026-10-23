import argparse
import torch, torch_npu, acl

parser = argparse.ArgumentParser(description="A5 DeepSeek-V4 native expert copy byte-parity check")
parser.add_argument("--library", required=True)
parser.add_argument("--checkpoint", default="/mnt/share/DeepSeek-V4-Flash-0731")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--scale-layout", choices=("row_major", "column_pair"), default="row_major")
args = parser.parse_args()
from types import SimpleNamespace
from kt_kernel.utils.native_mxfp4 import NativeMXFP4Weights
from vllm_ascend.expert_offload.device_transfer import DeviceExpertTransfer
from vllm_ascend.expert_offload.native_mxfp4_h2d import NativeMXFP4H2DTransport, NativeExpertUploadTask
from vllm_ascend.expert_offload.h2d_transfer import TorchCopyH2DTransport
from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import apply_mxfp4_weight_scale_layout as layout
torch.set_num_threads(1); torch.npu.set_device(args.device)
o=NativeMXFP4Weights.from_checkpoint(args.checkpoint,16,3,4096,2048,numa_node=1)
def cast(x): return torch_npu.npu_format_cast(x,29,customize_dtype=torch.float8_e4m3fn,input_dtype=torch_npu.float4_e2m1fn_x2)
def make_layer():
 return SimpleNamespace(w13_weight=cast(torch.zeros((2,4096,2048),dtype=torch.uint8,device='npu')).transpose(1,2),w2_weight=cast(torch.zeros((2,4096,1024),dtype=torch.uint8,device='npu')).transpose(1,2),w13_weight_scale=layout(torch.zeros((2,4096,128),dtype=torch.uint8,device='npu')),w2_weight_scale=layout(torch.zeros((2,4096,64),dtype=torch.uint8,device='npu')))
def read(t):
 s=t.untyped_storage(); h=torch.empty(s.nbytes(),dtype=torch.uint8,pin_memory=True);torch.npu.synchronize();assert acl.rt.memcpy(h.data_ptr(),h.numel(),s.data_ptr(),s.nbytes(),2)==0; return h
layer=make_layer(); reference=make_layer()
if args.scale_layout == "column_pair":
 for target in (layer, reference):
  for field in ('w13_weight_scale', 'w2_weight_scale'):
   setattr(target, field, getattr(target, field).contiguous())
m=SimpleNamespace(native_mxfp4_enabled=True,ep_size=1,moe_layers=[layer],_native_owners={0:o},num_device_experts_for_layer=lambda _:2,w13_expert_size_bytes=layer.w13_weight.untyped_storage().nbytes()//2,w2_expert_size_bytes=layer.w2_weight.untyped_storage().nbytes()//2,offload_config=SimpleNamespace(cpu_moe=SimpleNamespace(transfer_policy='first_miss',transfer_copy_library=args.library)))
t=DeviceExpertTransfer(m,layer); old=NativeMXFP4H2DTransport(TorchCopyH2DTransport())
ids=torch.tensor([[0]],dtype=torch.int64,device='npu'); mapping=torch.full((3,),-1,dtype=torch.int32,device='npu')
def old_upload(e,s):
 n13=m.w13_expert_size_bytes;n2=m.w2_expert_size_bytes
 old.copy_batch([NativeExpertUploadTask(o,e,reference.w13_weight.untyped_storage()[s*n13:(s+1)*n13],reference.w2_weight.untyped_storage()[s*n2:(s+1)*n2],reference.w13_weight_scale[s],reference.w2_weight_scale[s])])
def verify():
 for field in ('w13_weight','w2_weight','w13_weight_scale','w2_weight_scale'):
  assert torch.equal(read(getattr(layer,field)),read(getattr(reference,field))),field
for e in (0,1,2,0):
 ids.fill_(e); task=t.prepare(ids,mapping); task.wait_for_weights(None); old_upload(e,0);verify()
print('NATIVE_EAGER_REPLACEMENT_BYTES_PASS',flush=True)
graph=torch.npu.NPUGraph(); stream=torch.npu.Stream(); stream.wait_stream(torch.npu.current_stream())
with torch.npu.graph(graph,stream=stream):
 task=t.prepare(ids,mapping);task.wait_for_weights(None)
torch.npu.current_stream().wait_stream(stream)
# Capture changes buffers once; initialize a known state before replay comparisons.
for e in (1,1,2,0,2,2):
 ids.fill_(e);graph.replay();old_upload(e,0);verify()
print('NATIVE_GRAPH_DYNAMIC_REPLACEMENT_BYTES_PASS',flush=True)
# Exercise slot1, protected resident slots, both-hit no-op, and graph batch2.
ids2=torch.tensor([[2,1]],dtype=torch.int64,device='npu')
for routes in ([2,1],[1,0],[0,2],[0,2]):
 ids2.copy_(torch.tensor([routes],dtype=torch.int64,device='npu'))
 task=t.prepare(ids2,mapping);task.wait_for_weights(None)
 if bool(task.active.cpu()):old_upload(int(task.candidate.cpu()),int(task.slot.cpu()))
 verify()
graph2=torch.npu.NPUGraph();stream2=torch.npu.Stream();stream2.wait_stream(torch.npu.current_stream())
with torch.npu.graph(graph2,stream=stream2):
 task2=t.prepare(ids2,mapping);task2.wait_for_weights(None)
torch.npu.current_stream().wait_stream(stream2)
# Capture records operations; materialize outputs before reading them.
graph2.replay()
if bool(task2.active.cpu()):old_upload(int(task2.candidate.cpu()),int(task2.slot.cpu()))
verify()
for routes in ([2,1],[1,0],[0,2],[0,2]):
 ids2.copy_(torch.tensor([routes],dtype=torch.int64,device='npu'));graph2.replay()
 if bool(task2.active.cpu()):old_upload(int(task2.candidate.cpu()),int(task2.slot.cpu()))
 verify()
print('FUSED_SLOT1_PROTECTION_NOOP_GRAPH_BATCH2_PASS',flush=True)
if args.benchmark:
    # Time a fixed admitted upload by NPU events around100 graph replays.
    from types import SimpleNamespace
    bench_task=SimpleNamespace(candidate=torch.tensor(1,dtype=torch.int64,device='npu'),slot=torch.tensor(0,dtype=torch.int64,device='npu'),active=torch.tensor(True,device='npu'))
    bench_graph=torch.npu.NPUGraph();bs=torch.npu.Stream();bs.wait_stream(torch.npu.current_stream())
    with torch.npu.graph(bench_graph,stream=bs):t.upload(bench_task)
    torch.npu.current_stream().wait_stream(bs)
    for _ in range(10):bench_graph.replay()
    begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
    begin.record()
    for _ in range(100):bench_graph.replay()
    end.record();end.synchronize()
    print('FUSED_UPLOAD_AVERAGE_US',begin.elapsed_time(end)*1000/100,flush=True)
    bench_task.active.fill_(False)
    begin.record()
    for _ in range(100):bench_graph.replay()
    end.record();end.synchronize()
    print('FUSED_NOOP_AVERAGE_US',begin.elapsed_time(end)*1000/100,flush=True)
torch.npu.synchronize();t.copier.close()
