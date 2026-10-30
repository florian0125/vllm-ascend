"""Bounded device-side transfer diagnostics. No D2H in the replay path."""
import ctypes
import gzip
import json
import os
import time
import weakref
from pathlib import Path
import torch

_sessions=weakref.WeakSet()
TICKS_PER_US=1000  # This library is compiled for Ascend950PR, not 910.
WORDS=112

class TransferTrace:
    def __init__(self,transfer,capacity,mode):
        from vllm_ascend.expert_offload.cpu_backend.kt_kernel_backend import _graph_capture_active
        if _graph_capture_active():
            raise RuntimeError('Initialize transfer diagnostics before NPU graph capture')
        self.transfer=weakref.ref(transfer)
        self.capacity,self.mode=capacity,mode
        self.layer=transfer.layer_idx
        self.device=transfer.device
        self.fields=list(transfer._field_names)
        if len(self.fields)>8:raise ValueError('Transfer trace supports at most 8 fields')
        lib=transfer.copier.library
        core_count=getattr(lib,"AccOffloadSparseCopyTraceCoreCount",None)
        if core_count is not None:
            core_count.argtypes=[];core_count.restype=ctypes.c_uint32
        self.cores=int(core_count()) if core_count is not None else 32
        if not 1<=self.cores<=128:raise ValueError("Invalid transfer trace core count")
        self.mark_launch=lib.AccExpertTransferMark
        self.mark_launch.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,
                                   ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint8]
        self.mark_launch.restype=None
        self.copy_launch=lib.AccOffloadSparseCopyTrace
        self.copy_launch.argtypes=[ctypes.c_void_p]*8+[ctypes.c_uint32]*3+[ctypes.c_uint8]
        self.copy_launch.restype=None
        self.markers=torch.zeros((capacity,8),dtype=torch.int64,device=self.device)
        self.marker_count=torch.zeros(8,dtype=torch.int64,device=self.device)
        self.records=torch.zeros((capacity,self.cores,WORDS),dtype=torch.int64,device=self.device)
        self.counts=torch.zeros((self.cores,8),dtype=torch.int64,device=self.device)
        _sessions.add(self)

    def mark(self,phase,candidate=None,slot=None,active=None):
        ptr=lambda value:0 if value is None else value.data_ptr()
        self.mark_launch(self.markers.data_ptr(),self.marker_count.data_ptr(),self.capacity,phase,
                         ptr(candidate),ptr(slot),ptr(active),self.device.index)

    def copy(self,task,count):
        self.copy_launch(task.sources.data_ptr(),task.destinations.data_ptr(),task.lengths.data_ptr(),
                         count.data_ptr(),self.records.data_ptr(),self.counts.data_ptr(),
                         task.candidate.data_ptr(),task.slot.data_ptr(),self.capacity,self.layer,self.mode,self.device.index)

def initialize_transfer_traces():
    from vllm_ascend.expert_offload.expert_offload_manager import has_expert_offload_manager, get_expert_offload_manager
    if not has_expert_offload_manager():return
    manager=get_expert_offload_manager()
    if not getattr(manager.offload_config.cpu_moe,'transfer_trace_capacity',0):return
    from vllm_ascend.expert_offload.hybrid_executor import get_hybrid_expert_executor
    from vllm_ascend.expert_offload.device_transfer import DeviceExpertTransfer
    for layer in manager.moe_layers:
        executor=get_hybrid_expert_executor(layer)
        if executor is not None and executor._weight_transfer is None:
            executor._weight_transfer=DeviceExpertTransfer(manager,layer)
    torch.npu.synchronize()

def reset_transfer_traces():
    if not _sessions:return
    torch.npu.synchronize()
    for session in list(_sessions):
        session.marker_count.zero_();session.counts.zero_()
        session.markers.zero_();session.records.zero_()
    torch.npu.synchronize()

def decode_trace(session,markers,records,counts,marker_count):
    if len(set(counts))!=1 or counts[0]!=marker_count:
        raise RuntimeError(f'Incomplete transfer trace: cores={counts}, markers={marker_count}')
    count=counts[0]
    yield {'event':'npu_expert_transfer.trace_status','layer':session.layer,'total_calls':count,
           'retained_calls':min(count,session.capacity),'overwritten_calls':max(0,count-session.capacity),
           'capacity':session.capacity,'core_count':len(counts),'mode':session.mode,'ticks_per_us':TICKS_PER_US,
           'diagnostic_copy_waves':1,
           'timing_note':'Diagnostic copy uses its own unsliced kernel; measure production scheduling with the profiler. Mode2 adds per-tile fences. Marker intervals include marker dispatch overhead.'}
    us=lambda ticks:ticks/TICKS_PER_US
    for seq in range(max(0,count-session.capacity),count):
        index=seq%session.capacity;m=markers[index];blocks=records[index]
        if not all(a <= b for a,b in zip(m[:5],m[1:5])):
            raise RuntimeError("Device marker timestamps are out of order")
        cores=[]
        for block,r in enumerate(blocks):
            if r[0]!=seq or r[15]!=0x4d4f4531:raise RuntimeError('Invalid trace record sequence or ABI magic')
            fields=[]
            for field,name in enumerate(session.fields):
                a=r[16+field*12:16+(field+1)*12]
                fields.append(dict(name=name,bytes=a[0],tiles=a[1],descriptor_us=us(a[2]),
                    field_start_ticks=a[3],field_end_ticks=a[4],field_span_us=us(a[4]-a[3]),
                    load_issue_us=us(a[5]),dram_to_ub_including_ready_wait_us=us(a[6]),
                    store_issue_us=us(a[7]),ub_to_hbm_including_completion_wait_us=us(a[8]),
                    dram_to_ub_ready_wait_us=us(a[6]-a[5]),
                    ub_to_hbm_completion_wait_us=us(a[8]-a[7]),
                    load_queue_and_ready_wait_us=us(a[9]),source_address=hex(a[10]),destination_address=hex(a[11])))
            cores.append(dict(core=block,bytes=r[7],tiles=r[8],start_ticks=r[9],end_ticks=r[11],
                init_us=us(r[10]-r[9]),elapsed_us=us(r[11]-r[9]),descriptor_us=us(r[12]),fields=fields))
        bytes_done=sum(c['bytes'] for c in cores)
        if bool(m[7]) != bool(bytes_done):raise RuntimeError("Active flag does not match actual bytes")
        copy_us=us(m[4]-m[3]);inner_span=us(max(c['end_ticks'] for c in cores)-min(c['start_ticks'] for c in cores))
        fields=[dict(name=name,bytes=sum(c['fields'][i]['bytes'] for c in cores),
                     max_core_span_us=max(c['fields'][i]['field_span_us'] for c in cores))
                for i,name in enumerate(session.fields)]
        yield dict(event='npu_expert_transfer',layer=session.layer,sequence=seq,expert=m[5],slot=m[6],
            active=bool(m[7]),bytes=bytes_done,tiles=sum(c['tiles'] for c in cores),
            policy_and_mapping_us=us(m[1]-m[0]),descriptor_build_us=us(m[2]-m[1]),
            descriptor_ready_to_copy_us=us(m[3]-m[2]),copy_device_interval_us=copy_us,
            copy_inner_core_span_us=inner_span,effective_gbps=bytes_done/(copy_us*1000) if copy_us else 0,
            critical_core=max(cores,key=lambda c:c['elapsed_us'])['core'],fields=fields,cores=cores,
            marker_ticks=m[:5],mode=session.mode)

def source_layout(session,expert):
    transfer=session.transfer()
    if transfer is None:return []
    result=[]
    for i,buffers in enumerate(transfer._sources):
        tensor=buffers[expert]
        storage=tensor.untyped_storage()
        pointer=storage.data_ptr() if i<2 else tensor.data_ptr()
        length=storage.nbytes() if i<2 else tensor.numel()*tensor.element_size()
        record=dict(field=session.fields[i],cpu_source_address=hex(pointer),storage_bytes=storage.nbytes(),
                    expert_field_bytes=length,pinned=bool(tensor.is_pinned()),cpu_shape=list(tensor.shape),
                    cpu_stride=list(tensor.stride()),numa_sample_timing='after capture; start/middle/end pages')
        if i<2:
            try:
                lib=ctypes.CDLL('libnuma.so.1',use_errno=True)
                query=lib.numa_move_pages
                query.argtypes=[ctypes.c_int,ctypes.c_ulong,ctypes.POINTER(ctypes.c_void_p),
                                ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.c_int]
                query.restype=ctypes.c_long
                page=os.sysconf('SC_PAGE_SIZE')
                addresses=sorted(set(((pointer+offset)//page)*page for offset in (0,length//2,length-1)))
                pages=(ctypes.c_void_p*len(addresses))(*addresses)
                statuses=(ctypes.c_int*len(addresses))(*([-999]*len(addresses)))
                rc=query(0,len(addresses),pages,None,statuses,0)
                record.update(numa_query_rc=rc,numa_query_errno=ctypes.get_errno() if rc<0 else 0,
                              numa_page_nodes_or_errors=list(statuses))
            except Exception as error:record['numa_query_error']=str(error)
        result.append(record)
    return result

def dump_transfer_traces(directory):
    if not _sessions:return None
    torch.npu.synchronize()
    path=Path(directory)/f'npu_expert_transfer_{os.getpid()}_{time.time_ns()}.jsonl.gz'
    summary={'event':'npu_expert_transfer.summary','layers':len(_sessions),'calls':0,'active_calls':0,
             'bytes':0,'policy_and_mapping_total_us':0.0,'copy_device_total_us':0.0,'overwritten_calls':0,'retained_calls':0}
    with gzip.open(path,'wt',encoding='utf-8') as output:
        output.write(json.dumps(dict(event='npu_expert_transfer.environment',pid=os.getpid(),
            capture_wall_ns=time.time_ns(),physical_devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
            torch_version=torch.__version__,cpu_affinity=sorted(os.sched_getaffinity(0)),
            ticks_per_us=TICKS_PER_US,architecture='Ascend950PR',schema_version=1))+'\n')
        for session in sorted(_sessions,key=lambda s:s.layer):
            core_totals=session.counts.cpu().tolist()
            counts=[core[0] for core in core_totals]
            marker_totals=session.marker_count.cpu().tolist()
            retained=min(counts[0],session.capacity)
            data=decode_trace(session,session.markers[:retained].cpu().tolist(),session.records[:retained].cpu().tolist(),
                              counts,int(marker_totals[0]))
            layouts={}
            for record in data:
                if record["event"]=="npu_expert_transfer":
                    expert=record["expert"]
                    if expert not in layouts:layouts[expert]=source_layout(session,expert)
                    record["host_source_layout"]=layouts[expert]
                if record['event'].endswith('trace_status'):
                    record.update(total_active_calls=marker_totals[1],total_bytes=sum(c[1] for c in core_totals),
                        policy_and_mapping_total_us=marker_totals[2]/TICKS_PER_US,
                        descriptor_build_total_us=marker_totals[3]/TICKS_PER_US,
                        descriptor_ready_to_copy_total_us=marker_totals[4]/TICKS_PER_US,
                        copy_device_total_us=marker_totals[5]/TICKS_PER_US,
                        per_core_cumulative_words=core_totals,
                        cumulative_word_schema=['calls','bytes','tiles','core_nonempty_calls','elapsed_ticks','load_ticks','store_ticks','descriptor_ticks'])
                    summary['calls']+=record['total_calls'];summary['active_calls']+=record['total_active_calls']
                    summary['bytes']+=record['total_bytes'];summary['overwritten_calls']+=record['overwritten_calls']
                    summary['policy_and_mapping_total_us']+=record['policy_and_mapping_total_us']
                    summary['copy_device_total_us']+=record['copy_device_total_us']
                else:summary['retained_calls']+=1
                output.write(json.dumps(record,separators=(',',':'))+'\n')
        summary['average_policy_and_mapping_us']=summary['policy_and_mapping_total_us']/summary['calls'] if summary['calls'] else 0
        summary['average_copy_device_us']=summary['copy_device_total_us']/summary['calls'] if summary['calls'] else 0
        summary['active_ratio']=summary['active_calls']/summary['calls'] if summary['calls'] else 0
        output.write(json.dumps(summary)+'\n')
    path.with_suffix('').with_suffix('.summary.json').write_text(json.dumps(summary,indent=2))
    return str(path)
