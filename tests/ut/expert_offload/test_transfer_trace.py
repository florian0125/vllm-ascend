from types import SimpleNamespace
import pytest
from vllm_ascend.expert_offload.transfer_trace import decode_trace

def fixture(active=True):
 session=SimpleNamespace(capacity=2,mode=2,layer=7,fields=['w13'])
 markers=[[0]*8 for _ in range(2)];records=[[[0]*112 for _ in range(32)] for _ in range(2)]
 for seq in [1,2]:
  ix=seq%2;markers[ix]=[1000,2000,3000,4000,9000,3,1,int(active)]
  for core,r in enumerate(records[ix]):
   n=100 if core==0 and active else 0
   r[:16]=[seq,7,core,3,1,int(n>0),1,n,int(n>0),4000,4100,8000,20,3900,2,0x4d4f4531]
   r[16:28]=[n,int(n>0),20,4200,7900,100,2000,100,500,1900,0x1000,0x2000]
 return session,markers,records,[3]*32,3

def test_ring_retains_tail_and_reports_overwritten_calls():
 rows=list(decode_trace(*fixture()))
 assert rows[0]['total_calls']==3 and rows[0]['overwritten_calls']==1
 assert [r['sequence'] for r in rows[1:]]==[1,2]
 for r in rows[1:]:
  assert r['bytes']==100 and r['active']
  assert r['policy_and_mapping_us']==1 and r['descriptor_build_us']==1
  assert r['descriptor_ready_to_copy_us']==1 and r['copy_device_interval_us']==5
  assert len(r['cores'])==32
  assert r['cores'][0]['fields'][0]['dram_to_ub_ready_wait_us']==1.9

def test_empty_transfer_is_distinct_from_missing_record():
 rows=list(decode_trace(*fixture(False)))
 assert len(rows)==3
 assert all(r['bytes']==0 and not r['active'] for r in rows[1:])

def test_mismatched_core_count_is_rejected():
 args=list(fixture());args[3][2]=2
 with pytest.raises(RuntimeError,match='Incomplete'):list(decode_trace(*args))

def test_invalid_record_magic_is_rejected():
 args=list(fixture());args[2][1][0][15]=0
 with pytest.raises(RuntimeError,match='ABI magic'):list(decode_trace(*args))

def test_out_of_order_device_markers_are_rejected():
 args=list(fixture());args[1][1][2]=0
 with pytest.raises(RuntimeError,match='out of order'):list(decode_trace(*args))

def test_active_byte_mismatch_is_rejected():
 args=list(fixture());args[1][1][7]=0
 with pytest.raises(RuntimeError,match='actual bytes'):list(decode_trace(*args))
