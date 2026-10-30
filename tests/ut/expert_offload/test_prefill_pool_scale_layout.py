"""Prefill pools must retain the checkpoint MXFP scale storage layout."""
from types import SimpleNamespace
import torch
from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager


def test_prefill_pool_preserves_mxfp_scale_physical_order():
    manager=object.__new__(ExpertOffloadManager)
    for name in ("w13","w2","w13_scale","w2_scale","w13_scale_fp32","w13_offset","w2_offset","w13_scale_bias","w2_scale_bias"):
        setattr(manager,"_prefill_"+name,[])
    source13=torch.arange(2*16*8,dtype=torch.int64).to(torch.uint8).reshape(2,16,8)
    source2=torch.arange(2*8*4,dtype=torch.int64).to(torch.uint8).reshape(2,8,4)
    layer=SimpleNamespace(w13_weight=torch.empty(2,8,16,dtype=torch.uint8),w2_weight=torch.empty(2,8,8,dtype=torch.uint8),w13_weight_scale=source13.reshape(2,16,4,2).transpose(1,2),w2_weight_scale=source2.reshape(2,8,2,2).transpose(1,2))
    manager._alloc_prefill_pool_slot(layer,"cpu",torch.uint8,5)
    for name,source in (("w13",source13),("w2",source2)):
        dst=getattr(manager,"_prefill_"+name+"_scale")[0]
        base=source[0].expand(5,*source.shape[1:]).clone()
        logical=base.reshape(5,base.shape[1],base.shape[2]//2,2).transpose(1,2)
        dst.copy_(logical)
        # Logical equality alone passed with the broken contiguous allocation.
        assert torch.equal(dst,logical)
        storage=torch.empty(0,dtype=torch.uint8).set_(dst.untyped_storage(),0,(dst.numel(),),(1,))
        assert torch.equal(storage,base.flatten())


def test_prefill_pool_retains_contiguous_per_channel_scales():
    manager=object.__new__(ExpertOffloadManager)
    for name in ("w13","w2","w13_scale","w2_scale","w13_scale_fp32","w13_offset","w2_offset","w13_scale_bias","w2_scale_bias"):
        setattr(manager,"_prefill_"+name,[])
    layer=SimpleNamespace(w13_weight=torch.empty(2,8,16,dtype=torch.int8),w2_weight=torch.empty(2,8,8,dtype=torch.int8),w13_weight_scale=torch.ones(2,16),w2_weight_scale=torch.ones(2,8))
    manager._alloc_prefill_pool_slot(layer,"cpu",torch.int8,5)
    assert manager._prefill_w13_scale[0].shape==(5,16)
    assert manager._prefill_w13_scale[0].is_contiguous()
    assert manager._prefill_w2_scale[0].is_contiguous()
