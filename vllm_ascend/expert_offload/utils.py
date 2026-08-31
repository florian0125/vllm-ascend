"""Utility functions for expert offload initialization."""

import torch


def init_expert_offload_config(offload_config, num_experts: int,
                               layer_idx: int = 0):
    """Prepare pre-super().__init__() expert offload setup.

    Builds an expert_map_offload that the upstream layer.py hook reads
    to shrink the device weight and skip loading cold experts. Single-card
    hot preload maps the ranked global expert IDs directly to initial slots;
    otherwise the ordinary contiguous ``[0, ndev)`` placement is retained.

    Args:
        offload_config: ExpertOffloadConfig instance from AscendConfig.
        num_experts: Total routed expert count from model config.
        layer_idx: MoE-layer registration index, used to resolve a per-layer
            num_device_experts (config may be a list). Scalars broadcast.

    Returns:
        (enable: bool, expert_map_offload: torch.Tensor | None,
         num_device_experts: int)
    """
    ndev = offload_config.num_device_experts_for_layer(layer_idx)
    enable = (offload_config.expert_offload
              and ndev > 0
              and ndev < num_experts)
    if not enable:
        return False, None, ndev

    initial_eids = offload_config.initial_device_experts_for_layer(
        layer_idx, num_experts)
    emap = torch.full((num_experts,), -1, dtype=torch.int32)
    for slot, eid in enumerate(initial_eids):
        emap[eid] = slot
    return True, emap, ndev


def init_log2phy_for_offload(
    global_num_experts: int,
    num_device_experts: int,
    expert_map: torch.Tensor | None = None,
):
    """Initialize the forward-pass log2phy mapping table."""
    if expert_map is not None:
        if expert_map.shape != (global_num_experts,):
            raise ValueError(
                "expert_map must cover every global expert; "
                f"expected={global_num_experts}, actual={expert_map.shape}")
        resident_count = int((expert_map >= 0).sum())
        if resident_count != num_device_experts:
            raise ValueError(
                "expert_map resident count does not match device capacity; "
                f"expected={num_device_experts}, actual={resident_count}")
        return expert_map.to(device="npu", dtype=torch.int32)
    log2phy = torch.arange(global_num_experts, dtype=torch.int32, device='npu')
    log2phy[num_device_experts:].fill_(-1)
    return log2phy


def init_log2phy_for_offload_multi_card(
    global_num_experts: int,
    num_device_experts: int,
    ep_size: int,
    ep_rank: int,
    device: str = "npu",
) -> torch.Tensor:
    """Multi-card initial log2phy for offload: rank r owns a contiguous shard.

    Static initial resident set only. Rank ``r`` maps its shard
    ``[r*num_device_experts : (r+1)*num_device_experts]`` to local slots
    ``[0 : num_device_experts]``, recording physical id
    ``ep_rank * num_device_experts + slot``. Everything else is -1 (not
    resident). ``multi_card_planner`` overrides this dynamically every layer
    once forward starts (see stage 2).

    Physical-id encoding matches what the MC2 dispatcher expects
    (``moe_comm_method.py`` applies log2phy to topk_ids before dispatch;
    ``physical_id // num_device_experts`` yields the target rank).
    """
    log2phy = torch.full((global_num_experts,), -1, dtype=torch.int32, device=device)
    start = ep_rank * num_device_experts
    end = min(start + num_device_experts, global_num_experts)
    for slot, eid in enumerate(range(start, end)):
        log2phy[eid] = ep_rank * num_device_experts + slot
    return log2phy
