"""Utility functions for expert offload initialization."""

import torch


def init_expert_offload_config(offload_config, num_experts: int,
                               layer_idx: int = 0):
    """Prepare pre-super().__init__() expert offload setup.

    Builds an expert_map_offload that the upstream layer.py hook reads
    to shrink the device weight and skip loading cold experts.

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

    emap = torch.full((num_experts,), -1, dtype=torch.int32)
    emap[:ndev] = torch.arange(ndev, dtype=torch.int32)
    return True, emap, ndev


def init_log2phy_for_offload(global_num_experts: int, num_device_experts: int):
    """Initialize the forward-pass log2phy mapping table."""
    log2phy = torch.arange(global_num_experts, dtype=torch.int32, device='npu')
    log2phy[num_device_experts:].fill_(-1)
    return log2phy
