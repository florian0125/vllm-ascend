#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Smoke-test MoE expert-weight H2D with MemFabric local DRAM.

This mirrors vLLM Ascend's single-card expert-offload data flow:
MemFabric-backed host w13/w2 weights -> NPU expert slots.  It intentionally
tests a batch of discontiguous expert buffers through ``offload.sparse_copy``.

Run on an Ascend host after sourcing the CANN and MemFabric environments:

    python3 script/test_memfabric_local_h2d.py --device-id 0
"""

import argparse
import sys

import torch


ONE_GIB = 1 << 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--intermediate-size", type=int, default=256)
    parser.add_argument("--reserve-gib", type=int, default=1)
    parser.add_argument("--log-level", type=int, default=3)
    return parser.parse_args()


def _import_npu_dependencies():
    try:
        import torch_npu  # noqa: F401
        import memfabric_hybrid as mf
        from memfabric_hybrid import offload
    except ImportError as exc:
        raise RuntimeError(
            "torch_npu and memfabric_hybrid are required; source their "
            "environment setup scripts before running this smoke test"
        ) from exc
    return mf, offload


def _pattern(expert_id: int, weight_id: int) -> float:
    return float(expert_id * 2 + weight_id + 1)


def main() -> int:
    args = parse_args()
    if min(args.num_experts, args.hidden_size, args.intermediate_size,
           args.reserve_gib) <= 0:
        raise ValueError("shape, expert count, and reserve size must be positive")

    mf, offload = _import_npu_dependencies()
    mf.set_log_level(args.log_level)
    torch.npu.set_device(args.device_id)

    config = offload.OffloadConfig()
    config.device_id = args.device_id
    config.reserve_size = args.reserve_gib * ONE_GIB
    config.alloc_size = config.reserve_size
    ret = offload.initialize(config)
    if ret != 0:
        raise RuntimeError(f"offload.initialize failed: ret={ret}")

    host_weights: list[torch.Tensor] = []
    device_slots: list[torch.Tensor] = []
    expected: list[float] = []
    try:
        shapes = (
            (args.hidden_size, 2 * args.intermediate_size),  # w13
            (args.intermediate_size, args.hidden_size),  # w2
        )
        for expert_id in range(args.num_experts):
            for weight_id, shape in enumerate(shapes):
                value = _pattern(expert_id, weight_id)
                host = offload.empty(shape, dtype=torch.bfloat16).fill_(value)
                slot = torch.empty(shape,
                                   dtype=torch.bfloat16,
                                   device=f"npu:{args.device_id}")
                host_weights.append(host)
                device_slots.append(slot)
                expected.append(value)

        src_ptrs = torch.tensor([item.data_ptr() for item in host_weights],
                                 dtype=torch.int64,
                                 device=f"npu:{args.device_id}")
        dst_ptrs = torch.tensor([item.data_ptr() for item in device_slots],
                                 dtype=torch.int64,
                                 device=f"npu:{args.device_id}")
        lengths = torch.tensor(
            [item.numel() * item.element_size() for item in host_weights],
            dtype=torch.int32,
            device=f"npu:{args.device_id}",
        )
        count = torch.tensor(len(host_weights),
                             dtype=torch.int32,
                             device=f"npu:{args.device_id}")

        ret = offload.sparse_copy(src_ptrs, dst_ptrs, lengths, count,
                                  device_slots[0].device)
        if ret != 0:
            raise RuntimeError(f"offload.sparse_copy failed: ret={ret}")
        torch.npu.synchronize()

        for index, (slot, value) in enumerate(zip(device_slots, expected)):
            reference = torch.full_like(slot, value)
            if not torch.equal(slot, reference):
                raise AssertionError(
                    f"H2D verification failed for buffer {index}")
    finally:
        offload.uninitialize()

    total_bytes = sum(item.numel() * item.element_size()
                      for item in host_weights)
    print(
        "memfabric local H2D OK: "
        f"device={args.device_id}, experts={args.num_experts}, "
        f"buffers={len(host_weights)}, bytes={total_bytes}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
