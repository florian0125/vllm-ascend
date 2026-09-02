#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Smoke-test the Torch/HCCL prefill byte-buffer path on Ascend 950DT.

This is a focused prerequisite test, not a full model accuracy test. It checks
the exact primitives used by exclusive_dynamic prefill:

* internal-format-style weights are packed and unpacked through raw Storage;
* an MXFP4-shaped non-contiguous uint8 scale uses stride-aware typed copy_;
* HCCL sees only contiguous, storage_offset=0 uint8 communication tensors;
* bounded communication buffers are reused across sequential chunks.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


MIB = 1 << 20
SCALE_SHAPE_BEFORE_TRANSPOSE = (1, 4096, 64, 2)


@dataclass(frozen=True)
class Component:
    name: str
    source: object
    destination: object
    nbytes: int
    dtype: torch.dtype | None = None
    shape: tuple[int, ...] | None = None
    element_size: int = 1

    @property
    def typed(self) -> bool:
        return self.dtype is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w13-bytes", type=int, default=8 * MIB)
    parser.add_argument("--w2-bytes", type=int, default=4 * MIB)
    parser.add_argument(
        "--chunk-bytes",
        type=int,
        default=8 * MIB,
        help="soft per-direction buffer limit; a component is never split",
    )
    parser.add_argument("--iterations", type=int, default=3)
    return parser.parse_args()


def rank_environment() -> tuple[int, int, int]:
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            "launch with torchrun; missing " + ", ".join(missing))
    return tuple(int(os.environ[name]) for name in required)


def layout(entries: list[Component]) -> tuple[list[tuple[Component, int]], int]:
    result = []
    offset = 0
    for component in entries:
        alignment = component.element_size
        offset = ((offset + alignment - 1) // alignment) * alignment
        result.append((component, offset))
        offset += component.nbytes
    return result, offset


def chunk_components(
    entries: list[Component], max_bytes: int
) -> list[tuple[list[tuple[Component, int]], int]]:
    if max_bytes <= 0:
        raise ValueError("chunk-bytes must be positive")
    chunks = []
    current = []
    for entry in entries:
        candidate = current + [entry]
        _, candidate_bytes = layout(candidate)
        if current and candidate_bytes > max_bytes:
            chunks.append(layout(current))
            current = [entry]
        else:
            current = candidate
    if current:
        chunks.append(layout(current))
    return chunks


def component_buffer_view(
    buffer: torch.Tensor, offset: int, component: Component
):
    if component.typed:
        if component.dtype is None or component.shape is None:
            raise AssertionError("typed component is missing dtype or shape")
        if offset % component.element_size:
            raise AssertionError("typed component is misaligned")
        return buffer.narrow(0, offset, component.nbytes).view(
            component.dtype).reshape(component.shape)
    return buffer.untyped_storage()[offset:offset + component.nbytes]


def scale_pattern(rank: int) -> torch.Tensor:
    numel = 1
    for dim in SCALE_SHAPE_BEFORE_TRANSPOSE:
        numel *= dim
    flat = (torch.arange(numel, dtype=torch.int32) + rank * 17) % 251
    return flat.to(torch.uint8).reshape(
        SCALE_SHAPE_BEFORE_TRANSPOSE).transpose(-3, -2)[0]


def assert_communication_tensor(buffer: torch.Tensor) -> None:
    if (buffer.dtype != torch.uint8 or not buffer.is_contiguous()
            or buffer.storage_offset() != 0):
        raise AssertionError(
            "HCCL buffer must be contiguous zero-offset uint8: "
            f"dtype={buffer.dtype}, contiguous={buffer.is_contiguous()}, "
            f"storage_offset={buffer.storage_offset()}")


def main() -> int:
    args = parse_args()
    if args.w13_bytes <= 0 or args.w2_bytes <= 0:
        raise ValueError("weight byte counts must be positive")
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    rank, local_rank, world_size = rank_environment()
    if world_size < 2:
        raise RuntimeError("prefill P2P validation requires at least two ranks")

    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "torch_npu is required; source the CANN environment first") from exc

    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    device = torch.device(f"npu:{local_rank}")
    send_peer = (rank + 1) % world_size
    recv_peer = (rank - 1 + world_size) % world_size

    try:
        send_dummy = torch.empty(1, dtype=torch.uint8, device=device)
        recv_dummy = torch.empty(1, dtype=torch.uint8, device=device)
        requests = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, send_dummy, send_peer),
            dist.P2POp(dist.irecv, recv_dummy, recv_peer),
        ])
        for request in requests:
            request.wait()

        w13_value = (rank * 31 + 11) % 251
        w2_value = (rank * 31 + 19) % 251
        source_w13 = torch.full(
            (args.w13_bytes,), w13_value, dtype=torch.uint8, device=device)
        source_w2 = torch.full(
            (args.w2_bytes,), w2_value, dtype=torch.uint8, device=device)
        scale_cpu = scale_pattern(rank)
        scale_base = scale_cpu.transpose(-3, -2).contiguous().unsqueeze(0)
        source_scale = scale_base.to(device).transpose(-3, -2)[0]
        if source_scale.is_contiguous():
            raise AssertionError(
                f"MXFP4 scale must be non-contiguous; stride={source_scale.stride()}")

        destination_w13 = torch.empty_like(source_w13)
        destination_w2 = torch.empty_like(source_w2)
        destination_scale = torch.empty(
            source_scale.shape, dtype=source_scale.dtype, device=device)
        components = [
            Component(
                "w13",
                source_w13.untyped_storage(),
                destination_w13.untyped_storage(),
                source_w13.numel(),
            ),
            Component(
                "w2",
                source_w2.untyped_storage(),
                destination_w2.untyped_storage(),
                source_w2.numel(),
            ),
            Component(
                "w13_weight_scale",
                source_scale,
                destination_scale,
                source_scale.numel() * source_scale.element_size(),
                source_scale.dtype,
                tuple(source_scale.shape),
                source_scale.element_size(),
            ),
        ]
        chunks = chunk_components(components, args.chunk_bytes)
        peak_chunk = max(size for _, size in chunks)
        send_storage = torch.empty(
            peak_chunk, dtype=torch.uint8, device=device)
        recv_storage = torch.empty(
            peak_chunk, dtype=torch.uint8, device=device)

        for _ in range(args.iterations):
            for chunk_layout, chunk_bytes in chunks:
                send_buffer = send_storage[:chunk_bytes]
                recv_buffer = recv_storage[:chunk_bytes]
                assert_communication_tensor(send_buffer)
                assert_communication_tensor(recv_buffer)
                for component, offset in chunk_layout:
                    component_buffer_view(
                        send_buffer, offset, component).copy_(component.source)
                requests = dist.batch_isend_irecv([
                    dist.P2POp(dist.isend, send_buffer, send_peer),
                    dist.P2POp(dist.irecv, recv_buffer, recv_peer),
                ])
                for request in requests:
                    request.wait()
                for component, offset in chunk_layout:
                    component.destination.copy_(component_buffer_view(
                        recv_buffer, offset, component))
                torch.npu.synchronize()

        expected_w13 = (recv_peer * 31 + 11) % 251
        expected_w2 = (recv_peer * 31 + 19) % 251
        if not torch.all(destination_w13 == expected_w13).cpu().item():
            raise AssertionError("received w13 bytes do not match peer")
        if not torch.all(destination_w2 == expected_w2).cpu().item():
            raise AssertionError("received w2 bytes do not match peer")
        expected_scale = scale_pattern(recv_peer).contiguous()
        if not torch.equal(destination_scale.cpu(), expected_scale):
            raise AssertionError(
                "received non-contiguous MXFP4 scale does not match peer")

        print(
            f"[TORCH_PREFILL_P2P][rank={rank}][PASS] "
            f"send_peer={send_peer} recv_peer={recv_peer} "
            f"scale_shape={tuple(source_scale.shape)} "
            f"scale_stride={tuple(source_scale.stride())} "
            f"chunks={len(chunks)} peak_chunk_bytes={peak_chunk}",
            flush=True,
        )
        dist.barrier()
        if rank == 0:
            print("TORCH_PREFILL_P2P_RESULT PASS_ALL_RANKS", flush=True)
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
