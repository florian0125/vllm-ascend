#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate one shared CPU expert pool feeding arbitrary Ascend 950DT ranks.

This is an independent API smoke test, not a production-path test of
``ExpertOffloadManager``.  It validates four prerequisites for a Torch-only
shared Host-weight design on one machine:

1. Every rank maps the same file-backed CPU storage and observes peer writes.
2. Every rank can ``copy_`` an expert owned by another logical rank to its NPU.
3. All ranks can concurrently read one expert from the shared CPU pool.
4. A per-rank pinned staging buffer supports non-blocking Torch H2D.

The default payload models one DeepSeek-V4-Flash expert by raw byte count:
8 MiB w13, 4 MiB w2, and 768 KiB quantization metadata (12.75 MiB total).
Raw bytes are intentional: expert offload transfers packed/NZ storage bytes.

Run with one process per NPU, for example:

    torchrun --standalone --nproc-per-node=8 \
        script/validate_torch_shared_cpu_h2d_950dt.py \
        --shared-file /dev/shm/torch_shared_cpu_h2d.bin

The shared file is created exclusively and removed after a successful run.
An existing path is never overwritten.  Use ``--hold-seconds`` together with
``script/inspect_torch_shared_cpu_pss.sh`` to inspect physical sharing.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist


MIB = 1 << 20
KIB = 1 << 10


@dataclass(frozen=True)
class Segment:
    name: str
    start: int
    stop: int

    @property
    def nbytes(self) -> int:
        return self.stop - self.start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shared-file",
        default=os.getenv(
            "SHARED_FILE", "/dev/shm/vllm_ascend_torch_shared_cpu_h2d.bin"
        ),
        help="file backing the MAP_SHARED CPU pool; must not already exist",
    )
    parser.add_argument(
        "--backend",
        choices=("gloo", "hccl"),
        default=os.getenv("DIST_BACKEND", "gloo"),
        help="control-plane process-group backend",
    )
    parser.add_argument(
        "--experts-per-rank",
        type=int,
        default=4,
        help="logical experts assigned to each rank in the test pattern",
    )
    parser.add_argument("--w13-bytes", type=int, default=8 * MIB)
    parser.add_argument("--w2-bytes", type=int, default=4 * MIB)
    parser.add_argument("--quant-bytes", type=int, default=768 * KIB)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--hold-seconds",
        type=int,
        default=int(os.getenv("HOLD_SECONDS", "0")),
        help="keep mappings alive for /proc/PID/smaps inspection",
    )
    return parser.parse_args()


def rank_environment() -> tuple[int, int, int]:
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            "launch this script with torchrun; missing " + ", ".join(missing)
        )
    return tuple(int(os.environ[name]) for name in required)


def import_torch_npu():
    try:
        import torch_npu
    except ImportError as exc:
        raise RuntimeError(
            "torch_npu is required; source the CANN/TorchNPU environment first"
        ) from exc
    return torch_npu


def build_segments(args: argparse.Namespace) -> tuple[Segment, ...]:
    sizes = (
        ("w13", args.w13_bytes),
        ("w2", args.w2_bytes),
        ("quant", args.quant_bytes),
    )
    if args.w13_bytes <= 0 or args.w2_bytes <= 0 or args.quant_bytes < 0:
        raise ValueError("w13/w2 bytes must be positive and quant bytes non-negative")
    segments = []
    offset = 0
    for name, size in sizes:
        if size == 0:
            continue
        segments.append(Segment(name, offset, offset + size))
        offset += size
    return tuple(segments)


def pattern(expert_id: int, segment_id: int) -> int:
    return (expert_id * 17 + segment_id * 31 + 3) % 251


def fill_shared_pool(pool: torch.Tensor, segments: tuple[Segment, ...]) -> None:
    for expert_id in range(pool.shape[0]):
        for segment_id, segment in enumerate(segments):
            pool[expert_id, segment.start : segment.stop].fill_(
                pattern(expert_id, segment_id)
            )


def validate_expert(
    expert_id: int,
    outputs: dict[str, torch.Tensor],
    segments: tuple[Segment, ...],
) -> None:
    for segment_id, segment in enumerate(segments):
        actual = outputs[segment.name].cpu()
        expected = pattern(expert_id, segment_id)
        if expert_id == 0 and segment.start == 0:
            if int(actual[0]) != 233 or not torch.all(actual[1:] == expected):
                raise AssertionError(
                    f"expert={expert_id} segment={segment.name} mismatch"
                )
        elif not torch.all(actual == expected):
            mismatch = torch.nonzero(actual != expected, as_tuple=False)
            first = int(mismatch[0].item()) if mismatch.numel() else -1
            raise AssertionError(
                f"expert={expert_id} segment={segment.name} "
                f"first_mismatch={first}"
            )


def copy_direct(
    pool: torch.Tensor,
    expert_id: int,
    destinations: dict[str, torch.Tensor],
    segments: tuple[Segment, ...],
) -> None:
    """Blocking Torch copies directly from pageable MAP_SHARED storage."""
    for segment in segments:
        source = pool[expert_id, segment.start : segment.stop]
        destinations[segment.name].copy_(source, non_blocking=False)
    torch.npu.synchronize()


def copy_via_pinned_staging(
    pool: torch.Tensor,
    expert_id: int,
    staging: dict[str, torch.Tensor],
    destinations: dict[str, torch.Tensor],
    segments: tuple[Segment, ...],
) -> None:
    """CPU memcpy into local pinned buffers followed by non-blocking H2D."""
    for segment in segments:
        source = pool[expert_id, segment.start : segment.stop]
        staging[segment.name].copy_(source)
    for segment in segments:
        destinations[segment.name].copy_(
            staging[segment.name], non_blocking=True
        )
    torch.npu.synchronize()


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def benchmark(operation, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        operation()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - start) * 1e3)
    return {
        "min_ms": min(samples),
        "p50_ms": statistics.median(samples),
        "p99_ms": percentile(samples, 0.99),
        "mean_ms": statistics.mean(samples),
    }


def record_failure(
    failures: list[str], rank: int, stage: str, error: BaseException
) -> None:
    message = f"{stage}: {type(error).__name__}: {error}"
    failures.append(message)
    print(f"[TORCH_SHARED_CPU][rank={rank}][FAIL] {message}", flush=True)


def main() -> int:
    args = parse_args()
    if args.experts_per_rank <= 0:
        raise ValueError("experts-per-rank must be positive")
    if args.warmup < 0 or args.iterations <= 0 or args.hold_seconds < 0:
        raise ValueError("warmup/hold must be non-negative and iterations positive")

    rank, local_rank, world_size = rank_environment()
    torch_npu = import_torch_npu()
    segments = build_segments(args)
    expert_bytes = segments[-1].stop
    num_experts = world_size * args.experts_per_rank
    total_bytes = num_experts * expert_bytes
    shared_path = Path(args.shared_file).resolve()
    if not shared_path.parent.is_dir():
        raise FileNotFoundError(f"shared-file parent does not exist: {shared_path.parent}")

    torch.npu.set_device(local_rank)
    dist.init_process_group(backend=args.backend)
    device = torch.device(f"npu:{local_rank}")
    failures: list[str] = []
    metrics: dict[str, dict[str, float]] = {}
    created_file = False

    try:
        # Reserve the exact path exclusively so a stale or user-owned file is
        # never truncated. torch.from_file grows this zero-length file.
        if rank == 0:
            fd = os.open(
                shared_path,
                os.O_CREAT | os.O_EXCL | os.O_RDWR,
                0o600,
            )
            try:
                os.ftruncate(fd, total_bytes)
            finally:
                os.close(fd)
            created_file = True
            flat = torch.from_file(
                str(shared_path),
                shared=True,
                size=total_bytes,
                dtype=torch.uint8,
            )
            pool = flat.view(num_experts, expert_bytes)
            fill_shared_pool(pool, segments)

        dist.barrier()
        if rank != 0:
            flat = torch.from_file(
                str(shared_path),
                shared=True,
                size=total_bytes,
                dtype=torch.uint8,
            )
            pool = flat.view(num_experts, expert_bytes)
        dist.barrier()

        # A peer-visible mutation proves all ranks have MAP_SHARED views. The
        # PSS helper is still required to audit physical accounting.
        if rank == 0:
            pool[0, 0] = 233
        dist.barrier()
        if int(pool[0, 0]) != 233:
            raise AssertionError("rank-0 shared-memory mutation is not visible")

        # Fault every page into every process so mapping PSS can be compared.
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_checksum = int(pool.view(-1)[::page_size].sum().item())

        destinations = {
            segment.name: torch.empty(
                segment.nbytes, dtype=torch.uint8, device=device
            )
            for segment in segments
        }
        remote_rank = (rank + 1) % world_size
        remote_expert = remote_rank * args.experts_per_rank

        dist.barrier()
        try:
            copy_direct(pool, remote_expert, destinations, segments)
            validate_expert(remote_expert, destinations, segments)
            print(
                f"[TORCH_SHARED_CPU][rank={rank}][PASS] "
                f"direct_remote expert={remote_expert}",
                flush=True,
            )
        except BaseException as error:
            record_failure(failures, rank, "direct_remote", error)
        dist.barrier()

        same_expert = num_experts - 1
        try:
            copy_direct(pool, same_expert, destinations, segments)
            validate_expert(same_expert, destinations, segments)
            print(
                f"[TORCH_SHARED_CPU][rank={rank}][PASS] "
                f"concurrent_same_expert expert={same_expert}",
                flush=True,
            )
        except BaseException as error:
            record_failure(failures, rank, "concurrent_same_expert", error)
        dist.barrier()

        staging: dict[str, torch.Tensor] = {}
        try:
            staging = {
                segment.name: torch.empty(
                    segment.nbytes,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=True,
                )
                for segment in segments
            }
            if not all(tensor.is_pinned() for tensor in staging.values()):
                raise AssertionError("one or more staging tensors are not pinned")
            copy_via_pinned_staging(
                pool, remote_expert, staging, destinations, segments
            )
            validate_expert(remote_expert, destinations, segments)
            print(
                f"[TORCH_SHARED_CPU][rank={rank}][PASS] pinned_staging",
                flush=True,
            )
        except BaseException as error:
            record_failure(failures, rank, "pinned_staging", error)
        dist.barrier()

        if not failures:
            metrics["direct_pageable"] = benchmark(
                lambda: copy_direct(
                    pool, remote_expert, destinations, segments
                ),
                args.warmup,
                args.iterations,
            )
            metrics["pinned_staging_end_to_end"] = benchmark(
                lambda: copy_via_pinned_staging(
                    pool,
                    remote_expert,
                    staging,
                    destinations,
                    segments,
                ),
                args.warmup,
                args.iterations,
            )

        summary = {
            "rank": rank,
            "world_size": world_size,
            "device": local_rank,
            "shared_file": str(shared_path),
            "shared_inode": shared_path.stat().st_ino,
            "pool_is_shared": pool.is_shared(),
            "pool_is_pinned": pool.is_pinned(),
            "page_checksum": page_checksum,
            "num_experts": num_experts,
            "expert_bytes": expert_bytes,
            "pool_bytes": total_bytes,
            "remote_expert": remote_expert,
            "segments": {segment.name: segment.nbytes for segment in segments},
            "metrics": metrics,
            "failures": failures,
            "torch": torch.__version__,
            "torch_npu": getattr(torch_npu, "__version__", "unknown"),
        }
        print("TORCH_SHARED_CPU_RESULT " + json.dumps(summary, sort_keys=True), flush=True)

        failure_tensor_device = device if args.backend == "hccl" else "cpu"
        global_failures = torch.tensor(
            [len(failures)], dtype=torch.int32, device=failure_tensor_device
        )
        dist.all_reduce(global_failures, op=dist.ReduceOp.SUM)
        global_failure_count = int(global_failures.cpu().item())

        dist.barrier()
        if args.hold_seconds:
            print(
                f"[TORCH_SHARED_CPU][rank={rank}][HOLD] "
                f"pid={os.getpid()} seconds={args.hold_seconds}",
                flush=True,
            )
            time.sleep(args.hold_seconds)
        dist.barrier()

        if rank == 0 and created_file:
            shared_path.unlink()
            created_file = False
        dist.barrier()

        if global_failure_count:
            raise RuntimeError(
                f"shared CPU H2D validation failed: "
                f"global_failure_count={global_failure_count}"
            )
        if rank == 0:
            print(
                "[TORCH_SHARED_CPU][PASS] all ranks completed; "
                "this remains an independent API smoke test",
                flush=True,
            )
    finally:
        # Only remove a file that this process created. An unexpected rank
        # failure may leave the test file behind for diagnosis; the README
        # gives a narrow, explicit cleanup command.
        if rank == 0 and created_file and shared_path.exists():
            try:
                shared_path.unlink()
            except OSError:
                pass
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
