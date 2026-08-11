#!/usr/bin/env python3
"""Two-rank ElasticBuffer/Engram proof-of-concept for expert weights.

This script validates the parts of CANN ElasticBuffer that are relevant to a
distributed CPU expert-weight store:

1. Every rank writes a different CPU shard into its local Engram storage.
2. Every NPU fetches both local and remote entries by global expert id.
3. The fetched tensor is copied into a preallocated NPU expert slot and checked
   without tolerance (Engram does not currently expose an ``out=`` argument).
4. Packed W4 bytes are transported through an FP16 carrier and compared byte
   for byte, because Engram accepts only BF16, FP16, and FP32 storage tensors.
5. Optionally, the same fetch is captured and replayed by an ACL Graph with a
   different set of global expert ids.

ElasticBuffer is documented as supporting Ascend 950PR/950DT only. Run this on
an environment with matching CANN, torch_npu, and cann_ops_transformer builds.

Examples:

    torchrun --standalone --nproc-per-node=2 \
        script/test_elastic_buffer_engram.py

    torchrun --standalone --nproc-per-node=2 \
        script/test_elastic_buffer_engram.py --graph required

The test intentionally uses torchrun instead of creating subprocesses itself,
so rank failures remain visible and torchrun can terminate the other ranks.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist


ENGRAM_HIDDEN_ALIGNMENT = 128
PACKED_CARRIER_DTYPE = torch.float16


@dataclass(frozen=True)
class TestCase:
    name: str
    hidden: int
    dtype: torch.dtype
    make_storage: Callable[[int, int, int], torch.Tensor]
    make_expected: Callable[[torch.Tensor, int, int], torch.Tensor]
    compare_as_bytes: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate distributed CPU expert fetch with ElasticBuffer/Engram."
    )
    parser.add_argument("--num-entries", type=int, default=8, help="CPU entries per rank")
    parser.add_argument(
        "--num-indices", type=int, default=8, help="entries fetched by every rank"
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=128,
        help="numeric test hidden size; must be divisible by 128",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="numeric test storage dtype",
    )
    parser.add_argument(
        "--packed-row-bytes",
        type=int,
        default=4096,
        help="W4-like byte payload per expert; rounded up to 256-byte alignment",
    )
    parser.add_argument(
        "--cases",
        choices=("all", "numeric", "packed-w4"),
        default="all",
        help="test case selection",
    )
    parser.add_argument(
        "--graph",
        choices=("off", "optional", "required"),
        default="off",
        help="ACL Graph test policy; optional reports failure without masking eager success",
    )
    parser.add_argument(
        "--device-offset",
        type=int,
        default=0,
        help="physical NPU id added to LOCAL_RANK",
    )
    return parser.parse_args()


def log(rank: int, message: str) -> None:
    print(f"[ENGRAM_POC][rank={rank}] {message}", flush=True)


def read_process_memory_kib() -> dict[str, int]:
    """Return Linux RSS/PSS breakdown without adding a dependency."""
    smaps_rollup = Path("/proc/self/smaps_rollup")
    if not smaps_rollup.exists():
        return {}

    wanted = {"Rss", "Pss", "Private_Clean", "Private_Dirty", "Shared_Clean", "Shared_Dirty"}
    result: dict[str, int] = {}
    for line in smaps_rollup.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if separator and key in wanted:
            result[key] = int(value.strip().split()[0])
    return result


def log_process_memory(rank: int, stage: str) -> None:
    values = read_process_memory_kib()
    if not values:
        log(rank, f"[CPU_MEM] stage={stage} unavailable (requires /proc/self/smaps_rollup)")
        return
    details = " ".join(f"{key.lower()}_mib={value / 1024:.2f}" for key, value in values.items())
    log(rank, f"[CPU_MEM] stage={stage} {details}")


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def make_numeric_storage(rank: int, num_entries: int, hidden: int, dtype: torch.dtype) -> torch.Tensor:
    entry = torch.arange(num_entries, dtype=torch.int64).unsqueeze(1)
    column = torch.arange(hidden, dtype=torch.int64).unsqueeze(0)
    values = (rank * 29 + entry * 7 + column % 19) % 97
    return values.to(dtype=dtype).contiguous()


def make_numeric_expected(
    global_indices: torch.Tensor,
    num_entries: int,
    hidden: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    indices = global_indices.to(dtype=torch.int64, device="cpu")
    owner = torch.div(indices, num_entries, rounding_mode="floor").unsqueeze(1)
    local_entry = torch.remainder(indices, num_entries).unsqueeze(1)
    column = torch.arange(hidden, dtype=torch.int64).unsqueeze(0)
    values = (owner * 29 + local_entry * 7 + column % 19) % 97
    return values.to(dtype=dtype).contiguous()


def make_packed_storage(rank: int, num_entries: int, hidden: int) -> torch.Tensor:
    """Create deterministic bytes, then expose them as an FP16 Engram table."""
    row_bytes = hidden * PACKED_CARRIER_DTYPE.itemsize
    entry = torch.arange(num_entries, dtype=torch.int64).unsqueeze(1)
    byte = torch.arange(row_bytes, dtype=torch.int64).unsqueeze(0)
    packed = ((rank * 37 + entry * 13 + byte) % 251).to(torch.uint8).contiguous()
    return packed.view(PACKED_CARRIER_DTYPE)


def make_packed_expected(
    global_indices: torch.Tensor, num_entries: int, hidden: int
) -> torch.Tensor:
    indices = global_indices.to(dtype=torch.int64, device="cpu")
    row_bytes = hidden * PACKED_CARRIER_DTYPE.itemsize
    owner = torch.div(indices, num_entries, rounding_mode="floor").unsqueeze(1)
    local_entry = torch.remainder(indices, num_entries).unsqueeze(1)
    byte = torch.arange(row_bytes, dtype=torch.int64).unsqueeze(0)
    return ((owner * 37 + local_entry * 13 + byte) % 251).to(torch.uint8).contiguous()


def build_indices(rank: int, world_size: int, num_entries: int, count: int) -> torch.Tensor:
    """Include every owner rank, then deterministically fill the requested count."""
    values = [owner * num_entries + ((rank + owner) % num_entries) for owner in range(world_size)]
    total_entries = world_size * num_entries
    for offset in range(max(0, count - len(values))):
        values.append((rank * 11 + offset * 17 + 3) % total_entries)
    return torch.tensor(values[:count], dtype=torch.int32)


def validate_exact(
    rank: int,
    stage: str,
    actual_npu: torch.Tensor,
    expected_cpu: torch.Tensor,
    compare_as_bytes: bool,
) -> None:
    torch.npu.synchronize()
    actual_cpu = actual_npu.cpu().contiguous()
    if compare_as_bytes:
        actual_cpu = actual_cpu.view(torch.uint8)
    if not torch.equal(actual_cpu, expected_cpu):
        mismatch = torch.nonzero(actual_cpu != expected_cpu, as_tuple=False)
        first = mismatch[0].tolist() if mismatch.numel() else []
        raise AssertionError(
            f"{stage} mismatch: first_index={first}, "
            f"actual_shape={tuple(actual_cpu.shape)}, expected_shape={tuple(expected_cpu.shape)}"
        )
    log(rank, f"[PASS] {stage} shape={tuple(actual_npu.shape)} dtype={actual_npu.dtype}")


def validate_preallocated_slot(
    rank: int,
    fetched: torch.Tensor,
    expected_cpu: torch.Tensor,
    compare_as_bytes: bool,
) -> None:
    slot = torch.empty_like(fetched)
    slot_ptr = slot.data_ptr()
    slot.copy_(fetched, non_blocking=True)
    torch.npu.synchronize()
    if slot.data_ptr() != slot_ptr:
        raise AssertionError("preallocated expert slot address changed")
    validate_exact(rank, "preallocated_slot_copy", slot, expected_cpu, compare_as_bytes)
    log(
        rank,
        "[SLOT] direct_out_supported=false extra_npu_copy=true "
        f"fetched_ptr={fetched.data_ptr()} slot_ptr={slot_ptr}",
    )


def run_graph_test(
    rank: int,
    buffer: object,
    group: dist.ProcessGroup,
    initial_indices_cpu: torch.Tensor,
    num_entries: int,
    expected_builder: Callable[[torch.Tensor], torch.Tensor],
    compare_as_bytes: bool,
) -> None:
    static_indices = initial_indices_cpu.npu()
    replay_indices_cpu = torch.roll(initial_indices_cpu, shifts=1)
    total_entries = dist.get_world_size(group) * num_entries
    replay_indices_cpu = torch.remainder(
        replay_indices_cpu + num_entries, total_entries
    ).to(torch.int32)
    replay_indices_npu = replay_indices_cpu.npu()

    # Warm up lazy Engram resources before capture.
    warmup = buffer.engram_fetch(static_indices)()
    torch.npu.synchronize()
    del warmup
    dist.barrier(group=group)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_output = buffer.engram_fetch(static_indices)()
    torch.npu.synchronize()
    validate_exact(
        rank,
        "acl_graph_capture_output",
        graph_output,
        expected_builder(initial_indices_cpu),
        compare_as_bytes,
    )

    static_indices.copy_(replay_indices_npu)
    dist.barrier(group=group)
    graph.replay()
    torch.npu.synchronize()
    validate_exact(
        rank,
        "acl_graph_replay_changed_indices",
        graph_output,
        expected_builder(replay_indices_cpu),
        compare_as_bytes,
    )


def reduce_pass(group: dist.ProcessGroup, passed: bool) -> bool:
    flag = torch.tensor([int(passed)], dtype=torch.int32, device="npu")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.cpu().item())


def run_case(
    rank: int,
    world_size: int,
    group: dist.ProcessGroup,
    elastic_buffer_class: type,
    test_case: TestCase,
    num_entries: int,
    num_indices: int,
    graph_policy: str,
) -> tuple[bool, bool | None]:
    log(rank, f"[CASE] start name={test_case.name} hidden={test_case.hidden} dtype={test_case.dtype}")
    log_process_memory(rank, f"{test_case.name}:before_storage")
    storage = test_case.make_storage(rank, num_entries, test_case.hidden)
    log_process_memory(rank, f"{test_case.name}:after_storage")

    num_cpu_bytes = elastic_buffer_class.get_engram_storage_size_hint(
        num_entries, test_case.hidden, test_case.dtype
    )
    payload_bytes = storage.numel() * storage.element_size()
    log(
        rank,
        f"[CPU_BUFFER] local_payload_bytes={payload_bytes} local_pinned_bytes={num_cpu_bytes} "
        f"aggregate_payload_bytes={payload_bytes * world_size} "
        f"aggregate_pinned_bytes={num_cpu_bytes * world_size} world_size={world_size}",
    )

    buffer = elastic_buffer_class(group, num_cpu_bytes=num_cpu_bytes)
    graph_passed: bool | None = None
    try:
        buffer.engram_write(storage)
        log_process_memory(rank, f"{test_case.name}:after_engram_write")
        del storage
        gc.collect()
        log_process_memory(rank, f"{test_case.name}:after_source_delete")

        indices_cpu = build_indices(rank, world_size, num_entries, num_indices)
        owner_ranks = torch.unique(indices_cpu // num_entries).tolist()
        log(
            rank,
            f"[FETCH] global_indices={indices_cpu.tolist()} owner_ranks={owner_ranks} "
            f"contains_remote={any(owner != rank for owner in owner_ranks)}",
        )
        expected_builder = lambda indices: test_case.make_expected(
            indices, num_entries, test_case.hidden
        )
        expected_cpu = expected_builder(indices_cpu)
        fetched = buffer.engram_fetch(indices_cpu.npu())()
        validate_exact(rank, "eager_local_and_remote_fetch", fetched, expected_cpu, test_case.compare_as_bytes)
        validate_preallocated_slot(rank, fetched, expected_cpu, test_case.compare_as_bytes)

        if graph_policy != "off":
            try:
                run_graph_test(
                    rank,
                    buffer,
                    group,
                    indices_cpu,
                    num_entries,
                    expected_builder,
                    test_case.compare_as_bytes,
                )
                graph_passed = True
                log(rank, "[GRAPH][PASS] capture and replay accepted changed expert ids")
            except Exception as error:  # noqa: BLE001 - this is a hardware capability probe
                graph_passed = False
                log(rank, f"[GRAPH][FAIL] {type(error).__name__}: {error}")
                if graph_policy == "required":
                    raise
        return True, graph_passed
    finally:
        buffer.destroy()
        log_process_memory(rank, f"{test_case.name}:after_destroy")


def load_runtime() -> tuple[object, type]:
    try:
        import torch_npu
    except ImportError as error:
        raise RuntimeError("torch_npu is not installed in this Python environment") from error

    try:
        from cann_ops_transformer import ElasticBuffer
    except ImportError as error:
        raise RuntimeError(
            "cann_ops_transformer is not installed. Install the build matching the active CANN release."
        ) from error
    return torch_npu, ElasticBuffer


def main() -> int:
    args = parse_args()
    if args.num_entries <= 0:
        raise ValueError("--num-entries must be positive for this remote-fetch test")
    if args.num_indices <= 0:
        raise ValueError("--num-indices must be positive")
    if args.hidden <= 0 or args.hidden % ENGRAM_HIDDEN_ALIGNMENT != 0:
        raise ValueError("--hidden must be positive and divisible by 128")
    if args.packed_row_bytes <= 0:
        raise ValueError("--packed-row-bytes must be positive")

    torch_npu, elastic_buffer_class = load_runtime()
    launcher_rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(launcher_rank)))
    physical_device = args.device_offset + local_rank
    torch_npu.npu.set_device(physical_device)
    dist.init_process_group(backend="hccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size < 2:
        raise RuntimeError("Engram requires world_size >= 2; launch with torchrun --nproc-per-node=2")
    if args.num_indices < world_size:
        raise ValueError("--num-indices must be >= world_size so every owner rank is fetched")

    group = dist.new_group(ranks=list(range(world_size)), backend="hccl")
    device_name_fn = getattr(torch_npu.npu, "get_device_name", None)
    device_name = device_name_fn(physical_device) if callable(device_name_fn) else "unknown"
    log(
        rank,
        f"[ENV] pid={os.getpid()} rank={rank}/{world_size} local_rank={local_rank} "
        f"physical_device={physical_device} device_name={device_name} "
        f"torch={torch.__version__} torch_npu={getattr(torch_npu, '__version__', 'unknown')}",
    )
    log(rank, "[SUPPORT] official ElasticBuffer documentation currently lists Ascend 950PR/950DT only")

    dtype = getattr(torch, args.dtype)
    packed_row_bytes = align_up(
        args.packed_row_bytes,
        ENGRAM_HIDDEN_ALIGNMENT * PACKED_CARRIER_DTYPE.itemsize,
    )
    cases: list[TestCase] = []
    if args.cases in ("all", "numeric"):
        cases.append(
            TestCase(
                name="numeric",
                hidden=args.hidden,
                dtype=dtype,
                make_storage=lambda rank_id, entries, hidden: make_numeric_storage(
                    rank_id, entries, hidden, dtype
                ),
                make_expected=lambda indices, entries, hidden: make_numeric_expected(
                    indices, entries, hidden, dtype
                ),
            )
        )
    if args.cases in ("all", "packed-w4"):
        cases.append(
            TestCase(
                name="packed_w4_bytes_via_fp16",
                hidden=packed_row_bytes // PACKED_CARRIER_DTYPE.itemsize,
                dtype=PACKED_CARRIER_DTYPE,
                make_storage=make_packed_storage,
                make_expected=make_packed_expected,
                compare_as_bytes=True,
            )
        )

    all_eager_passed = True
    all_graph_passed: bool | None = None
    for test_case in cases:
        eager_passed, graph_passed = run_case(
            rank,
            world_size,
            group,
            elastic_buffer_class,
            test_case,
            args.num_entries,
            args.num_indices,
            args.graph,
        )
        all_eager_passed = all_eager_passed and eager_passed
        if graph_passed is not None:
            all_graph_passed = (
                graph_passed if all_graph_passed is None else all_graph_passed and graph_passed
            )

    global_eager_passed = reduce_pass(group, all_eager_passed)
    global_graph_passed = None
    if all_graph_passed is not None:
        global_graph_passed = reduce_pass(group, all_graph_passed)

    if rank == 0:
        print(
            "[ENGRAM_POC][SUMMARY] "
            f"eager_passed={global_eager_passed} graph_policy={args.graph} "
            f"graph_passed={global_graph_passed} cases={[case.name for case in cases]}",
            flush=True,
        )
    return 0 if global_eager_passed and (args.graph != "required" or global_graph_passed) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001 - preserve rank-local diagnostics before torchrun exits
        traceback.print_exc()
        sys.exit(1)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
