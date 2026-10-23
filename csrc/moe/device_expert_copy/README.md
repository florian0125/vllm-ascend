# Fused device-owned native expert transfer (A5 prototype)

The upload is one `FusedNativeExpertCopyOps` kernel:

```
selected expert + selected slot + active flag
                  |
mapped native DRAM weights/scales
                  |
32 AIV cores: address calculation + byte-layout conversion + final slot writes
                  |
             same-stream GEMM
```

There is no full-expert HBM staging, separate NZ conversion, descriptor arithmetic,
or second HBM-to-HBM copy. With `active=false`, the kernel returns without copying.
Placement/statistics still use device tensor operations and are not fused here.
A derived fp32 scale refresh, when present on the layer, remains separate.

The A5 `FRACTAL_NZ_C0_16` byte mapping was checked against CANN format conversion.
For byte matrix `[N,K]`, its physical order is `[K/16,N,16]`. Gate/up are written
into their respective row ranges of w13; down goes into w2. Scale strides select
row-major or column-pair layout. Unsupported widths, padding and scale strides
are rejected; other chip layouts are not claimed to work.

The tiled implementation fetches 16 complete native rows contiguously from
mapped DRAM, gathers their byte layout inside UB, and writes column chunks into
the final slot. Weights use 32-bit bit-preserving gather; column-pair scales use
16-bit gather; row-major scales use a direct queue-bound copy. Jobs from all six
fields are distributed round-robin across 32 cores. No full-expert HBM staging
or intermediate HBM conversion is introduced. Gather offsets are built once at
layer initialization and retained on device.

The library exports `AccOffloadNativeCopyPlanVersion() == 2` for the tiled recipe.
Python retains recipe-v1 compatibility with older libraries. The tiled recipe
currently requires source row counts divisible by 16 and row sizes <= 2048 bytes.

The sparse-copy code derives from MemFabric Hybrid under Mulan PSL v2. Its
torch-npu dispatcher preserves operation-queue order and ACL graph capture.
Existing CPU weight buffers are registered once with ACL; no MemFabric pool or
additional CPU weight copy is required.

## Build

Source the container's CANN environment. Use a new build directory; do not replace
a library loaded by a live service.

```bash
bash csrc/moe/device_expert_copy/build.sh /absolute/build/directory
```

Set `expert_offload_config.cpu_moe`:

```json
{
  "transfer_one_expert": true,
  "transfer_scheduler": "npu",
  "transfer_copy_library": "/absolute/build/directory/lib/libnpu_expert_copy_dispatch.so",
  "weight_format": "native_mxfp4",
  "transfer_policy": "first_miss"
}
```

The library must export `AccOffloadNativeCopy`; rebuild older libraries. CPU
scheduler remains the compatibility default. Supports single-rank replicated
native experts and at least one HBM slot. Keep prefetch disabled as in the native
graph baseline. `frequency` policy is also supported.

## Validation (run only when NPU resources are available)

The fixture is specific to DeepSeek-V4 layer 16, 4096 hidden, 2048 intermediate.
It compares actual storage bytes with the previous CANN-based converter and
checks eager, graph replay, slot 1, protected resident routes and no-upload cases.

```bash
python csrc/moe/device_expert_copy/tests/check_native_transfer.py \
  --library /absolute/build/directory/lib/libnpu_expert_copy_dispatch.so \
  --checkpoint /mnt/share/DeepSeek-V4-Flash-0731 --device 0
```

Optional `--benchmark` enables the microbenchmark.
Use `--scale-layout column_pair` to exercise column-pair scale storage.
Real-weight parity checks passed on x86 Ascend950PR for eager replacement,
dynamic graph replay, slot 1, protected routes, no-op replay and batch 2, with
both supported scale layouts. Isolated event timings for a 12.75 MiB expert were
about 0.58--0.64 ms for the tiled copy+layout kernel versus 4.32 ms for the original
strided kernel; contiguous copy without layout conversion measured 0.57--0.59 ms.
These measurements cover the complete fused operator, excluding admission,
CPU expert compute, GEMM and join. They are not a model-throughput speedup claim.
