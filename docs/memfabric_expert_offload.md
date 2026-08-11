# MemFabric single-card expert H2D

The second-stage MemFabric integration replaces the expert-offload Host memory
allocator and H2D copy implementation while leaving routing, cache eviction,
expert placement, and NPU weight layout unchanged. It currently supports only
single-card expert offload.

## Configuration

Install MemFabric Hybrid release 1.2 in the serving environment and source its
runtime environment before starting vLLM. Select the backend through
`additional_config.expert_offload_config`:

```json
{
  "expert_offload_config": {
    "expert_offload": true,
    "enable_multi_card": false,
    "h2d_backend": "memfabric",
    "memfabric_pool_size_gib": 512,
    "memfabric_log_level": 3
  }
}
```

`memfabric_pool_size_gib` is the LOCAL DRAM pool reserved by each serving
process. Size it to hold that process's CPU expert weights and quantization
metadata, plus startup format-conversion intermediates. The default backend is
`torch`, so existing configurations continue to use pinned CPU tensors and
`copy_()`.

## Data path

- Expert weights and quantization metadata are allocated with
  `memfabric_hybrid.offload.empty`.
- Decode misses, expert prefetch, hot-expert preload, and single-card prefill
  are submitted through `memfabric_hybrid.offload.sparse_copy`.
- Copies are launched on the manager's current NPU load stream and synchronized
  at the existing stream boundaries. No device-wide synchronization is added.
- MemFabric is uninitialized when the transport is closed or the process exits.

The integration pads odd sparse-copy descriptor batches with a zero-byte entry
because the release 1.2 kernel processes descriptors as two equal halves.

## Current boundary

Setting both `h2d_backend` to `memfabric` and `enable_multi_card` to `true` is
rejected during configuration validation. Shared-DRAM and multi-card transport
are reserved for the next stage.
