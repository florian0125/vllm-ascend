# CPU MoE for DSpark verification

The CPU kernel and stream/host-callback scheduling are copied from
`single_card_cpu_moe_0917`. `cpu_backend/kt_kernel_backend.py` is byte-for-byte
identical to that branch. The adapter changes route ownership and execution
eligibility; it does not introduce a new CPU kernel or scheduling protocol.

## Routing contract

After anchor-union selection and optional threshold pruning, the hybrid executor
uses the current `log2phy` map. Retained resident experts execute on NPU;
retained nonresident experts execute on CPU. Negative/out-of-range IDs and
zero-weight routes never execute. Pruning does not renormalize the retained
weights; the CPU path preserves the selector's weighting/scaling semantics.

`submit(layer_idx, hidden_states, topk_ids, topk_weights)` returns a task.
CPU IDs are global logical expert IDs; `-1` marks skipped routes. `wait()`
returns `[num_tokens, hidden_size]` in input token order, with each routing
weight applied once. The RoutedExperts boundary merges the CPU output after NPU
token combine and before communication finalize. Shared experts keep their
existing execution path.

Threshold pruning previously lived in the paging hook. Hybrid execution calls
the same pruning helper before CPU/NPU splitting, since it bypasses paging.
The core CPU backend is independent of either pruning strategy.

## Enablement and capacity

An attached CPU executor handles batches whose actual input rows, including
ACL Graph padding, fit `cpu_moe.max_num_tokens`. Larger batches retain the
existing NPU paging/layerwise path, independent of the request phase. Startup
memory profiling (`in_profile_run=True`) also retains the existing NPU path.

Reuse `expert_offload_config.cpu_moe` from the CPU backup, including `enabled`,
`backend`, `weight_path`, `layers`, CPU threads, NUMA settings and
`max_num_tokens`. Layer indices are MODEL indices, matching GGUF `blk.N`, not
MoE registration ordinals. All selected layers need matching GGUF weights.
The default remains disabled; only single-card TP=EP=1 is supported.

`max_num_tokens` sets both preallocated CPU buffer capacity and hybrid admission
threshold. The graph launch script uses 100. Batches at or below capacity use
CPU/NPU splitting; larger batches return to the original NPU path before CPU
submission, pruning or promotion. No scheduler-wide capacity minimum is imposed.

The existing `transfer_one_expert` option is preserved, defaults to false, and
uses the backup's transfer/submit/deferred-join protocol. Pruned/zero-weight
routes are excluded before any promotion decision.

## Validation

Focused tests:

```bash
pytest -q tests/ut/expert_offload/test_hybrid_executor.py \
  tests/ut/expert_offload/test_kt_kernel_backend.py \
  tests/ut/expert_offload/test_hybrid_transfer.py \
  tests/ut/expert_offload/test_cpu_dspark.py \
  tests/ut/expert_offload/test_cpu_moe_wiring.py
```

They cover final routes from both policies, production threshold pruning,
duplicate routes, invalid IDs, zero weights, CPU/NPU ownership, token order,
comparison to gated expert MLPs, enabled/disabled behavior independent of phase, configured capacity,
padded-call fallback and existing graph callback ordering.

Hardware scheduling regression, **on an idle NPU**:

```bash
pytest -q tests/e2e/pull_request/one_card/test_cpu_moe_dspark.py
```

This reuses the backup's synthetic CPU worker and tests 1/6/10-token batches,
one/two layers, eager/graph, both transfer settings and 24 changing-route
replays. It validates scheduling and ownership, not real kt-kernel numerics or
model performance. Those require a compatible vLLM/Triton environment and
fresh DSpark model runs under each pruning policy.


## Main 0928 migration

Base: `moe_offload_main_0928`, commit `643ee9f`.
CPU backend attachment and output merge now live in
`ops/fused_moe/routed_experts.py`. Route splitting before paging lives in
`expert_offload/routed_experts.py`, shared by quantization methods including
V4 MXFP4. The CPU kernel and stream/host callbacks remain unchanged.
The branch already contains activation routing and threshold pruning.
The existing V4 DSpark main projection quantization override is preserved in
`models/deepseek_v4/dspark.py`.
