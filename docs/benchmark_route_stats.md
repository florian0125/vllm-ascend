# Per-benchmark expert and speculative decoding statistics

Single-card expert offload, V1 model runner. Tested with synchronous scheduling
and full-decode ACL graphs. The same counter path covers CPU hybrid and ordinary
offload. This first implementation requires `--no-async-scheduling` and
`expert_prefetch_enabled=false`; unsupported modes are rejected, not reported as
exact statistics. These match the current run_dpark.sh configuration.

Enable in top-level `--additional-config` and restart the service once:

```json
{
  "decode_stats_enabled": true,
  "benchmark_route_stats_enabled": true,
  "decode_stats_path": "/absolute/path/to/this-service-stats"
}
```

Startup prints the cumulative snapshot path. It contains a session ID, sequence,
per-layer counters and aggregate counters excluding hash-routed layers. Capture
and startup warmups are gated off. Runtime actual token rows mask graph padding.
Counters are fixed-shape int64 NPU tensors and run on each graph replay.
One async pinned-host copy is queued after each real forward; a host writer
atomically publishes JSON at most four times per second and flushes on idle.
There is no per-layer host synchronization or per-expert log printing.
This instrumentation adds reductions and a per-forward copy: compare performance
with instrumentation consistently enabled, and measure its overhead separately.

Inside the same container, wrap each normal benchmark command:

```bash
python /mnt/data/y50063808/dspark/vllm-ascend/tools/aisbench_with_stats.py \
  --counter-file /absolute/path/to/this-service-stats/benchmark_counters_rank0.json \
  --out /absolute/path/to/new-bs1-stats \
  -- ais_bench --work-dir /absolute/path/to/outputs --models 8150 \
     --datasets gsm8k_gen_0_shot_cot_str_perf --summarizer default_perf \
     --mode perf --num-prompts 20 --num-warmups 0
```

Keep other clients idle and warm up before the wrapper. Warmups included in the
wrapped command are included in counters. Set BS/concurrency in the normal
ais_bench model config; this wrapper does not change it. Every output directory
must be fresh. Service restarts, decreased counters and stale snapshots are
rejected. Files: `before.json`, `after.json`, `summary.json`, metrics snapshots.
The wrapper saves failed benchmark results with their exit code as well.

Metrics in `summary.json` are ratios of summed counter deltas, not means of
per-layer percentages:

- `pre_route_hit_rate`: resident routes / valid routes BEFORE substitution and
  migration. `pre_expert_hit_rate` deduplicates experts within each layer-call.
- `post_*_hit_rate`: after substitution/pruning, against the original residency.
- `npu_execution_share`: final NPU routes / valid post-substitution routes. This
  includes newly admitted experts and is NOT the original cache hit rate.
- `admitted_experts`: previously absent experts now resident in the persistent
  decode cache. This is NOT a byte counter or prefill-pool transfer counter.
- `draft_token_acceptance_rate`: accepted draft tokens / proposed draft tokens.
- `mean_acceptance_length`: 1 + accepted draft tokens / draft rounds. The bonus
  token is excluded from the acceptance-rate numerator.

All rates are null if their denominator is zero. Direct `/metrics` snapshots are
also retained, but exporter refresh can lag; primary acceptance comes from the
existing host sampler bookkeeping, not differencing stale exporter gauges.
No historical measurements can be recovered before this instrumentation exists.
