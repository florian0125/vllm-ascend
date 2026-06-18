#!/usr/bin/env bash
# =============================================================================
#  bench_official.sh  —  vLLM-Ascend benchmark  (moe_offload_v2.0 branch)
#
#  Reports official `vllm bench` numbers for MoE expert-offload on Atlas A5.
#
#  Prereq : conda activate vllm-ascend && source env_a5.sh
#  Run    : MODEL=... QUANT=ascend CARD=5 MAX_NUM_SEQS=1 ./bench_official.sh serve
#  Modes  : throughput | latency | serve            (default: throughput)
#  Preview: prepend DRY_RUN=1 to print the exact command without running it.
#
#  HOW TO EDIT: change only the five "EDIT HERE" blocks below. Everything under
#  the "MACHINERY" banner is plumbing you rarely need to touch.
#  (Dataset table, the offload threshold rule, and examples are at the BOTTOM.)
# =============================================================================
set -uo pipefail

# ─── EDIT HERE ▸ 1. WHAT TO RUN ──────────────────────────────────────────────
MODE="${1:-throughput}"                                  # throughput | latency | serve
MODEL="${MODEL:-/data/keyi/llms/DeepSeek-V2-Lite-w8a8}"  # path or ModelScope id (auto-downloads)
CARD="${CARD:-5}"                                        # NPU id from `npu-smi info`

# ─── EDIT HERE ▸ 2. WORKLOAD ─────────────────────────────────────────────────
#   (throughput & serve consume the dataset; latency is synthetic and ignores it)
DATASET="${DATASET:-sharegpt}"                                                                                         # random | random-mm | sharegpt | sonnet | hf | ...
DATASET_PATH="${DATASET_PATH:-/data/keyi/llms/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json}"                     # required for non-random datasets
ILEN="${ILEN:-1024}"                                 # input  length (tokens)
OLEN="${OLEN:-128}"                                  # output length (tokens)
NUM="${NUM:-20}"                                     # number of prompts (throughput & serve)
RANGE_RATIO="${RANGE_RATIO:-0}"                      # 0 = fixed lengths (random datasets)
# CHANGE: BS default 8 -> 4. Latency-mode decode concurrency IS the batch size;
# 4 keeps every decode step on the cache path with NUM_DEVICE_EXPERTS=32
# (offload_threshold = 32 / topk(6) = 5 >= 4), per the new bench-mode policy.
BS="${BS:-4}"                                        # batch size (latency mode only)
# NEW: latency iteration counts (were hardcoded 10/30 in run_latency). Surfaced
# so the seq-stats warmup/measured windows can align with them exactly — each
# latency iteration is one generated batch = one measured "sequence".
ITERS_WARMUP="${ITERS_WARMUP:-10}"                   # latency warmup iterations
ITERS="${ITERS:-30}"                                 # latency measured iterations
RATE="${RATE:-inf}"                                  # request rate req/s (serve); inf = max load

# ─── EDIT HERE ▸ 3. ENGINE KNOBS ─────────────────────────────────────────────
TP="${TP:-1}"                                        # tensor-parallel size (all modes)
DP="${DP:-1}"                                        # data-parallel size (serve only)
SEED="${SEED:-1024}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
# CHANGE: was a flat 256. Mode-dependent default per the new policy: bench
# modes (throughput/latency) default to 4 so decode stays on the cache path
# with 32 device experts; serve defaults to 1 (gold-standard decode/LRC
# exercise). Applies for BOTH OFFLOAD=0 and OFFLOAD=1 so A/B runs are fair.
# An explicit MAX_NUM_SEQS=... still wins.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$([[ "${MODE}" == "serve" ]] && echo 1 || echo 4)}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"                   # empty = model's native max
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-}"         # empty = vLLM default (reference set this = max_model_len)
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"                  # 1 = --enforce-eager (safest for offload)

# ─── EDIT HERE ▸ 4. MoE OFFLOAD FEATURE (the point of this branch) ───────────
#   Requires an int8/W8A8 model AND explicit QUANT=ascend. bf16 is NOT supported.
OFFLOAD="${OFFLOAD:-1}"                              # 1 = enable expert offload
# CHANGE: was a flat 12. Mode-dependent default per the new policy: 32 for
# bench modes (threshold = 32/6 = 5 covers the batch of 4 so the cache takes
# effect), 12 for serve (threshold = 12/6 = 2 covers max_num_seqs=1).
# An explicit NUM_DEVICE_EXPERTS=... still wins.
NUM_DEVICE_EXPERTS="${NUM_DEVICE_EXPERTS:-$([[ "${MODE}" == "serve" ]] && echo 12 || echo 32)}"
NUM_DEVICE_LAYERS="${NUM_DEVICE_LAYERS:-2}"          # prefill: full-expert layers kept on NPU
TOPK="${TOPK:-6}"                                    # NEW: model top_k (DeepSeek-V2-Lite=6); used only for the threshold warning below
CACHE_POLICY="${CACHE_POLICY:-1}"                    # 1 = LRC eviction policy
CACHE_DEBUG="${CACHE_DEBUG:-0}"                      # 1 = per-paging [UPDATE-W] logs (0 = clean perf)
CPU_BINDING="${CPU_BINDING:-0}"                      # enable_cpu_binding (reference = 0)
WPREFETCH="${WPREFETCH:-0}"                          # 1 = L2 weight prefetch (separate feature)

# NEW: end-of-test hit-rate summary ([EXPERT-OFFLOAD-FINAL] line; requires the
# seq-stats patch on this branch — set SEQ_STATS=0 on an UNPATCHED build, the
# config validator rejects unknown keys). Also needs CACHE_POLICY=1 to record.
# Per-mode defaults:
#   latency    warmup=ITERS_WARMUP, num=ITERS   (1 iteration = 1 sequence)
#   serve      warmup=0,            num=NUM     (at MAX_NUM_SEQS=1, 1 request = 1 sequence)
#   throughput warmup=0,            num=0       (continuous batching fragments the
#                                                windows; 0 = print at engine
#                                                teardown via the atexit dump —
#                                                trust the token_mean numbers there)
SEQ_STATS="${SEQ_STATS:-1}"                          # 1 = pass seq_stats_* keys
TIMING="${TIMING:-0}"                                # 1 = per-layer compute/upload timing in the final summary (eager only)
case "${MODE}" in
  latency) SEQ_WARMUP="${SEQ_WARMUP:-${ITERS_WARMUP}}"; SEQ_NUM="${SEQ_NUM:-${ITERS}}" ;;
  serve)   SEQ_WARMUP="${SEQ_WARMUP:-0}";               SEQ_NUM="${SEQ_NUM:-${NUM}}" ;;
  *)       SEQ_WARMUP="${SEQ_WARMUP:-0}";               SEQ_NUM="${SEQ_NUM:-0}" ;;
esac

QUANT="${QUANT:-}"                                   # quantization: empty/unset = OFF; QUANT=ascend = W8A8 models
EP="${EP:-0}"                                        # 1 = --enable-expert-parallel

PROFILE="${PROFILE:-0}"                              # 1 = attach torch_npu profiler
PROFILE_DIR="${PROFILE_DIR:-./bench_results/profiling}"

# ─── EDIT HERE ▸ 5. ADVANCED (usually leave as-is) ───────────────────────────
# Paste a full JSON string to OVERRIDE the offload block built from section 4:
ADDITIONAL_CONFIG_RAW="${ADDITIONAL_CONFIG_RAW:-}"
# Raw passthrough flags (space-separated). _ENGINE -> model engine, _BENCH -> bench client:
EXTRA_ENGINE_FLAGS=( ${EXTRA_ENGINE_ARGS:-} )
EXTRA_BENCH_FLAGS=(  ${EXTRA_BENCH_ARGS:-}  )
# Flags used ONLY by `vllm serve` (add e.g. --block-size 128 here):
SERVE_ONLY_FLAGS=( --enable-chunked-prefill --enable-prefix-caching
                   --aggregate-engine-logging --safetensors-load-strategy prefetch
                   --api-server-count 1 )
PORT="${PORT:-7061}"                                 # serve port
WAIT="${WAIT:-600}"                                  # seconds to wait for serve /health
DRY_RUN="${DRY_RUN:-0}"                              # 1 = print command(s), do not execute


# =============================================================================
#  MACHINERY  —  you normally don't need to edit below this line
# =============================================================================
export ASCEND_RT_VISIBLE_DEVICES="${CARD}"

die()  { echo "ERROR: $*" >&2; exit 1; }
bool() { [[ "${1}" == "1" ]] && echo true || echo false; }
show() { printf '+ %s\n' "$*"; }                     # echo a command line

# NEW: reap vLLM engine-core worker processes (process-title prefix
# "VLLMEngineCor", e.g. VLLMEngineCore_DP0) that can outlive the bench/serve
# command and keep holding NPU memory. Current user only, excludes this shell,
# graceful SIGTERM then SIGKILL. Matches the process NAME (comm) OR the first
# argv token (setproctitle rewrites argv), anchored to the prefix so unrelated
# processes that merely mention the string are never hit.
SERVE_PID=""                                         # set by run_serve; reaped by cleanup_on_exit
ENGINE_CORE_PREFIX="VLLMEngineCor"
_find_engine_cores() {
  ps -u "$(id -u)" -o pid=,comm=,args= 2>/dev/null \
    | awk -v p="${ENGINE_CORE_PREFIX}" -v me="$$" \
          '$1 != me && ($2 ~ "^"p || $3 ~ "^"p) { print $1 }'
}
kill_engine_cores() {
  local pids i
  pids="$(_find_engine_cores)"
  [[ -z "${pids}" ]] && { echo "[cleanup] no ${ENGINE_CORE_PREFIX}* processes to reap"; return 0; }
  echo "[cleanup] SIGTERM ${ENGINE_CORE_PREFIX}* -> $(echo ${pids})"
  kill -TERM ${pids} 2>/dev/null
  for ((i=0; i<20; i++)); do                          # wait up to ~10s for graceful exit
    pids="$(_find_engine_cores)"
    [[ -z "${pids}" ]] && { echo "[cleanup] all ${ENGINE_CORE_PREFIX}* processes exited"; return 0; }
    sleep 0.5
  done
  echo "[cleanup] SIGKILL stragglers -> $(echo ${pids})"
  kill -KILL ${pids} 2>/dev/null
}
# Unified EXIT handler for ALL modes: stop the serve parent (if any), then reap
# any lingering engine-core workers. Runs on normal completion and on Ctrl-C/error.
cleanup_on_exit() {
  local rc=$?
  if [[ -n "${SERVE_PID:-}" ]]; then
    echo "[cleanup] stopping serve pid ${SERVE_PID}"
    kill -TERM "${SERVE_PID}" 2>/dev/null
  fi
  kill_engine_cores
  return "${rc}"
}

# Build the --additional-config JSON from section 4 (or honor the raw override).
build_additional_config() {
  if [[ -n "${ADDITIONAL_CONFIG_RAW}" ]]; then printf '%s' "${ADDITIONAL_CONFIG_RAW}"; return; fi
  local parts=()
  if [[ "${OFFLOAD}" == "1" ]]; then
    parts+=( "\"enable_cpu_binding\":$(bool "${CPU_BINDING}")" )
    # NEW: append the end-of-test summary keys when SEQ_STATS=1. Kept as a
    # separate suffix so SEQ_STATS=0 reproduces the pre-patch JSON byte-for-
    # byte (needed when running this script against an unpatched build).
    local seq_keys=""
    if [[ "${SEQ_STATS}" == "1" ]]; then
      seq_keys=",\"seq_stats_warmup_seqs\":${SEQ_WARMUP},\"seq_stats_num_seqs\":${SEQ_NUM}"
      [[ "${TIMING}" == "1" ]] && seq_keys="${seq_keys},\"cache_profile_timing\":true"
    fi
    parts+=( "\"expert_offload_config\":{\"expert_offload\":true,\"num_device_experts\":${NUM_DEVICE_EXPERTS},\"num_device_layers\":${NUM_DEVICE_LAYERS},\"cache_policy_enabled\":$(bool "${CACHE_POLICY}"),\"cache_debug_log_updates\":$(bool "${CACHE_DEBUG}")${seq_keys}}" )
  fi
  [[ "${WPREFETCH}" == "1" ]] && parts+=( "\"weight_prefetch_config\":{\"enabled\":true}" )
  [[ ${#parts[@]} -eq 0 ]] && return
  local IFS=','; printf '{%s}' "${parts[*]}"
}

# Engine args shared by ALL THREE modes (bench throughput/latency and serve).
build_engine_args() {
  ENGINE_ARGS=( --trust-remote-code
                --tensor-parallel-size "${TP}"
                --seed "${SEED}"
                --gpu-memory-utilization "${GPU_MEM_UTIL}" )
  [[ "${ENFORCE_EAGER}" == "1" ]] && ENGINE_ARGS+=( --enforce-eager )
  [[ -n "${MAX_MODEL_LEN}" ]]     && ENGINE_ARGS+=( --max-model-len "${MAX_MODEL_LEN}" )
  [[ -n "${MAX_BATCHED_TOKENS}" ]] && ENGINE_ARGS+=( --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" )
  # empty = quantization OFF; 'none' tolerated as OFF (not a valid vllm value)
  [[ -n "${QUANT}" && "${QUANT}" != "none" ]] && ENGINE_ARGS+=( --quantization "${QUANT}" )
  [[ "${EP}" == "1" ]]            && ENGINE_ARGS+=( --enable-expert-parallel )

  ADDL="$(build_additional_config)"
  [[ -n "${ADDL}" ]]              && ENGINE_ARGS+=( --additional-config "${ADDL}" )

  if [[ "${PROFILE}" == "1" ]]; then
    mkdir -p "${PROFILE_DIR}"
    ENGINE_ARGS+=( --profiler-config \
      "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILE_DIR}\",\"torch_profiler_with_stack\":true,\"torch_profiler_with_memory\":false,\"torch_profiler_record_shapes\":true}" )
  fi
  ENGINE_ARGS+=( "${EXTRA_ENGINE_FLAGS[@]}" )
}

# Dataset args shared by throughput and the serve client.
build_dataset_args() {
  DATASET_ARGS=( --dataset-name "${DATASET}" --num-prompts "${NUM}" )
  case "${DATASET}" in
    random|random-mm|random-rerank)
      DATASET_ARGS+=( --random-input-len "${ILEN}" --random-output-len "${OLEN}" )
      [[ -n "${RANGE_RATIO}" ]] && DATASET_ARGS+=( --random-range-ratio "${RANGE_RATIO}" ) ;;
    *)
      if [[ -n "${DATASET_PATH}" ]]; then DATASET_ARGS+=( --dataset-path "${DATASET_PATH}" )
      else echo "WARN: dataset '${DATASET}' usually needs DATASET_PATH=... (continuing)"; fi ;;
  esac
}

print_summary() {
  echo "────────────────────────────────────────────────────────────────────"
  echo "  mode=${MODE}  model=${MODEL}  card=${CARD}"
  echo "  tp=${TP} dp=${DP} util=${GPU_MEM_UTIL} max_num_seqs=${MAX_NUM_SEQS}"\
       "max_model_len=${MAX_MODEL_LEN:-native} max_batched=${MAX_BATCHED_TOKENS:-default} eager=${ENFORCE_EAGER} quant=${QUANT:-none}"
  if [[ "${DATASET}" == random* ]]; then
    echo "  dataset=${DATASET} num=${NUM} ilen=${ILEN} olen=${OLEN}"
  else
    echo "  dataset=${DATASET} num=${NUM} path=${DATASET_PATH:-none}"
  fi
  echo "  offload=${OFFLOAD} device_experts=${NUM_DEVICE_EXPERTS} device_layers=${NUM_DEVICE_LAYERS}"\
       "lrc=${CACHE_POLICY} dbg=${CACHE_DEBUG} cpu_bind=${CPU_BINDING} prefetch=${WPREFETCH} profile=${PROFILE}"
  # NEW: surface the summary-window settings next to the offload knobs.
  if [[ "${OFFLOAD}" == "1" && "${SEQ_STATS}" == "1" ]]; then
    echo "  seq_stats: warmup_seqs=${SEQ_WARMUP} num_seqs=${SEQ_NUM}"\
         "(grep [EXPERT-OFFLOAD-FINAL]; num_seqs=0 prints at engine teardown)"
  fi
  [[ -n "${ADDL}" ]] && echo "  additional-config=${ADDL}"
  echo "────────────────────────────────────────────────────────────────────"
  if [[ "${OFFLOAD}" == "1" ]]; then
    local conc knob thr
    if [[ "${MODE}" == "latency" ]]; then conc="${BS}"; knob="BS"; else conc="${MAX_NUM_SEQS}"; knob="MAX_NUM_SEQS"; fi
    # CHANGE: was a hardcoded "> 2" (written for the old 12-expert default and
    # falsely firing at the new bench defaults conc=4/ndev=32). Compute the
    # real rule: offload_threshold = num_device_experts / topk.
    thr=$(( NUM_DEVICE_EXPERTS / TOPK ))
    if [[ "${conc}" -gt "${thr}" ]]; then
      echo "  warn: ${knob}=${conc} > offload_threshold=${thr} (= ${NUM_DEVICE_EXPERTS}/${TOPK})."
      echo "  warn: decode uses the prefill-pool path; LRC won't run and seq-stats record nothing."
      echo "  warn: lower ${knob} to <= ${thr} or raise NUM_DEVICE_EXPERTS."
    fi
  fi
  if [[ -z "${QUANT}" || "${QUANT}" == "none" ]] && [[ "${MODEL}" == *[Ww]8[Aa]8* ]]; then
    echo "  warn: model name looks W8A8-quantized but quantization is OFF — did you forget QUANT=ascend?"
  fi
}

run_throughput() {
  show vllm bench throughput --model "${MODEL}" "${ENGINE_ARGS[@]}" "${DATASET_ARGS[@]}" \
      --max-num-seqs "${MAX_NUM_SEQS}" "${EXTRA_BENCH_FLAGS[@]}"
  [[ "${DRY_RUN}" == "1" ]] && return 0
  vllm bench throughput --model "${MODEL}" "${ENGINE_ARGS[@]}" "${DATASET_ARGS[@]}" \
      --max-num-seqs "${MAX_NUM_SEQS}" "${EXTRA_BENCH_FLAGS[@]}"
}

run_latency() {
  [[ "${DATASET}" != "random" ]] && echo "NOTE: latency is synthetic-only; DATASET='${DATASET}' ignored."
  # CHANGE: iteration counts were hardcoded 10/30; now ITERS_WARMUP/ITERS so
  # they stay in lockstep with seq_stats_warmup_seqs/seq_stats_num_seqs above.
  show vllm bench latency --model "${MODEL}" "${ENGINE_ARGS[@]}" \
      --input-len "${ILEN}" --output-len "${OLEN}" --batch-size "${BS}" \
      --num-iters-warmup "${ITERS_WARMUP}" --num-iters "${ITERS}" "${EXTRA_BENCH_FLAGS[@]}"
  [[ "${DRY_RUN}" == "1" ]] && return 0
  vllm bench latency --model "${MODEL}" "${ENGINE_ARGS[@]}" \
      --input-len "${ILEN}" --output-len "${OLEN}" --batch-size "${BS}" \
      --num-iters-warmup "${ITERS_WARMUP}" --num-iters "${ITERS}" "${EXTRA_BENCH_FLAGS[@]}"
}

run_serve() {
  mkdir -p ./bench_results
  local log="./bench_results/server_$(date +%Y%m%d_%H%M%S).log"
  local serve_cmd=( vllm serve "${MODEL}"
                    --host 0.0.0.0 --port "${PORT}"
                    --data-parallel-size "${DP}"
                    --served-model-name bench
                    --max-num-seqs "${MAX_NUM_SEQS}"
                    "${ENGINE_ARGS[@]}" "${SERVE_ONLY_FLAGS[@]}" )

  echo "[serve] launching on :${PORT} (card ${CARD}); server log -> ${log}"
  # CHANGE: added [EXPERT-OFFLOAD-FINAL] to the grep hint — the end-of-test
  # summary lands in the SERVER log (printed by the engine worker), not in
  # the bench-client output.
  [[ "${OFFLOAD}" == "1" ]] && echo "[serve] grep '${log}' for [EXPERT-OFFLOAD-FINAL] / [EXPERT-OFFLOAD-CACHE] / [UPDATE-W]"
  show "${serve_cmd[@]}"
  [[ "${DRY_RUN}" == "1" ]] && { echo "[serve] DRY_RUN: client step skipped."; return 0; }

  "${serve_cmd[@]}" > >(tee "${log}") 2>&1 &
  local pid=$!
  # CHANGE: record the serve PID globally instead of installing a local EXIT
  # trap; the unified cleanup_on_exit trap (installed before dispatch) stops
  # this PID and then reaps any lingering VLLMEngineCor* engine-core workers.
  SERVE_PID="${pid}"

  echo "[serve] waiting up to ${WAIT}s for /health ..."
  local ok=0 i
  for ((i=0; i<WAIT/2; i++)); do
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { ok=1; break; }
    sleep 2
  done
  [[ "${ok}" -ne 1 ]] && die "server not ready in ${WAIT}s (see ${log})"

  echo "[serve] ready -> running vllm bench serve"
  # Talk DIRECT to localhost (corporate http_proxy otherwise hangs 127.0.0.1).
  export no_proxy="127.0.0.1,localhost,${no_proxy:-}"; export NO_PROXY="${no_proxy}"
  show vllm bench serve --model "${MODEL}" --served-model-name bench \
      --base-url "http://127.0.0.1:${PORT}" "${DATASET_ARGS[@]}" \
      --request-rate "${RATE}" --seed "${SEED}" \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
      --save-result --result-dir ./bench_results "${EXTRA_BENCH_FLAGS[@]}"
  vllm bench serve --model "${MODEL}" --served-model-name bench \
      --base-url "http://127.0.0.1:${PORT}" "${DATASET_ARGS[@]}" \
      --request-rate "${RATE}" --seed "${SEED}" \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
      --save-result --result-dir ./bench_results "${EXTRA_BENCH_FLAGS[@]}"
  # NEW: the seq-stats summary fires on the NUM-th finished request (Group 3
  # hook) and lands in the server log; surface it here for convenience.
  if [[ "${OFFLOAD}" == "1" && "${SEQ_STATS}" == "1" ]]; then
    echo "[serve] expert-offload final stats (from ${log}):"
    grep -a "EXPERT-OFFLOAD-FINAL" "${log}" || \
      echo "[serve] no [EXPERT-OFFLOAD-FINAL] line yet — fewer than seq_stats_num_seqs=${SEQ_NUM} requests finished, or the seq-stats patch / Group 3 hook is missing."
  fi
}

# ─── dispatch ────────────────────────────────────────────────────────────────
build_engine_args
build_dataset_args
print_summary
# NEW: ensure engine-core workers are reaped when the test ends — any mode,
# normal exit OR Ctrl-C/error. Skipped under DRY_RUN (nothing was launched).
[[ "${DRY_RUN}" != "1" ]] && trap cleanup_on_exit EXIT
case "${MODE}" in
  throughput) run_throughput ;;
  latency)    run_latency ;;
  serve)      run_serve ;;
  *)          die "unknown mode '${MODE}' (use: throughput | latency | serve)" ;;
esac


# =============================================================================
#  REFERENCE NOTES
# =============================================================================
#  DATASETS (throughput & serve; latency is synthetic-only):
#    random        synthetic text (default)              no DATASET_PATH
#    random-mm     synthetic multimodal (needs MM model) no DATASET_PATH
#    sharegpt      realistic chat                         DATASET_PATH=ShareGPT_V3_*.json
#    sonnet        controllable prefix/len                DATASET_PATH=sonnet.txt
#    hf            any HuggingFace dataset                DATASET_PATH=<hf id>
#
#  OFFLOAD requires an int8/W8A8 model (QUANT=ascend). On bf16 it crashes in
#  create_prefill_pool (the offload weight-copy path is int8-only by design).
#  Get a W8A8 model, e.g.:  modelscope download --model vllm-ascend/DeepSeek-V2-Lite-W8A8
#
#  DECODE-PATH RULE: the engine pages experts per-token (the LRC path) only when
#    tokens_per_step <= offload_threshold = num_device_experts / topk.
#  DeepSeek-V2-Lite has topk=6. Mode defaults are sized to satisfy this rule:
#    bench (throughput/latency): NUM_DEVICE_EXPERTS=32 -> threshold 5, with
#      MAX_NUM_SEQS=4 (throughput) / BS=4 (latency) decode batches.
#    serve: NUM_DEVICE_EXPERTS=12 -> threshold 2, with MAX_NUM_SEQS=1.
#  Exceeding the threshold falls back to the prefill-pool path: LRC never runs
#  and the seq-stats summary records nothing (print_summary warns about this).
#
#  SEQ-STATS SUMMARY (requires the seq-stats patch + CACHE_POLICY=1):
#    One [EXPERT-OFFLOAD-FINAL] line per process — at the SEQ_NUM-th measured
#    sequence, or at engine teardown when SEQ_NUM=0 (the throughput default,
#    where continuous batching fragments per-sequence windows; read the
#    token_mean numbers there). Latency aligns SEQ_WARMUP/SEQ_NUM with
#    ITERS_WARMUP/ITERS automatically. Set SEQ_STATS=0 on an UNPATCHED build —
#    the config validator rejects the unknown keys.
#
#  EXAMPLES:
#    # offload + LRC exercised (serve, gold-standard latency metrics; ndev=12, seqs=1)
#    MODEL=vllm-ascend/DeepSeek-V2-Lite-W8A8 QUANT=ascend CARD=5 ./bench_official.sh serve
#    # offline throughput (defaults: seqs=4, ndev=32, summary at teardown)
#    MODEL=... QUANT=ascend CARD=5 ./bench_official.sh throughput
#    # offline latency (defaults: BS=4, ndev=32, summary after ITERS=30 iterations)
#    MODEL=... QUANT=ascend CARD=5 ./bench_official.sh latency
#    # report-grade perf (silence per-paging logs)
#    MODEL=... CARD=5 CACHE_DEBUG=0 ./bench_official.sh throughput
#    # baseline for A/B (offload off; same seqs=4 default keeps the comparison fair)
#    MODEL=... CARD=5 OFFLOAD=0 ./bench_official.sh throughput
#    # running against an UNPATCHED build (no seq-stats keys in the config)
#    MODEL=... CARD=5 SEQ_STATS=0 ./bench_official.sh throughput
#    # preview the exact command without running
#    MODEL=... CARD=5 DRY_RUN=1 ./bench_official.sh serve
#    # full manual control of the offload JSON
#    ADDITIONAL_CONFIG_RAW='{"expert_offload_config":{"expert_offload":true,"num_device_experts":8,"num_device_layers":2}}' ./bench_official.sh throughput
#
#  VERIFY flag spellings if a flag is rejected (they drift across versions):
#    vllm bench throughput --help | grep -iE 'dataset|num-prompt|additional-config|profiler-config'
#    vllm serve            --help | grep -iE 'additional-config|profiler-config'
# =============================================================================