#!/usr/bin/env bash
# =============================================================================
#  bench_serve_v4.sh  —  SERVE-ONLY benchmark, adapted to the DeepSeek-V4
#                        (v4_w8a8) working command.
#
#  Copy of bench_official.sh trimmed to `serve` only, with serve flags set to
#  match your known-good V4 command, and carrying the latest serve-mode
#  additions: the [EXPERT-OFFLOAD-FINAL] seq-stats summary, engine-core worker
#  reaping on exit, and the topk-aware offload-threshold warning.
#
#  It does NOT set any env vars (PYTHONPATH, NIC/IP, sysctl, VLLM_*, USE_MULTI_*,
#  VLLM_ASCEND_ENABLE_FUSED_MC2, mooncake LD_LIBRARY_PATH, ...). Source YOUR env
#  first, exactly as you do today:
#      source <your_v4_env>.sh
#      ./bench_serve_v4.sh
#  Preview without launching:  DRY_RUN=1 ./bench_serve_v4.sh
# =============================================================================
set -uo pipefail

# ─── EDIT HERE ▸ 1. WHAT TO RUN ──────────────────────────────────────────────
MODEL="${MODEL:-/home/keyi/llms/deepseekv4-w8a8}"    # Target model w4a8 qunat version, a5 only support w4a8, a3 support both
REMOE_GATE="${REMOE_GATE:-}"
CARD="${CARD:-6}"                                    # NPU id (your env uses 7)
SERVED_NAME="${SERVED_NAME:-glm-5}"                  # must match server <-> bench client
PORT="${PORT:-7001}"

# ─── EDIT HERE ▸ 2. WORKLOAD (serve client) ──────────────────────────────────
DATASET="${DATASET:-sharegpt}"                                                                                         # random | random-mm | sharegpt | sonnet | hf | ...
DATASET_PATH="${DATASET_PATH:-/home/keyi/llms/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json}"                     # required for non-random datasets
ILEN="${ILEN:-256}"                                  # input length  (tokens)
OLEN="${OLEN:-128}"                                  # output length (tokens)
NUM="${NUM:-10}"                                     # number of prompts
RANGE_RATIO="${RANGE_RATIO:-0}"                      # 0 = fixed lengths (random datasets)
RATE="${RATE:-inf}"                                  # request rate req/s; inf = max load

# ─── EDIT HERE ▸ 3. ENGINE KNOBS (mirror your V4 command) ────────────────────
TP="${TP:-1}"                                        # --tensor-parallel-size
DP="${DP:-1}"                                        # --data-parallel-size
SEED="${SEED:-1024}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"                 # your V4 command used 0.9
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"                # empty = model native
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-}"      # empty = vLLM default
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"                  # 1 = --enforce-eager

# ─── EDIT HERE ▸ 4. MoE OFFLOAD FEATURE (V4 values from your command) ─────────
OFFLOAD="${OFFLOAD:-1}"                              # 1 = enable expert offload
NUM_DEVICE_EXPERTS="${NUM_DEVICE_EXPERTS:-24}"       # decode : resident experts per layer
NUM_DEVICE_LAYERS="${NUM_DEVICE_LAYERS:-1}"          # prefill: full-expert layers on NPU
TOPK="${TOPK:-8}"                                    # model top_k; ONLY used for the threshold warning below
CACHE_POLICY="${CACHE_POLICY:-1}"                    # 1 = LRC eviction policy
CACHE_DEBUG="${CACHE_DEBUG:-0}"                      # your V4 command used true; set 0 for clean perf (seq-stats still summarizes)
CPU_BINDING="${CPU_BINDING:-0}"                      # enable_cpu_binding
WPREFETCH="${WPREFETCH:-0}"                          # 1 = L2 weight prefetch (not in your command)

# PROACTIVE EXPERT PREFETCH (the v5.0 next-layer predictive prefetch).
# This is the feature you want to test "with prefetch enabled". It is OFF by
# default in the engine, so it must be requested explicitly here.
#   PREFETCH=1 -> "expert_prefetch_enabled":true   (default for this script)
#   PREFETCH=0 -> omit it / false  (baseline for an A/B against prefetch=on)
# Needs CACHE_POLICY=1 (LRC supplies choose_victim) and the v5.0 build with the
# prefetch feature. Decode-only: requires MAX_NUM_SEQS <= offload_threshold.
PREFETCH="${PREFETCH:-1}"                            # 1 = enable proactive expert prefetch
# EXPERT_PREFETCH_MAX — cap the experts each prefetch LOADS to the top-N by
# predicted score (fate + both learned predictors). Empty = no cap (load every
# predicted miss; current behavior). 1 = only the highest-score missing expert;
# 2 = top-2; etc. Emitted into expert_offload_config only when PREFETCH=1 and
# non-empty, so PREFETCH=0 baselines and pre-cap builds start unchanged.
EXPERT_PREFETCH_MAX="${EXPERT_PREFETCH_MAX:-}"
ON_DEMAND_LOAD_MAX="${ON_DEMAND_LOAD_MAX:-}"  # deprecated, this conflicts SMoE expert substitution

# PREDICTOR — which next-layer expert predictor the prefetch path uses,
# i.e. the expert_offload_config "expert_predictor" enum. Default is the
# cross-layer-gate FATE method.
#   PREDICTOR=""      -> omit the key; engine uses its default ("fate").
#   PREDICTOR=fate    -> explicit FATE (same behavior, key emitted).
#   PREDICTOR=<name>  -> any other registered predictor (e.g. "nn").
# Only takes effect when PREFETCH=1 (the predictor is consumed only by the
# prefetch path). Left empty by default so PREFETCH=0 baselines AND any build
# that predates the predictor-enum patch (which would reject an unknown
# "expert_predictor" key) start unchanged. Set a non-empty value only on a
# build that has the predictor-enum refactor applied.
PREDICTOR="${PREDICTOR:-fate}"                                     # existing

# Checkpoint for learned predictors (e.g. mode2_pa_prevhs). Emitted into the
# PREDICTOR=mode2_pa_prevhs
# PREDICTOR=mode2_prevpa_prevhs_2tower
# PREDICTOR=mode2_pa_prevhs_ptlg
# PREDICTOR=mode2_prevpa_prevhs_ptlg
# config as "expert_predictor_ckpt". Default = mode2_pa_prevhs lowrank w=2048.
PREDICTOR_CKPT="${PREDICTOR_CKPT:-/path/pa+prevhs_prompt_std_dist_lowrank_w=2048.pt}"
# PREDICTOR_CKPT="${PREDICTOR_CKPT:-/path/prevpa+prevhs_prompt_std_dist_twotower_w=2048.pt}"
# PREDICTOR_CKPT="${PREDICTOR_CKPT:-/path/pa+prevhs+ptlg_prompt_std_dist_lowrank_w=2048_miss=0.pt}"
# PREDICTOR_CKPT="${PREDICTOR_CKPT:-/path/prevpa+prevhs+ptlg_prompt_std_dist_lowrank_w=2048_miss=0.pt}"

EXPERT_SUBSTITUTION="${EXPERT_SUBSTITUTION:-0}"
EXPERT_SUBSTITUTION_THRESHOLD="${EXPERT_SUBSTITUTION_THRESHOLD:-0.25}"

# end-of-test hit-rate summary ([EXPERT-OFFLOAD-FINAL]). Requires the
# seq-stats patch on the build AND CACHE_POLICY=1. Serve window: warmup=0,
# num=NUM (at MAX_NUM_SEQS=1, 1 request = 1 sequence).
# !! Your V4 build is a DIFFERENT checkout (moeoffload_v5 / vLLM 0.21.0). If it
#    lacks the seq-stats patch, the config validator REJECTS these keys and the
#    engine won't start -> set SEQ_STATS=0 to reproduce the original V4 JSON.
SEQ_STATS="${SEQ_STATS:-1}"                          # 1 = pass seq_stats_* keys
TIMING="${TIMING:-0}"                                # 1 = per-layer compute/upload timing (eager only)

export CSV="${CSV:-1}"
export CSV_PATH=/home/keyi/code/vllm-ascend-moe5/bench_csvs

SEQ_WARMUP="${SEQ_WARMUP:-0}"
SEQ_NUM="${SEQ_NUM:-${NUM}}"

QUANT="${QUANT:-ascend}"                             # your V4 command uses ascend; QUANT=none to disable
EP="${EP:-1}"                                        # 1 = --enable-expert-parallel (your command has it)

# ─── EDIT HERE ▸ 5. DeepSeek-V4 PARSER/TOKENIZER FLAGS (the V4 adaptation) ────
V4_MODE="${V4_MODE:-1}"                              # 1 = add the deepseek_v4 tokenizer/tool/reasoning flags

# ─── EDIT HERE ▸ 6. PROFILER (your command has it ON) ────────────────────────
PROFILE="${PROFILE:-0}"                             # 1 = attach torch profiler (set 0 for clean perf numbers)
PROFILE_DIR="${PROFILE_DIR:-./profile_results}"
PROFILE_WITH_STACK="${PROFILE_WITH_STACK:-0}"

# ─── EDIT HERE ▸ 7. ADVANCED (usually leave as-is) ───────────────────────────
ADDITIONAL_CONFIG_RAW="${ADDITIONAL_CONFIG_RAW:-}"   # paste full JSON to OVERRIDE section 4
EXTRA_ENGINE_FLAGS=( ${EXTRA_ENGINE_ARGS:-} )        # raw flags appended to `vllm serve`
EXTRA_BENCH_FLAGS=(  ${EXTRA_BENCH_ARGS:-}  )        # raw flags appended to `vllm bench serve` (client)
SERVE_ONLY_FLAGS=( --enable-chunked-prefill --enable-prefix-caching
                   --aggregate-engine-logging --safetensors-load-strategy prefetch
                   --api-server-count 1 )
WAIT="${WAIT:-1000}"                                  # seconds to wait for /health
DRY_RUN="${DRY_RUN:-0}"                               # 1 = print command(s), do not execute


# =============================================================================
#  MACHINERY  —  you normally don't need to edit below this line
# =============================================================================
export ASCEND_RT_VISIBLE_DEVICES="${CARD}"
 
die()  { echo "ERROR: $*" >&2; exit 1; }
bool() { [[ "${1}" == "1" ]] && echo true || echo false; }
show() { printf '+ %s\n' "$*"; }
 
# Reap vLLM engine-core worker processes ("VLLMEngineCor*", e.g.
# VLLMEngineCore_DP0) that can outlive serve and keep holding NPU memory.
# Current user only, excludes this shell; SIGTERM then SIGKILL.
SERVE_PID=""
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
  for ((i=0; i<20; i++)); do
    pids="$(_find_engine_cores)"
    [[ -z "${pids}" ]] && { echo "[cleanup] all ${ENGINE_CORE_PREFIX}* processes exited"; return 0; }
    sleep 0.5
  done
  echo "[cleanup] SIGKILL stragglers -> $(echo ${pids})"
  kill -KILL ${pids} 2>/dev/null
}
cleanup_on_exit() {
  local rc=$?
  if [[ -n "${SERVE_PID:-}" ]]; then
    echo "[cleanup] stopping serve pid ${SERVE_PID}"
    kill -TERM "${SERVE_PID}" 2>/dev/null
  fi
  kill_engine_cores
  return "${rc}"
}
 
build_additional_config() {
  if [[ -n "${ADDITIONAL_CONFIG_RAW}" ]]; then printf '%s' "${ADDITIONAL_CONFIG_RAW}"; return; fi
  local parts=()
  if [[ "${OFFLOAD}" == "1" ]]; then
    parts+=( "\"enable_cpu_binding\":$(bool "${CPU_BINDING}")" )
    local prefetch_key="" predictor_key="" prefetch_max_key=""
    if [[ "${PREFETCH}" == "1" ]]; then
      prefetch_key=",\"expert_prefetch_enabled\":true"
      [[ -n "${PREDICTOR}" ]] && predictor_key=",\"expert_predictor\":\"${PREDICTOR}\""
      [[ -n "${PREDICTOR}" && "${PREDICTOR}" != "fate" && -n "${PREDICTOR_CKPT}" ]] && \
        predictor_key="${predictor_key},\"expert_predictor_ckpt\":\"${PREDICTOR_CKPT}\""
      [[ -n "${EXPERT_PREFETCH_MAX}" ]] && prefetch_max_key=",\"expert_prefetch_max\":${EXPERT_PREFETCH_MAX}"
    fi
    local subst_key=""
    [[ -n "${ON_DEMAND_LOAD_MAX}" ]] && subst_key=",\"on_demand_load_max\":${ON_DEMAND_LOAD_MAX}"
    local seq_keys=""
    if [[ "${SEQ_STATS}" == "1" ]]; then
      seq_keys=",\"seq_stats_num_seqs\":${SEQ_NUM}"
      [[ "${TIMING}" == "1" ]] && seq_keys="${seq_keys},\"cache_profile_timing\":true"
    fi
    parts+=( "\"expert_offload_config\":{\"expert_offload\":true,\"num_device_experts\":${NUM_DEVICE_EXPERTS},\"num_device_layers\":${NUM_DEVICE_LAYERS},\"cache_policy_enabled\":$(bool "${CACHE_POLICY}")${subst_key}${prefetch_key}${predictor_key}${prefetch_max_key},\"moe_offload_debug\":$(bool "${CACHE_DEBUG}")${seq_keys},\"expert_substitution_enabled\":$(bool "${EXPERT_SUBSTITUTION}"),\"expert_substitution_threshold\":${EXPERT_SUBSTITUTION_THRESHOLD}}" )
  fi
  [[ "${WPREFETCH}" == "1" ]] && parts+=( "\"weight_prefetch_config\":{\"enabled\":true}" )
  # ReMoE fine-tuned router-gate override. When REMOE_GATE is a non-empty
  # path, inject moe_gate_override_path so vllm-ascend swaps the MoE gate
  # weights in-memory at load time (no on-disk model copy). Independent of
  # OFFLOAD, so it also works with OFFLOAD=0. Empty (default) => key omitted =>
  # feature off, nothing overwritten. (NOTE: ADDITIONAL_CONFIG_RAW still fully
  # overrides this — include the key yourself if you use RAW.)
  [[ -n "${REMOE_GATE}" ]] && parts+=( "\"moe_gate_override_path\":\"${REMOE_GATE}\"" )
  [[ ${#parts[@]} -eq 0 ]] && return
  local IFS=','; printf '{%s}' "${parts[*]}"
}
 
build_engine_args() {
  ENGINE_ARGS=( --trust-remote-code
                --tensor-parallel-size "${TP}"
                --seed "${SEED}"
                --generation-config vllm
                --override-generation-config '{"temperature":0.0,"top_p":1.0}'
                --gpu-memory-utilization "${GPU_MEM_UTIL}" )
  [[ "${ENFORCE_EAGER}" == "1" ]]  && ENGINE_ARGS+=( --enforce-eager )
  [[ -n "${MAX_MODEL_LEN}" ]]      && ENGINE_ARGS+=( --max-model-len "${MAX_MODEL_LEN}" )
  [[ -n "${MAX_BATCHED_TOKENS}" ]] && ENGINE_ARGS+=( --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" )
  [[ -n "${QUANT}" && "${QUANT}" != "none" ]] && ENGINE_ARGS+=( --quantization "${QUANT}" )
  [[ "${EP}" == "1" ]]             && ENGINE_ARGS+=( --enable-expert-parallel )
 
  # DeepSeek-V4 tokenizer / tool-call / reasoning parsers
  if [[ "${V4_MODE}" == "1" ]]; then
    ENGINE_ARGS+=( --tokenizer-mode deepseek_v4
                   --tool-call-parser deepseek_v4
                   --enable-auto-tool-choice
                   --reasoning-parser deepseek_v4 )
  fi
 
  ADDL="$(build_additional_config)"
  [[ -n "${ADDL}" ]]               && ENGINE_ARGS+=( --additional-config "${ADDL}" )
 
  if [[ "${PROFILE}" == "1" ]]; then
    mkdir -p "${PROFILE_DIR}"
    ENGINE_ARGS+=( --profiler-config \
      "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILE_DIR}\",\"torch_profiler_with_stack\":$(bool "${PROFILE_WITH_STACK}")}" )
  fi
  ENGINE_ARGS+=( "${EXTRA_ENGINE_FLAGS[@]}" )
}
 
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
  echo "  serve  model=${MODEL}  card=${CARD}  served_name=${SERVED_NAME}  port=${PORT}"
  echo "  tp=${TP} dp=${DP} util=${GPU_MEM_UTIL} max_num_seqs=${MAX_NUM_SEQS}"\
       "max_model_len=${MAX_MODEL_LEN:-native} max_batched=${MAX_BATCHED_TOKENS:-default} eager=${ENFORCE_EAGER}"
  echo "  quant=${QUANT:-none} ep=${EP} v4_mode=${V4_MODE} profile=${PROFILE}"
  if [[ "${DATASET}" == random* ]]; then
    echo "  dataset=${DATASET} num=${NUM} ilen=${ILEN} olen=${OLEN} rate=${RATE}"
  else
    echo "  dataset=${DATASET} num=${NUM} path=${DATASET_PATH:-none} rate=${RATE}"
  fi
  # CHANGE: disambiguate the two prefetch features — expert_prefetch (PREFETCH,
  # the v5.0 MoE feature) vs wprefetch (WPREFETCH, the L2 weight prefetch).
  # CHANGE: also show predictor= (the expert_predictor enum); empty -> the
  # engine default "fate".
  echo "  offload=${OFFLOAD} device_experts=${NUM_DEVICE_EXPERTS} device_layers=${NUM_DEVICE_LAYERS}"\
       "lrc=${CACHE_POLICY} expert_prefetch=${PREFETCH} predictor=${PREDICTOR:-fate(default)} dbg=${CACHE_DEBUG} cpu_bind=${CPU_BINDING} wprefetch=${WPREFETCH}"
  if [[ "${OFFLOAD}" == "1" && "${SEQ_STATS}" == "1" ]]; then
    echo "  seq_stats: warmup_seqs=${SEQ_WARMUP} num_seqs=${SEQ_NUM} timing=${TIMING} (grep [EXPERT-OFFLOAD-FINAL] in the server log)"
  fi
  [[ -n "${ADDL}" ]] && echo "  additional-config=${ADDL}"
  echo "────────────────────────────────────────────────────────────────────"
  # offload_threshold = num_device_experts / topk; decode stays on the LRC path
  # (the one that records hit rate / timing AND runs prefetch) only while
  # max_num_seqs <= threshold (else it falls to the prefill-pool path).
  if [[ "${OFFLOAD}" == "1" ]]; then
    local thr
    thr=$(( NUM_DEVICE_EXPERTS / TOPK ))
    if [[ "${MAX_NUM_SEQS}" -gt "${thr}" ]]; then
      echo "  warn: MAX_NUM_SEQS=${MAX_NUM_SEQS} > offload_threshold=${thr} (= ${NUM_DEVICE_EXPERTS}/${TOPK})."
      echo "  warn: decode uses the prefill-pool path; LRC + prefetch won't run and seq-stats record nothing."
      echo "  warn: lower MAX_NUM_SEQS to <= ${thr} or raise NUM_DEVICE_EXPERTS (set TOPK to your model's real top_k)."
    fi
    if [[ "${PREFETCH}" == "1" && "${CACHE_POLICY}" != "1" ]]; then
      echo "  warn: PREFETCH=1 but CACHE_POLICY=0 — prefetch falls back to arbitrary eviction (no LRC brain)."
    fi
    # a predictor choice only matters when prefetch is on; flag the no-op.
    if [[ -n "${PREDICTOR}" && "${PREFETCH}" != "1" ]]; then
      echo "  warn: PREDICTOR='${PREDICTOR}' set but PREFETCH=0 — the predictor is only used by the prefetch path; the expert_predictor key is omitted."
    fi
  fi
}
 
run_serve() {
  mkdir -p ./bench_results
  mkdir -p ./bench_csvs
  local log="./bench_results/server_v4_$(date +%Y%m%d_%H%M%S).log"
  local serve_cmd=( vllm serve "${MODEL}"
                    --host 0.0.0.0 --port "${PORT}"
                    --data-parallel-size "${DP}"
                    --served-model-name "${SERVED_NAME}"
                    --max-num-seqs "${MAX_NUM_SEQS}"
                    "${ENGINE_ARGS[@]}" "${SERVE_ONLY_FLAGS[@]}" )
 
  echo "[serve] launching on :${PORT} (card ${CARD}); server log -> ${log}"
  [[ "${OFFLOAD}" == "1" ]] && echo "[serve] grep '${log}' for [EXPERT-OFFLOAD-FINAL] / [EXPERT-OFFLOAD-CACHE] / [UPDATE-W]"
  show "${serve_cmd[@]}"
  [[ "${DRY_RUN}" == "1" ]] && { echo "[serve] DRY_RUN: client step skipped."; return 0; }
 
  "${serve_cmd[@]}" > >(tee "${log}") 2>&1 &
  SERVE_PID=$!   # reaped (plus engine-core workers) by cleanup_on_exit
 
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
  show vllm bench serve --model "${MODEL}" --served-model-name "${SERVED_NAME}" \
      --base-url "http://127.0.0.1:${PORT}" "${DATASET_ARGS[@]}" \
      --request-rate "${RATE}" --seed "${SEED}" \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
      --save-result --result-dir ./bench_results "${EXTRA_BENCH_FLAGS[@]}"
  vllm bench serve --model "${MODEL}" --served-model-name "${SERVED_NAME}" \
      --base-url "http://127.0.0.1:${PORT}" "${DATASET_ARGS[@]}" \
      --request-rate "${RATE}" --seed "${SEED}" \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
      --save-result --result-dir ./bench_results "${EXTRA_BENCH_FLAGS[@]}"
 
  # seq-stats summary fires on the SEQ_NUM-th finished request (needs the v8
  # Group 3 finished-request hook in model_runner) and lands in the SERVER log;
  # surface it here for convenience.
  if [[ "${OFFLOAD}" == "1" && "${SEQ_STATS}" == "1" ]]; then
    echo "[serve] expert-offload final stats (from ${log}):"
    grep -a "EXPERT-OFFLOAD-FINAL" "${log}" || \
      echo "[serve] no [EXPERT-OFFLOAD-FINAL] line yet — fewer than seq_stats_num_seqs=${SEQ_NUM} requests finished, the finished-request flush hook is not applied, or the seq-stats patch is missing on this build."
  fi
}
 
# ─── dispatch (serve only) ───────────────────────────────────────────────────
build_engine_args
build_dataset_args
print_summary
# reap engine-core workers when the test ends (normal exit OR Ctrl-C/error);
# skipped under DRY_RUN since nothing was launched.
[[ "${DRY_RUN}" != "1" ]] && trap cleanup_on_exit EXIT
run_serve
 
 
# =============================================================================
#  NOTES
# =============================================================================
#  Mirrors your V4 command's serve flags: --enable-expert-parallel,
#  --max-num-seqs 1, --max-model-len 512, --max-num-batched-tokens 512,
#  --gpu-memory-utilization 0.9, --tokenizer-mode/--tool-call-parser/
#  --reasoning-parser deepseek_v4, --enable-auto-tool-choice,
#  --quantization ascend, --additional-config {num_device_experts 24,
#  num_device_layers 1}, --profiler-config {torch, with_stack:true}, plus
#  --enable-chunked-prefill/--enable-prefix-caching/--aggregate-engine-logging/
#  --safetensors-load-strategy prefetch/--api-server-count 1.
#
#  NEW since the last copy:
#    * EXPERT PREFETCH    : PREFETCH=1 (default) adds "expert_prefetch_enabled":
#                           true — the v5.0 proactive next-layer prefetch you
#                           want to test. PREFETCH=0 for the A/B baseline.
#                           Needs CACHE_POLICY=1 and a build with the feature.
#    * PREDICTOR          : selects the expert_offload_config "expert_predictor"
#                           enum (the prediction method behind prefetch). Empty
#                           (default) -> engine default "fate"; set PREDICTOR=nn
#                           etc. to swap. Emitted only when PREFETCH=1 and
#                           non-empty, so baselines / pre-enum builds are
#                           unchanged.
#    * device_experts     : default raised 12 -> 24 to match your V4 command and
#                           give prefetch/LRC slack (threshold 24/8 = 3).
#    * seq-stats summary  : SEQ_STATS=1 adds seq_stats_warmup_seqs/num_seqs
#                           (serve: 0 / NUM) -> one [EXPERT-OFFLOAD-FINAL] line
#                           in the SERVER log. SEQ_STATS=0 reproduces the
#                           original V4 JSON exactly (use on an UNPATCHED build).
#    * TIMING=1           : adds cache_profile_timing (per-layer compute/upload).
#    * engine-core reaping: cleanup_on_exit reaps VLLMEngineCor* workers that
#                           would otherwise keep holding NPU memory after exit.
#    * threshold warning  : computed as num_device_experts/TOPK (set TOPK to V4's
#                           real top_k; only affects the warning).
#
#  BUILD PREREQUISITES (the V4 checkout must have these or the keys are rejected
#  / the logs won't appear):
#    * expert_prefetch_enabled : present on v5.0 (the prefetch branch).
#    * expert_predictor : the pluggable-predictor (enum) refactor. Required ONLY
#      if you set PREDICTOR to a non-empty value. Leave PREDICTOR empty on builds
#      without it (the key is then omitted and the engine default "fate" applies).
#    * seq_stats_* / cache_profile_timing : the summary + timing patches (apply
#      the consolidated v8 + v6/v7 guides to the V4 checkout). If absent, set
#      SEQ_STATS=0 TIMING=0 — you then only get the upstream [EXPERT-OFFLOAD-CACHE]
#      per-interval line (every cache_stats_log_interval calls) and [UPDATE-W].
#    * finished-request flush hook (v8 Group 3) in model_runner.execute_model:
#      required for the [EXPERT-OFFLOAD-FINAL] line to appear DURING the run
#      (before the grep). Without it the summary only prints at engine teardown
#      via atexit, after the client step — the grep would miss it.
#
#  SPEC DECODE / MTP: this script passes no --speculative-config, so
#  num_speculative_tokens=0 and decode steps carry 1 token/seq — fine for
#  logging at MAX_NUM_SEQS=1. If you later enable MTP, each decode step carries
#  (1 + num_spec) tokens/seq; keep MAX_NUM_SEQS*(1+num_spec) <= offload_threshold
#  (raise NUM_DEVICE_EXPERTS to topk*MAX_NUM_SEQS*(1+num_spec)) or the steps fall
#  to the prefill-pool path and record nothing.
#
#  ENV is NOT set here — source your own env first.
#
#  If the `vllm bench serve` CLIENT errors on the deepseek_v4 tokenizer:
#    EXTRA_BENCH_ARGS="--tokenizer-mode deepseek_v4" ./bench_serve_v4.sh
#
#  Clean perf numbers: PROFILE=0 and CACHE_DEBUG=0 (seq-stats still summarizes).
# =============================================================================
