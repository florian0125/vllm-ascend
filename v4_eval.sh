#!/usr/bin/env bash
# =============================================================================
#  serve_eval.sh — launch the DeepSeek-V4 offload server, wait for /health, then
#  run lm-eval (gsm8k + mmlu) against it via the local-completions client.
#
#  Datasets are read from a pre-seeded HF cache (no per-task YAML, no download).
#  Seed it ONCE with network (see the preflight message if it's missing).
#
#  Usage:   source <your_v4_env>.sh
#           ./serve_eval.sh
#  Preview: DRY_RUN=1 ./serve_eval.sh
# =============================================================================
set -uo pipefail

# ── server ───────────────────────────────────────────────────────────────────
MODEL="${MODEL:-/home/keyi/llms/deepseekv4-w8a8}"
REMOE_GATE="${REMOE_GATE:-}"            # fine-tuned MoE gate dir/file; empty = off (adds moe_gate_override_path)
SERVED_NAME="${SERVED_NAME:-glm-5}"     # lm-eval model= MUST match this
CARD="${CARD:-6}"
PORT="${PORT:-7001}"
TP="${TP:-1}"; DP="${DP:-1}"
SEED="${SEED:-1024}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"       # keep <= num_device_experts/topk to hold the offload path

# ── MoE offload (the config whose accuracy you're measuring) ─────────────────
OFFLOAD="${OFFLOAD:-1}"
NUM_DEVICE_EXPERTS="${NUM_DEVICE_EXPERTS:-24}"
NUM_DEVICE_LAYERS="${NUM_DEVICE_LAYERS:-1}"
CACHE_POLICY="${CACHE_POLICY:-1}"       # 1 = LRC (required by prefetch)
PREFETCH="${PREFETCH:-1}"               # 1 = proactive expert prefetch
PREDICTOR="${PREDICTOR:-fate}"
PREDICTOR_CKPT="${PREDICTOR_CKPT:-}"    # only used when PREDICTOR != fate

# ── eval ─────────────────────────────────────────────────────────────────────
TASKS="${TASKS:-gsm8k,mmlu}"
LIMIT="${LIMIT:-5}"                    # per task; empty = full
TOKENIZER="${TOKENIZER:-/home/keyi/llms/dsv4-tokenizer}"  # tokenizer-only dir (no config.json)
MAX_LENGTH="${MAX_LENGTH:-8192}"
CONCURRENCY="${CONCURRENCY:-1}"
OUT_DIR="${OUT_DIR:-./eval_results/$(date +%Y%m%d_%H%M%S)}"

# ── dataset cache (pre-seeded; no download) ──────────────────────────────────
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/keyi/llms/benchmarks/keyi/hf_cache}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export USE_MODELSCOPE_HUB=0

WAIT="${WAIT:-1000}"
DRY_RUN="${DRY_RUN:-0}"

# =============================================================================
export ASCEND_RT_VISIBLE_DEVICES="${CARD}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"                 # client demands a key; value ignored
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"; export NO_PROXY="${no_proxy}"

die() { echo "ERROR: $*" >&2; exit 1; }

# reap vLLM engine-core workers that outlive serve and keep holding NPU memory
SERVE_PID=""
reap() {
  local pids; pids=$(pgrep -u "$(id -u)" -f 'VLLMEngineCor' 2>/dev/null | grep -v "^$$\$" || true)
  [[ -n "${SERVE_PID}" ]] && kill -TERM "${SERVE_PID}" 2>/dev/null
  [[ -n "${pids}" ]] && { kill -TERM ${pids} 2>/dev/null; sleep 3; kill -KILL ${pids} 2>/dev/null; }
  return 0
}

# ── additional-config JSON (offload) ─────────────────────────────────────────
offload_json() {
  [[ "${OFFLOAD}" != "1" ]] && return
  local p="\"expert_offload\":true,\"num_device_experts\":${NUM_DEVICE_EXPERTS}"
  p="${p},\"num_device_layers\":${NUM_DEVICE_LAYERS},\"cache_policy_enabled\":$([[ ${CACHE_POLICY} == 1 ]] && echo true || echo false)"
  [[ "${PREFETCH}" == "1" ]] && {
    p="${p},\"expert_prefetch_enabled\":true,\"expert_predictor\":\"${PREDICTOR}\""
    [[ "${PREDICTOR}" != "fate" && -n "${PREDICTOR_CKPT}" ]] && p="${p},\"expert_predictor_ckpt\":\"${PREDICTOR_CKPT}\""
  }
  printf '{"expert_offload_config":{%s}}' "${p}"
}

# ── serve command ────────────────────────────────────────────────────────────
build_serve() {
  SERVE=( vllm serve "${MODEL}"
          --host 0.0.0.0 --port "${PORT}"
          --served-model-name "${SERVED_NAME}"
          --tensor-parallel-size "${TP}" --data-parallel-size "${DP}"
          --max-num-seqs "${MAX_NUM_SEQS}"
          --seed "${SEED}" --gpu-memory-utilization "${GPU_MEM_UTIL}"
          --enforce-eager --quantization ascend --enable-expert-parallel
          --trust-remote-code
          --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4
          --enable-auto-tool-choice --reasoning-parser deepseek_v4
          --enable-chunked-prefill --enable-prefix-caching
          --safetensors-load-strategy prefetch --api-server-count 1 )
  local addl; addl="$(offload_json)"
  # ReMoE fine-tuned router-gate override.
  if [[ -n "${REMOE_GATE}" ]]; then
    local gate="\"moe_gate_override_path\":\"${REMOE_GATE}\""
    if [[ -n "${addl}" ]]; then
      addl="${addl%\}}"           # drop the offload object's trailing brace
      addl="${addl},${gate}}"     # append the gate key as a sibling, re-close
    else
      addl="{${gate}}"            # OFFLOAD=0: gate override is the only key
    fi
  fi
  [[ -n "${addl}" ]] && SERVE+=( --additional-config "${addl}" )
}

# ── eval command ─────────────────────────────────────────────────────────────
build_eval() {
  local margs="model=${SERVED_NAME},base_url=http://127.0.0.1:${PORT}/v1/completions"
  margs="${margs},num_concurrent=${CONCURRENCY},max_retries=3,tokenized_requests=False"
  margs="${margs},tokenizer=${TOKENIZER},max_length=${MAX_LENGTH}"
  EVAL=( lm_eval --model local-completions --model_args "${margs}"
         --tasks "${TASKS}" --batch_size 1 --output_path "${OUT_DIR}" --log_samples )
  [[ -n "${LIMIT}" ]] && EVAL+=( --limit "${LIMIT}" )
}

# ── preflight ────────────────────────────────────────────────────────────────
preflight() {
  [[ -d "${TOKENIZER}" ]] || die "TOKENIZER dir not found: ${TOKENIZER} (tokenizer.json + tokenizer_config.json, NO config.json)"
  [[ -d "${HF_DATASETS_CACHE}" ]] || {
    echo "[!!] No dataset cache at ${HF_DATASETS_CACHE}. Seed it ONCE (with network):" >&2
    echo "     HF_DATASETS_CACHE=${HF_DATASETS_CACHE} python -c \"from datasets import load_dataset;" >&2
    echo "       load_dataset('openai/gsm8k','main'); load_dataset('cais/mmlu','all')\"" >&2
    die "dataset cache not seeded"
  }
}

# ── run ──────────────────────────────────────────────────────────────────────
build_serve; build_eval

echo "────────────────────────────────────────────────────────────"
echo "  serve : ${SERVED_NAME} on :${PORT} (card ${CARD})  offload=${OFFLOAD} prefetch=${PREFETCH} predictor=${PREDICTOR}"
echo "  eval  : tasks=${TASKS} limit=${LIMIT:-full} tokenizer=${TOKENIZER}"
echo "  cache : ${HF_DATASETS_CACHE}   out=${OUT_DIR}"
echo "────────────────────────────────────────────────────────────"
printf '+ %q ' "${SERVE[@]}"; echo
printf '+ %q ' "${EVAL[@]}";  echo
[[ "${DRY_RUN}" == "1" ]] && { echo "[DRY_RUN] nothing launched."; exit 0; }

preflight
mkdir -p "${OUT_DIR}" ./bench_results
trap reap EXIT

log="./bench_results/serve_$(date +%Y%m%d_%H%M%S).log"
echo "[serve] launching; log -> ${log}"
"${SERVE[@]}" > >(tee "${log}") 2>&1 &
SERVE_PID=$!

echo "[serve] waiting up to ${WAIT}s for /health ..."
ok=0
for ((i=0; i<WAIT/2; i++)); do
  kill -0 "${SERVE_PID}" 2>/dev/null || die "serve exited during startup (see ${log})"
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { ok=1; break; }
  sleep 2
done
[[ "${ok}" == "1" ]] || die "server not ready in ${WAIT}s (see ${log})"

echo "[eval] server healthy -> running lm_eval"
"${EVAL[@]}"; rc=$?
echo "[eval] done rc=${rc}; results under ${OUT_DIR}"
exit "${rc}"