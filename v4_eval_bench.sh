#!/usr/bin/env bash
# =============================================================================
#  v4_eval_bench.sh — serve DeepSeek-V4 (offload config under test), then run
#  lm-eval over gsm8k / mmlu_pro / gpqa_diamond. Does not touch v4_eval.sh.
#
#  `vllm serve` owns decoding and thinking. lm-eval is a dumb client: it sends
#  no sampling field except `temperature`, which its API backend always writes
#  into the payload (gen_kwargs.pop("temperature", 0)) — so a request can never
#  omit it and inherit the server default. SERVE_TEMPERATURE is therefore echoed
#  into --gen_kwargs: one variable, two wires, same number. top_p and everything
#  else are omitted from the request and come from --override-generation-config.
#  Exception: where a benchmark's own yaml pins temperature (mmlu_pro), the yaml
#  wins and nothing is echoed — see the `pins_temp` column in preset().
#
#  Usage:   source <your_v4_env>.sh
#           TASKS=gsm8k,mmlu_pro,gpqa_diamond THINK=1 ./v4_eval_bench.sh
#  Preview: DRY_RUN=1 ./v4_eval_bench.sh
# =============================================================================
set -uo pipefail

# ── server ───────────────────────────────────────────────────────────────────
MODEL="${MODEL:-/home/keyi/llms/deepseekv4-w8a8}"
REMOE_GATE="${REMOE_GATE:-}"            # fine-tuned MoE gate dir/file; empty = off
SERVED_NAME="${SERVED_NAME:-glm-5}"
CARD="${CARD:-6}"
PORT="${PORT:-7001}"
TP="${TP:-1}"; DP="${DP:-1}"
SEED="${SEED:-1024}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"       # keep <= num_device_experts/topk to hold the offload path

# ── decode + thinking (serve-owned) ──────────────────────────────────────────
SERVE_TEMPERATURE="${SERVE_TEMPERATURE:-0.0}"
SERVE_TOP_P="${SERVE_TOP_P:-1.0}"
THINK="${THINK:-0}"                     # 0|1
THINK_EFFORT="${THINK_EFFORT:-}"        # ""|low|medium|high   ("" = send nothing = model default)
THINK_KEY="${THINK_KEY:-thinking}"      # chat-template kwarg: DeepSeek-V3.1/Granite="thinking",
                                        # Qwen3="enable_thinking". A key the template doesn't declare
                                        # is dropped silently -> that's what the probe below catches.

# ── MoE offload (the config whose accuracy you're measuring) ─────────────────
OFFLOAD="${OFFLOAD:-1}"
NUM_DEVICE_EXPERTS="${NUM_DEVICE_EXPERTS:-24}"
NUM_DEVICE_LAYERS="${NUM_DEVICE_LAYERS:-1}"
CACHE_POLICY="${CACHE_POLICY:-1}"       # 1 = LRC (required by prefetch)
PREFETCH="${PREFETCH:-1}"
EXPERT_PREFETCH_MAX="${EXPERT_PREFETCH_MAX:-}"
ON_DEMAND_LOAD_MAX="${ON_DEMAND_LOAD_MAX:-}"
PREDICTOR="${PREDICTOR:-fate}"
PREDICTOR_CKPT="${PREDICTOR_CKPT:-}"    # only used when PREDICTOR != fate
TIMING="${TIMING:-0}"
SEQ_STATS_NUM_SEQS="${SEQ_STATS_NUM_SEQS:-0}"
MOE_DEBUG="${MOE_DEBUG:-0}"
CPU_BIND="${CPU_BIND:-0}"

# ── eval ─────────────────────────────────────────────────────────────────────
TASKS="${TASKS:-gsm8k}"                 # gsm8k | mmlu_pro | gpqa_diamond (comma/space separated)
LIMIT="${LIMIT:-}"                      # docs per task; empty = full (1319 / 12032 / 198)
CONCURRENCY="${CONCURRENCY:-1}"
TOKENIZER="${TOKENIZER:-/home/keyi/llms/dsv4-tokenizer}"
OUT_DIR="${OUT_DIR:-./eval_results/$(date +%Y%m%d_%H%M%S)}"
PROBE="${PROBE:-1}"                     # 1 = verify thinking mode before the run, abort on mismatch
WAIT="${WAIT:-1000}"
DRY_RUN="${DRY_RUN:-0}"

# ── constants (edit here if ever needed; not worth an env knob) ──────────────
API_TIMEOUT=12000            # lm-eval timeout=; its default is far too short for a 32k-token
                             # thinking generation at concurrency 1 (lm-eval issue #3391)
MAX_RETRIES=3
MAX_LENGTH=8192              # prompt budget, THINK=0 (mmlu_pro's task README asks for >= 8192)
MAX_LENGTH_THINK=40960       # prompt budget, THINK=1
MAX_GEN_TOKS_THINK=32768     # output budget, THINK=1. Too small -> the think block never closes,
                             # `content` comes back empty and the task scores 0.
EVAL_SEED=0,1234,1234,1234   # lm-eval --seed python,numpy,torch,fewshot (= harness default; the
                             # 4th value picks the few-shot examples, so it moves scores)

# ── dataset cache (pre-seeded; no download) ──────────────────────────────────
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/keyi/llms/benchmarks/hf_cache}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export USE_MODELSCOPE_HUB=0
export VLLM_BATCH_INVARIANT=0
export ASCEND_RT_VISIBLE_DEVICES="${CARD}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"; export NO_PROXY="${no_proxy}"

# =============================================================================
die() { echo "ERROR: $*" >&2; exit 1; }

[[ "${THINK}" =~ ^[01]$ ]] || die "THINK must be 0 or 1 (got '${THINK}')"
[[ -z "${THINK_EFFORT}" || "${THINK_EFFORT}" =~ ^(low|medium|high)$ ]] || die "THINK_EFFORT must be empty|low|medium|high"

# Resolved once, used by build_serve, the banner and the probe.
GEN_CFG="{\"temperature\":${SERVE_TEMPERATURE},\"top_p\":${SERVE_TOP_P}}"
THINK_KWARGS="{\"${THINK_KEY}\":$([[ ${THINK} == 1 ]] && echo true || echo false)"
[[ -n "${THINK_EFFORT}" ]] && THINK_KWARGS="${THINK_KWARGS},\"reasoning_effort\":\"${THINK_EFFORT}\""
THINK_KWARGS="${THINK_KWARGS}}"

# reap vLLM engine-core workers that outlive serve and keep holding NPU memory
SERVE_PID=""
reap() {
  local pids; pids=$(pgrep -u "$(id -u)" -f 'VLLMEngineCor' 2>/dev/null | grep -v "^$$\$" || true)
  [[ -n "${SERVE_PID}" ]] && kill -TERM "${SERVE_PID}" 2>/dev/null
  [[ -n "${pids}" ]] && { kill -TERM ${pids} 2>/dev/null; sleep 3; kill -KILL ${pids} 2>/dev/null; }
  return 0
}

offload_json() {
  [[ "${OFFLOAD}" != "1" ]] && return
  local p="\"expert_offload\":true,\"num_device_experts\":${NUM_DEVICE_EXPERTS}"
  p="${p},\"num_device_layers\":${NUM_DEVICE_LAYERS},\"cache_policy_enabled\":$([[ ${CACHE_POLICY} == 1 ]] && echo true || echo false)"
  [[ -n "${ON_DEMAND_LOAD_MAX}" ]] && p="${p},\"on_demand_load_max\":${ON_DEMAND_LOAD_MAX}"
  [[ "${PREFETCH}" == "1" ]] && {
    p="${p},\"expert_prefetch_enabled\":true,\"expert_predictor\":\"${PREDICTOR}\""
    [[ "${PREDICTOR}" != "fate" && -n "${PREDICTOR_CKPT}" ]] && p="${p},\"expert_predictor_ckpt\":\"${PREDICTOR_CKPT}\""
    [[ -n "${EXPERT_PREFETCH_MAX}" ]] && p="${p},\"expert_prefetch_max\":${EXPERT_PREFETCH_MAX}"
  }
  p="${p},\"moe_offload_debug\":$([[ ${MOE_DEBUG} == 1 ]] && echo true || echo false)"
  p="${p},\"seq_stats_num_seqs\":${SEQ_STATS_NUM_SEQS}"
  p="${p},\"cache_profile_timing\":$([[ ${TIMING} == 1 ]] && echo true || echo false)"
  printf '{"enable_cpu_binding":%s,"expert_offload_config":{%s}}' \
         "$([[ ${CPU_BIND} == 1 ]] && echo true || echo false)" "${p}"
}

build_serve() {
  SERVE=( vllm serve "${MODEL}"
          --host 0.0.0.0 --port "${PORT}"
          --served-model-name "${SERVED_NAME}"
          --tensor-parallel-size "${TP}" --data-parallel-size "${DP}"
          --max-num-seqs "${MAX_NUM_SEQS}"
          --seed "${SEED}" --gpu-memory-utilization "${GPU_MEM_UTIL}"
          --generation-config vllm
          --override-generation-config "${GEN_CFG}"
          --default-chat-template-kwargs "${THINK_KWARGS}"
          --enforce-eager --quantization ascend --enable-expert-parallel
          --trust-remote-code
          --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4
          --enable-auto-tool-choice --reasoning-parser deepseek_v4
          --enable-chunked-prefill --enable-prefix-caching
          --aggregate-engine-logging
          --safetensors-load-strategy prefetch --api-server-count 1 )
  local addl; addl="$(offload_json)"
  if [[ -n "${REMOE_GATE}" ]]; then                 # ReMoE fine-tuned router-gate override
    local gate="\"moe_gate_override_path\":\"${REMOE_GATE}\""
    if [[ -n "${addl}" ]]; then addl="${addl%\}},${gate}}"; else addl="{${gate}}"; fi
  fi
  [[ -n "${addl}" ]] && SERVE+=( --additional-config "${addl}" )
}

# Per-benchmark settings. Everything else (endpoint, chat template, stop strings,
# max_length, seeds) is identical across the three and lives in build_eval.
#
#   task                       = lm-eval task name    fewshot = its yaml default, stated
#   max_gen_toks               = its yaml default, except gpqa (see below)
#   pins_temp = yes            -> the yaml fixes temperature, so we don't echo the serve value
preset() {
  case "$1" in
    gsm8k)        P_TASK=gsm8k;                     P_FEWSHOT=5; P_MAXGEN=256;  P_PINS_TEMP=no  ;;
    mmlu_pro)     P_TASK=mmlu_pro;                  P_FEWSHOT=5; P_MAXGEN=2048; P_PINS_TEMP=yes ;;
    gpqa_diamond) P_TASK=gpqa_diamond_cot_zeroshot; P_FEWSHOT=0; P_MAXGEN=1024; P_PINS_TEMP=no  ;;
    *) die "unknown task '$1' (valid: gsm8k mmlu_pro gpqa_diamond)" ;;
  esac
  # gsm8k: 5-shot, until=["Question:","</s>","<|im_end|>"], no temperature in the yaml, 256 = harness default.
  # mmlu_pro: 5-shot CoT, until=["Question:"], temperature: 0.0, max_gen_toks: 2048 (task v3).
  # gpqa_diamond_cot_zeroshot: 0-shot CoT, until=["</s>"], strict-match on "The answer is X" —
  #   1024 is a deliberate bump from the harness default 256, which truncates that sentence away.
  P_MAXLEN=${MAX_LENGTH}; P_UNTIL=""
  if [[ ${THINK} == 1 ]]; then
    P_MAXGEN=${MAX_GEN_TOKS_THINK}; P_MAXLEN=${MAX_LENGTH_THINK}
    P_UNTIL="[]"   # stop strings match the RAW stream: "Question:" inside a CoT would cut the trace
  fi
}

build_eval() {
  preset "$1"
  # Chat endpoint on purpose: only there does the server apply the chat template and
  # --reasoning-parser, which keeps the <think> trace out of `content` (and out of the
  # answer regex). lm-eval's own think-stripping doesn't exist for API backends.
  local gk="max_gen_toks=${P_MAXGEN}"
  [[ ${P_PINS_TEMP} == no ]] && gk="${gk},temperature=${SERVE_TEMPERATURE}"
  [[ -n ${P_UNTIL} ]] && gk="${gk},until=${P_UNTIL}"
  EVAL=( lm_eval --model local-chat-completions
         --model_args "model=${SERVED_NAME},base_url=http://127.0.0.1:${PORT}/v1/chat/completions,num_concurrent=${CONCURRENCY},max_retries=${MAX_RETRIES},tokenized_requests=False,tokenizer=${TOKENIZER},max_length=${P_MAXLEN},timeout=${API_TIMEOUT}"
         --tasks "${P_TASK}" --num_fewshot "${P_FEWSHOT}" --gen_kwargs "${gk}"
         --seed "${EVAL_SEED}" --apply_chat_template --batch_size 1
         --output_path "${OUT_DIR}/$1" --log_samples )
  [[ ${P_FEWSHOT} -gt 0 ]] && EVAL+=( --fewshot_as_multiturn )
  [[ -n ${LIMIT} ]] && EVAL+=( --limit "${LIMIT}" )
  return 0
}

# One 64-token request: THINK=1 with a key the template doesn't declare is dropped
# silently, and you'd only find out days later. Non-empty reasoning_content = thinking.
probe_thinking() {
  [[ "${PROBE}" == "1" ]] || return 0
  local resp state want
  resp=$(curl -sf -m 600 "http://127.0.0.1:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
         -d "{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2?\"}],\"max_tokens\":64}") \
    || { echo "[probe] WARN: probe request failed — skipping check" >&2; return 0; }
  state=$(printf '%s' "${resp}" | python3 -c '
import json, sys
m = json.load(sys.stdin)["choices"][0]["message"]
r = (m.get("reasoning_content") or m.get("reasoning") or "")
sys.stderr.write("[probe] reasoning=%d chars, content=%d chars\n" % (len(r), len(m.get("content") or "")))
print("on" if r.strip() else "off")') || { echo "[probe] WARN: unparsable response — skipping check" >&2; return 0; }
  want=$([[ ${THINK} == 1 ]] && echo on || echo off)
  echo "[probe] thinking observed=${state} wanted=${want}  (${THINK_KWARGS})"
  [[ "${state}" == "${want}" ]] || die "the server is not honouring ${THINK_KWARGS} —
       the template likely declares a different key: try THINK_KEY=enable_thinking, or grep it:
       python3 -c \"import json;print(json.load(open('${TOKENIZER}/tokenizer_config.json'))['chat_template'])\" | grep -o 'thinking[a-z_]*'
       (PROBE=0 to skip this check)"
  return 0
}

preflight() {
  [[ -d "${TOKENIZER}" ]] || die "TOKENIZER dir not found: ${TOKENIZER} (tokenizer files, NO config.json)"
  [[ -d "${HF_DATASETS_CACHE}" ]] || {
    echo "[!!] No dataset cache at ${HF_DATASETS_CACHE}. Seed it ONCE (with network):" >&2
    echo "     HF_DATASETS_CACHE=${HF_DATASETS_CACHE} python -c \"from datasets import load_dataset;" >&2
    echo "       load_dataset('openai/gsm8k','main'); load_dataset('TIGER-Lab/MMLU-Pro');" >&2
    echo "       load_dataset('Idavidrein/gpqa','gpqa_diamond')\"     # gpqa is GATED: export HF_TOKEN" >&2
    die "dataset cache not seeded"
  }
}

# ── run ──────────────────────────────────────────────────────────────────────
TASK_LIST="$(echo "${TASKS}" | tr ',' ' ')"
build_serve
for t in ${TASK_LIST}; do build_eval "${t}"; done   # bad task name dies before anything loads

echo "────────────────────────────────────────────────────────────"
echo "  serve : ${SERVED_NAME} on :${PORT} (card ${CARD})  offload=${OFFLOAD} prefetch=${PREFETCH} predictor=${PREDICTOR} prefetch_max=${EXPERT_PREFETCH_MAX:-none} ondemand_load_max=${ON_DEMAND_LOAD_MAX:-none}"
echo "  stats : timing=${TIMING} seq_stats_num_seqs=${SEQ_STATS_NUM_SEQS} moe_debug=${MOE_DEBUG} cpu_bind=${CPU_BIND}"
echo "  decode: ${GEN_CFG}   thinking: ${THINK_KWARGS}   (both owned by serve)"
echo "  eval  : tasks=${TASK_LIST} limit=${LIMIT:-full} concurrency=${CONCURRENCY} probe=${PROBE}"
for t in ${TASK_LIST}; do
  preset "${t}"
  echo "          - ${t}: ${P_TASK} fewshot=${P_FEWSHOT} max_gen_toks=${P_MAXGEN} max_length=${P_MAXLEN} until=${P_UNTIL:-task-default} temperature=$([[ ${P_PINS_TEMP} == yes ]] && echo "task-yaml (pinned)" || echo "${SERVE_TEMPERATURE} (from serve)")"
done
echo "  cache : ${HF_DATASETS_CACHE}   out=${OUT_DIR}"
echo "────────────────────────────────────────────────────────────"
printf '+ %q ' "${SERVE[@]}"; echo
for t in ${TASK_LIST}; do build_eval "${t}"; printf '+ %q ' "${EVAL[@]}"; echo; done
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
probe_thinking

rc=0
for t in ${TASK_LIST}; do
  build_eval "${t}"
  echo "[eval] === ${t}: ${P_TASK} (think=${THINK}) ==="
  "${EVAL[@]}"; trc=$?
  echo "[eval] ${t} rc=${trc} -> ${OUT_DIR}/${t}"
  [[ ${trc} -ne 0 ]] && rc=${trc}
done
echo "[eval] done rc=${rc}; results under ${OUT_DIR}"
exit "${rc}"
