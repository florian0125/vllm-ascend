#!/usr/bin/env bash
# =============================================================================
#  v4_eval_bench.sh — serve DeepSeek-V4-Flash (the offload config under test),
#  then run lm-eval on gsm8k / mmlu_pro / gpqa_diamond.
#
#  TASKS is the only switch. Each preset carries its own documented protocol —
#  endpoint, reasoning mode, shots, budgets — and prints the number the vendor
#  published for that exact configuration, so every run has a target.
#  Number of samples used for each task for fast evaluation (non-thinking across all)
#  1. gsm8k 300 samples
#  2. gpqa_diamand, LIMIT not set, use all 198 samples
#  3. mmlu_pro 300 samples
#
#    source <your_v4_env>.sh
#    TASKS=gsm8k ./v4_eval_bench.sh                    # Non-think, completions
#    TASKS=mmlu_pro,gpqa_diamond ./v4_eval_bench.sh    # Think High, chat
#    TASKS=gsm8k,mmlu_pro ./v4_eval_bench.sh           # mixed: one server, both protocols
#    TASKS=gpqa_diamond THINK_EFFORT=max ./v4_eval_bench.sh
#    DRY_RUN=1 TASKS=gsm8k ./v4_eval_bench.sh          # print the plan, launch nothing
#
#  Every knob below is env-overridable. One run = one ${OUT_DIR} containing
#  run.log (everything), serve.log (server only), and per-task results + metrics.
#
#  Tags ([D] [R] [N] [P] [Y] [!]) point at APPENDIX A, which carries every source.
# =============================================================================

set -uo pipefail

# ── server ───────────────────────────────────────────────────────────────────
MODEL="${MODEL:-/mnt/nvme0n1_data/DeepSeek-V4-Flash-W8A8}"
REMOE_GATE="${REMOE_GATE:-}"            # ReMoE fine-tuned router-gate override
SERVED_NAME="${SERVED_NAME:-deepseekv4-flash-w8a8}"     # served id only; not the model name
CARD="${CARD:-6}"
PORT="${PORT:-7001}"
TP="${TP:-1}"; DP="${DP:-1}"
SEED="${SEED:-1024}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"       # keep <= num_device_experts/topk to hold the offload path

# ── MoE offload ──────────────────────────────────────────────────────────────
OFFLOAD="${OFFLOAD:-1}"
NUM_DEVICE_EXPERTS="${NUM_DEVICE_EXPERTS:-24}"
NUM_DEVICE_LAYERS="${NUM_DEVICE_LAYERS:-1}"
CACHE_POLICY="${CACHE_POLICY:-1}"       # 1 = LRC (required by prefetch)
PREFETCH="${PREFETCH:-1}"
EXPERT_PREFETCH_MAX="${EXPERT_PREFETCH_MAX:-}"
ON_DEMAND_LOAD_MAX="${ON_DEMAND_LOAD_MAX:-}"
PREDICTOR="${PREDICTOR:-fate}"
PREDICTOR_CKPT="${PREDICTOR_CKPT:-}"    # only used when PREDICTOR != fate
EXPERT_SUBSTITUTION="${EXPERT_SUBSTITUTION:-1}"
EXPERT_SUBSTITUTION_THRESHOLD="${EXPERT_SUBSTITUTION_THRESHOLD:-0.25}"
TIMING="${TIMING:-0}"
SEQ_STATS_NUM_SEQS="${SEQ_STATS_NUM_SEQS:-0}"
MOE_DEBUG="${MOE_DEBUG:-0}"
CPU_BIND="${CPU_BIND:-0}"

# ── eval ─────────────────────────────────────────────────────────────────────
TASKS="${TASKS:-gsm8k}"                 # gsm8k | mmlu_pro | gpqa_diamond
THINK="${THINK:-}"                      # ""=preset default, 0=Non-think, 1=Think
THINK_EFFORT="${THINK_EFFORT:-}"        # ""=preset default, high|max [D]
LIMIT="${LIMIT:-}"                      # docs per LEAF task; empty=full; <1.0=fraction (A.7)
CONCURRENCY="${CONCURRENCY:-1}"         # pinned to 1 by MAX_NUM_SEQS=1
TOKENIZER="${TOKENIZER:-/mnt/nvme1n1_data/TRT_HeteroCompute/keyi/benchmarks/dsv4-tokenizer}"
EOS_STRING="${EOS_STRING:-}"            # empty = derive from TOKENIZER (A.8)
OUT_DIR="${OUT_DIR:-./eval_results/$(date +%Y%m%d_%H%M%S)}"
PROBE="${PROBE:-1}"                     # verify the served reasoning mode before running
WAIT="${WAIT:-1000}"                    # seconds to wait for /health
DRY_RUN="${DRY_RUN:-0}"

VERBOSE="${VERBOSE:-0}"                 # 1 = also mirror lm_eval's stdout (results/INFO) to the terminal; serve + progress are already live
RUN_LOG="${OUT_DIR}/run.log"            # config echo, plan, per-task results, metrics
SERVE_LOG="${OUT_DIR}/serve.log"        # full server output
EVAL_LOG_DIR="${OUT_DIR}"               # per-task lm_eval output -> <task>/eval.log

# ── constants (chat path only; completions sends no gen_kwargs) ──────────────
THINK_KEY="${THINK_KEY:-thinking}"   # [D][R][N]
MAX_RETRIES=10                       # [R]
API_TIMEOUT=60000                    # [R]
THINK_MAX_GEN_TOKS=32768             # [N]
NONTHINK_MAX_GEN_TOKS=16384
CHAT_MAX_LENGTH=32768
CHAT_TEMPERATURE=0.6                 # [D]
CHAT_TOP_P=0.95                      # [N]
THINK_DROP_STOPS=1                   # [!] see A.5

# ── environment (dataset cache pre-seeded; no download) ──────────────────────
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/nvme1n1_data/TRT_HeteroCompute/keyi/benchmarks/hf_cache}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export USE_MODELSCOPE_HUB=0
export VLLM_BATCH_INVARIANT=0
export ASCEND_RT_VISIBLE_DEVICES="${CARD}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"; export NO_PROXY="${no_proxy}"
export LMEVAL_LOG_LEVEL="${LMEVAL_LOG_LEVEL:-WARNING}"

# =============================================================================
#  helpers
# =============================================================================
die() { echo "ERROR: $*" >&2; exit 1; }

[[ -z "${THINK}" || "${THINK}" =~ ^[01]$ ]] || die "THINK must be empty (use the preset), 0 or 1"
[[ -z "${THINK_EFFORT}" || "${THINK_EFFORT}" =~ ^(high|max)$ ]] \
  || die "THINK_EFFORT must be empty, high or max — [D] documents exactly three modes"
[[ -z "${LIMIT}" || "${LIMIT}" =~ ^[1-9][0-9]*$ || "${LIMIT}" =~ ^0?\.[0-9]*[1-9][0-9]*$ ]] \
  || die "LIMIT must be empty, a positive integer (docs per LEAF task), or a decimal in (0,1)
       meaning a fraction of each task — lm-eval reads limit < 1.0 as a percentage. Got '${LIMIT}'."

# Reproduction line for the log: env assignments are not in \$@, so reconstruct them.
repro_line() {
  local v out=""
  for v in CARD PORT TASKS THINK THINK_EFFORT LIMIT CONCURRENCY MAX_NUM_SEQS SEED \
           GPU_MEM_UTIL OFFLOAD NUM_DEVICE_EXPERTS NUM_DEVICE_LAYERS CACHE_POLICY \
           PREFETCH EXPERT_PREFETCH_MAX ON_DEMAND_LOAD_MAX PREDICTOR TIMING \
           SEQ_STATS_NUM_SEQS MOE_DEBUG CPU_BIND PROBE MODEL SERVED_NAME REMOE_GATE; do
    out+="${v}=$(printf '%q' "${!v}") "
  done
  echo "${out}$0"
}

# ── presets ──────────────────────────────────────────────────────────────────
# The mode follows the table the vendor published the benchmark in, not taste.
# Reasoning and numbers in A.2. An empty field means "the task yaml documents it,
# so we don't pass the flag" [Y] — note "" and 0 are NOT the same thing.
#   P_ENDPOINT  completions = raw prompt, no chat template, thinking unreachable
#               chat        = server applies the chat template + --reasoning-parser
#   P_MAXGEN    max_gen_toks for the non-thinking path (thinking uses THINK_MAX_GEN_TOKS)
#   P_REF       the vendor's published number for exactly this configuration
preset() {
  P_FORCED_CHAT=0
  case "$1" in
  gsm8k)          # [D] base-only benchmark; [R] validated instruct run reproduced
    P_TASK=gsm8k;                           P_ENDPOINT=completions; P_THINK=0; P_EFFORT=""
    P_FEWSHOT="";    P_MAXGEN=2048;         P_MAXLEN=""             # [R] 8-shot/0-shot by HQ
    P_REF="[R] instruct non-think 8-shot: strict 0.9431 / flexible 0.9439  |  [D] base 8-shot EM 90.8"
    ;;
  mmlu_pro)       # [D] High 86.4 is the headline and beats Max 86.2
    P_TASK=mmlu_pro;    P_ENDPOINT=chat;        P_THINK=1; P_EFFORT=high
    P_FEWSHOT="";       P_MAXGEN=${NONTHINK_MAX_GEN_TOKS};           P_MAXLEN=${CHAT_MAX_LENGTH}             # [Y][P] yaml 5-shot
    P_REF="[D] V4-Flash MMLU-Pro EM: non-think 83.0 | high 86.4 | max 86.2"
    ;;
  gpqa_diamond)   # [D] thinking is worth +16 here; Max 88.1 wants --max-model-len >= 393216
    P_TASK=gpqa_diamond_generative_n_shot; P_ENDPOINT=chat;        P_THINK=1; P_EFFORT=high
    P_FEWSHOT=0;    P_MAXGEN=${NONTHINK_MAX_GEN_TOKS};             P_MAXLEN=${CHAT_MAX_LENGTH}               # [P]
    P_REF="[D] V4-Flash GPQA-D Pass@1: non-think 71.2 | high 87.4 | max 88.1"
    ;;
  *) die "unknown task '$1' (valid: gsm8k mmlu_pro gpqa_diamond)" ;;
  esac

  [[ -n "${THINK}" ]] && P_THINK="${THINK}"
  [[ -n "${THINK_EFFORT}" ]] && P_EFFORT="${THINK_EFFORT}"
  if [[ "${P_THINK}" == "1" ]]; then
    [[ -z "${P_EFFORT}" ]] && P_EFFORT=high     # thinking with no effort is not a mode [D]
    # Thinking lives behind the chat template; unreachable on /v1/completions — A.4.
    [[ "${P_ENDPOINT}" == "completions" ]] && { P_ENDPOINT=chat; P_FORCED_CHAT=1; }
  else
    P_EFFORT=""
  fi
  return 0
}

# One server carries one --default-chat-template-kwargs, so every chat task in a run
# must agree on the mode. Completions tasks never touch it and can share the run.
# Sets RUN_KWARGS (empty => this run has no chat task at all).
resolve_run_mode() {
  RUN_KWARGS=""; local t sig="" first="" first_task=""
  for t in ${TASK_LIST}; do
    preset "${t}"
    [[ "${P_ENDPOINT}" == "chat" ]] || continue
    sig="${P_THINK}:${P_EFFORT}"
    if [[ -z "${first}" ]]; then first="${sig}"; first_task="${t}"
    elif [[ "${sig}" != "${first}" ]]; then
      die "chat tasks want different reasoning modes (${first_task}=${first}, ${t}=${sig}).
       One server carries one --default-chat-template-kwargs. Run them separately, or
       force one mode with THINK=/THINK_EFFORT=."
    fi
  done
  [[ -z "${first}" ]] && return 0
  if [[ "${first%%:*}" == "1" ]]; then
    RUN_KWARGS="{\"${THINK_KEY}\":true,\"reasoning_effort\":\"${first#*:}\"}"   # [R]
  else
    RUN_KWARGS="{\"${THINK_KEY}\":false}"                                       # [D]
  fi
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
  [[ "${EXPERT_SUBSTITUTION}" == "1" ]] && {
    p="${p},\"expert_substitution_enabled\":true,\"expert_substitution_threshold\":${EXPERT_SUBSTITUTION_THRESHOLD}"
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
          --override-generation-config '{"temperature":0.0,"top_p":1.0}'
          --enforce-eager --quantization ascend --enable-expert-parallel
          --trust-remote-code
          --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4
          --enable-auto-tool-choice --reasoning-parser deepseek_v4
          --enable-chunked-prefill --enable-prefix-caching
          --aggregate-engine-logging
          --safetensors-load-strategy prefetch --api-server-count 1 )
  # No --generation-config / --override-generation-config on purpose — A.3.
  [[ -n "${RUN_KWARGS}" ]] && SERVE+=( --default-chat-template-kwargs "${RUN_KWARGS}" )
  local addl; addl="$(offload_json)"
  if [[ -n "${REMOE_GATE}" ]]; then
    local gate="\"moe_gate_override_path\":\"${REMOE_GATE}\""
    if [[ -n "${addl}" ]]; then addl="${addl%\}},${gate}}"; else addl="{${gate}}"; fi
  fi
  [[ -n "${addl}" ]] && SERVE+=( --additional-config "${addl}" )
}

# Sets the EVAL array for one preset. Call preset() implicitly; P_* are globals.
build_eval() {
  preset "$1"
  local margs gk=""
  margs="model=${SERVED_NAME},num_concurrent=${CONCURRENCY},max_retries=${MAX_RETRIES}"
  margs="${margs},timeout=${API_TIMEOUT},tokenized_requests=False,tokenizer=${TOKENIZER}"
  [[ -n ${P_MAXLEN} ]] && margs="${margs},max_length=${P_MAXLEN}"

  if [[ "${P_ENDPOINT}" == "chat" ]]; then
    # chat endpoint only: --reasoning-parser splits the trace out (A.4).
    # tokenizer_backend=huggingface so lm-eval derives the EOS stop string (A.8).
    margs="${margs},tokenizer_backend=huggingface,trust_remote_code=True"
    [[ -n "${EOS_STRING}" ]] && margs="${margs},eos_string=${EOS_STRING}"

    EVAL=( lm_eval --model local-chat-completions
           --model_args "${margs},base_url=http://127.0.0.1:${PORT}/v1/chat/completions"
           --apply_chat_template )
    gk="temperature=${CHAT_TEMPERATURE},top_p=${CHAT_TOP_P}"       # [D]
    if [[ "${P_THINK}" == "1" ]]; then
      gk="${gk},max_gen_toks=${THINK_MAX_GEN_TOKS}"                # [N]
      [[ "${THINK_DROP_STOPS}" == "1" ]] && gk="${gk},until=[]"    # [!] A.5
    elif [[ -n ${P_MAXGEN} ]]; then
      gk="${gk},max_gen_toks=${P_MAXGEN}"
    fi
  else
    # [R] path: raw completions, no chat template, no gen_kwargs.
    [[ -n ${P_MAXGEN} ]] && margs="${margs},max_gen_toks=${P_MAXGEN}"
    EVAL=( lm_eval --model local-completions
           --model_args "${margs},base_url=http://127.0.0.1:${PORT}/v1/completions" )
  fi
  EVAL+=( --tasks "${P_TASK}" --batch_size 1 --output_path "${OUT_DIR}/$1" --log_samples )
  [[ -n ${P_FEWSHOT} ]] && EVAL+=( --num_fewshot "${P_FEWSHOT}" )
  [[ -n ${gk} ]] && EVAL+=( --gen_kwargs "${gk}" )
  [[ -n ${LIMIT} ]] && EVAL+=( --limit "${LIMIT}" )
  return 0
}

describe_task() {   # expects preset()/build_eval() to have run for "$1"
  local mode budget forced=""
  if [[ "${P_THINK}" == "1" ]]; then mode="Think ${P_EFFORT}"; budget="${THINK_MAX_GEN_TOKS} [N]"
  else                               mode="Non-think";         budget="${P_MAXGEN:-task-yaml}"; fi
  [[ "${P_FORCED_CHAT}" == "1" ]] && forced="  (forced to chat: thinking is unreachable on completions)"
  echo "     - $1 -> ${P_TASK}"
  echo "         mode=${mode}  endpoint=${P_ENDPOINT}${forced}"
  echo "         fewshot=${P_FEWSHOT:-task-yaml}  max_gen_toks=${budget}  max_length=${P_MAXLEN:-lm-eval-default}"
  echo "         target: ${P_REF}"
}

# ── metrics ──────────────────────────────────────────────────────────────────
# Sum a Prometheus metric across all label combinations.
metric_value() {
  awk -v metric="$2" '
    $1 == metric || index($1, metric "{") == 1 { total += $2 }
    END { print total + 0 }
  ' "$1"
}

# Which TPOT histogram does this build expose? Names are not stable across versions.
#   request_time_per_output_token_seconds  request-level, ONE observation per request
#   inter_token_latency_seconds            per-interval, one per token after the first
#   time_per_output_token_seconds          older v0-era name
resolve_tpot_metric() {
  local m
  for m in vllm:request_time_per_output_token_seconds \
           vllm:inter_token_latency_seconds \
           vllm:time_per_output_token_seconds; do
    grep -q "^${m}_count" "$1" && { printf '%s' "${m}"; return 0; }
  done
  return 1
}

# Non-fatal by design: lm_eval results are already on disk by the time these run.
fetch_metrics() {
  curl -sf -m 30 "http://127.0.0.1:${PORT}/metrics" > "$1" && return 0
  echo "[metrics] WARN: could not read /metrics ${2:-} (port ${PORT})" >&2
  : > "$1"          # readable empty file so metric_value() still returns 0
  return 1
}

report_tpot() {     # reads METRICS_OK / METRICS_BEFORE / METRICS_AFTER for this task
  local label="$1"
  [[ "${METRICS_OK}" == "1" ]] || { echo "[metrics] ${label}: skipped, no baseline." >&2; return 0; }
  kill -0 "${SERVE_PID}" 2>/dev/null \
    || { echo "[metrics] ${label}: skipped, server (pid ${SERVE_PID}) is gone." >&2; return 0; }
  fetch_metrics "${METRICS_AFTER}" "after ${label}" || return 0

  local metric unit   # declared separately: `local m="$(cmd)"` masks the exit status

  metric="$(resolve_tpot_metric "${METRICS_AFTER}")" || metric=""
  if [[ -z "${metric}" ]]; then
    echo "[metrics] ${label}: no known TPOT histogram. This build exposes:" >&2
    grep -E '^# TYPE vllm:.* histogram' "${METRICS_AFTER}" | awk '{print "           " $3}' >&2
    return 0
  fi
  [[ "${metric}" == "vllm:request_time_per_output_token_seconds" ]] \
    && unit="Requests" || unit="Inter-token intervals"

  awk -v bs="$(metric_value "${METRICS_BEFORE}" "${metric}_sum")" \
      -v as="$(metric_value "${METRICS_AFTER}"  "${metric}_sum")" \
      -v bc="$(metric_value "${METRICS_BEFORE}" "${metric}_count")" \
      -v ac="$(metric_value "${METRICS_AFTER}"  "${metric}_count")" \
      -v metric="${metric}" -v unit="${unit}" -v task="${label}" '
  BEGIN {
    d_sum = as - bs; d_cnt = ac - bc
    if (d_cnt <= 0) { print "[metrics] " task ": " metric " recorded no new observations."; exit }
    printf "────────────────────────────────────────────────────────────\n"
    printf "[metrics] %s — Mean TPOT: %.2f ms/token   (source: %s)\n", task, (d_sum/d_cnt)*1000, metric
    printf "[metrics] %s measured: %d\n", unit, d_cnt
    printf "────────────────────────────────────────────────────────────\n"
  }'
  return 0
}

# Pull the scores out of the results_*.json lm_eval wrote under --output_path,
# so the console shows numbers even though lm_eval's own log went to a file.
report_results() {
  local task="$1" trc="$2" eval_log="$3"
  if [[ "${trc}" -ne 0 ]]; then
    echo "[result] ${task}: lm_eval FAILED rc=${trc} — see ${eval_log}" >&2
    return 0
  fi
  python3 - "${OUT_DIR}/${task}" "${task}" <<'PY'
import glob, json, os, sys
out_dir, task = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(out_dir, "**", "results_*.json"), recursive=True))
if not files:
    print("[result] %s: no results json found" % task); sys.exit(0)
res = json.load(open(files[-1])).get("results", {})
for tname, m in res.items():
    for k in sorted(m):
        if k.startswith("alias") or "_stderr" in k or not isinstance(m[k], (int, float)):
            continue
        se = m.get(k.replace(",", "_stderr,", 1), "")
        se = " ± %.4f" % se if isinstance(se, (int, float)) else ""
        print("[result] %s  %s = %.4f%s" % (tname, k, m[k], se))
PY
  return 0
}

# ── run-phase helpers ────────────────────────────────────────────────────────
SERVE_PID=""
reap() {   # reap engine-core workers that outlive serve and keep holding NPU memory
  local pids; pids=$(pgrep -u "$(id -u)" -f 'VLLMEngineCor' 2>/dev/null | grep -v "^$$\$" || true)
  [[ -n "${SERVE_PID}" ]] && kill -TERM "${SERVE_PID}" 2>/dev/null
  [[ -n "${pids}" ]] && { kill -TERM ${pids} 2>/dev/null; sleep 3; kill -KILL ${pids} 2>/dev/null; }
  return 0
}

preflight() {
  [[ -d "${TOKENIZER}" ]] || die "TOKENIZER dir not found: ${TOKENIZER}"
  # chat path derives its EOS from this tokenizer; prove it loads before the model does.
  if [[ -n "${RUN_KWARGS}" && -z "${EOS_STRING}" ]]; then
    local eos
    eos="$(python3 - "${TOKENIZER}" <<'PY'
import sys
from transformers import AutoTokenizer
try:
    tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
except Exception as e:
    sys.stderr.write("  %s: %s\n" % (type(e).__name__, e)); sys.exit(2)
print(tok.eos_token or "")
PY
)" || die "AutoTokenizer could not load ${TOKENIZER}. The chat path needs it for the EOS
       stop string. Fix the tokenizer dir, or set EOS_STRING=... to bypass tokenizer loading."
    [[ -n "${eos}" ]] || die "${TOKENIZER} loads but defines no eos_token, so lm-eval would send
       no EOS stop string. Set EOS_STRING=... explicitly."
    echo "[preflight] chat EOS from tokenizer: ${eos}"
  fi
  [[ -d "${HF_DATASETS_CACHE}" ]] || {
    echo "[!!] No dataset cache at ${HF_DATASETS_CACHE}. Seed it ONCE (with network):" >&2
    echo "     HF_DATASETS_CACHE=${HF_DATASETS_CACHE} python -c \"from datasets import load_dataset;" >&2
    echo "       load_dataset('openai/gsm8k','main'); load_dataset('TIGER-Lab/MMLU-Pro');" >&2
    echo "       load_dataset('Idavidrein/gpqa','gpqa_diamond')\"     # gpqa is GATED: export HF_TOKEN" >&2
    die "dataset cache not seeded"
  }
}

# A chat-template kwarg the template doesn't declare is dropped silently, so "Think High"
# can be a lie you discover days later. One 64-token request settles it.
probe_thinking() {
  [[ "${PROBE}" == "1" && -n "${RUN_KWARGS}" ]] || return 0
  local resp think want
  want=$([[ "${RUN_KWARGS}" == *true* ]] && echo on || echo off)
  resp=$(curl -sf -m 600 "http://127.0.0.1:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
         -d "{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2?\"}],\"max_tokens\":64}") \
    || { echo "[probe] WARN: request failed — skipping check" >&2; return 0; }
  think=$(printf '%s' "${resp}" | python3 -c '
import json, sys
m = json.load(sys.stdin)["choices"][0]["message"]
r = (m.get("reasoning_content") or m.get("reasoning") or "")
sys.stderr.write("[probe] reasoning=%d chars, content=%d chars\n" % (len(r), len(m.get("content") or "")))
print("on" if r.strip() else "off")') || { echo "[probe] WARN: unparsable response" >&2; return 0; }
  echo "[probe] thinking=${think}  wanted=${want}  ${RUN_KWARGS}"
  [[ "${think}" == "${want}" ]] || die "the server is not honouring ${RUN_KWARGS}.
       The template may declare a different key (try THINK_KEY=enable_thinking), or this build
       may not support --default-chat-template-kwargs. PROBE=0 to run anyway."
  return 0
}

# =============================================================================
#  run
# =============================================================================
TASK_LIST="$(echo "${TASKS}" | tr ',' ' ')"
for t in ${TASK_LIST}; do preset "${t}"; done   # bad task name dies before anything else
resolve_run_mode
build_serve

# Terminal shows everything live for monitoring: config echo, server startup, the
# lm_eval progress bar, and per-task results. The log files stay lean and durable —
#   run.log     : config echo, plan, results, metrics (no server/progress noise)
#   serve.log   : full server output, incl. the offload cache summary at shutdown
#   <task>/eval.log : that task's full lm_eval output
# fd3 is the real terminal, saved before the redirect so live streams bypass run.log.
exec 3>&2
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${OUT_DIR}"
  for t in ${TASK_LIST}; do mkdir -p "${OUT_DIR}/${t}"; done
  exec > >(tee -a "${RUN_LOG}") 2>&1
fi

echo "────────────────────────────────────────────────────────────"
echo "  started : $(date '+%F %T')   host $(hostname)   pid $$"
echo "  repro   : $(repro_line)"
echo "  serve   : ${SERVED_NAME} on :${PORT} (card ${CARD})  offload=${OFFLOAD} prefetch=${PREFETCH}"
echo "            predictor=${PREDICTOR} prefetch_max=${EXPERT_PREFETCH_MAX:-none} ondemand_load_max=${ON_DEMAND_LOAD_MAX:-none}"
echo "  stats   : timing=${TIMING} seq_stats_num_seqs=${SEQ_STATS_NUM_SEQS} moe_debug=${MOE_DEBUG} cpu_bind=${CPU_BIND}"
echo "  chat    : ${RUN_KWARGS:-not used (completions-only run)}"
echo "            sampling temperature=${CHAT_TEMPERATURE} top_p=${CHAT_TOP_P} (chat path only)"
echo "  eval    : tasks=${TASK_LIST} limit=${LIMIT:-full} concurrency=${CONCURRENCY} retries=${MAX_RETRIES} timeout=${API_TIMEOUT}"
for t in ${TASK_LIST}; do
  build_eval "${t}"
  describe_task "${t}"
  printf '         $ '; printf '%q ' "${EVAL[@]}"; echo
done
echo "  cache   : ${HF_DATASETS_CACHE}"
echo "  out     : ${OUT_DIR}   (run.log, serve.log, <task>/)"
echo "────────────────────────────────────────────────────────────"
printf '  $ '; printf '%q ' "${SERVE[@]}"; echo
echo "────────────────────────────────────────────────────────────"
[[ "${DRY_RUN}" == "1" ]] && { echo "[DRY_RUN] nothing launched."; exit 0; }

preflight
trap reap EXIT

# Server output streams live to the terminal (weight loading, engine init, per-request
# activity) so you can watch startup and progress. It goes to serve.log too — the
# expert-offload cache summary the engine prints at shutdown is a key result and would
# otherwise be lost to scrollback — but NOT to run.log, which stays a clean summary.
echo "[serve] launching (live below; full copy -> ${SERVE_LOG})"
"${SERVE[@]}" > >(tee -a "${SERVE_LOG}" >&3) 2>&1 &
SERVE_PID=$!

echo "[serve] waiting up to ${WAIT}s for /health ..."
ok=0
for ((i=0; i<WAIT/2; i++)); do
  kill -0 "${SERVE_PID}" 2>/dev/null || die "serve exited during startup (see ${SERVE_LOG})"
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { ok=1; break; }
  sleep 2
done
[[ "${ok}" == "1" ]] || die "server not ready in ${WAIT}s (see ${SERVE_LOG})"
probe_thinking

rc=0
for t in ${TASK_LIST}; do
  build_eval "${t}"
  METRICS_BEFORE="${OUT_DIR}/${t}/metrics_before.prom"
  METRICS_AFTER="${OUT_DIR}/${t}/metrics_after.prom"
  METRICS_OK=0
  fetch_metrics "${METRICS_BEFORE}" "before ${t}" && METRICS_OK=1

  echo "[eval] ${t}: ${P_TASK}  $([[ ${P_THINK} == 1 ]] && echo "Think ${P_EFFORT}" || echo "Non-think")  target: ${P_REF}"
  # stdout (results table) -> eval.log; stderr (tqdm progress + warnings) -> the terminal
  # live AND appended to eval.log. Numbers are parsed from results_*.json below, so the
  # console still shows them regardless. VERBOSE=1 additionally streams stdout to console.
  eval_log="${OUT_DIR}/${t}/eval.log"
  if [[ "${VERBOSE}" == "1" ]]; then
    "${EVAL[@]}" > >(tee -a "${eval_log}" >&3) 2> >(tee -a "${eval_log}" >&3); trc=$?
  else
    "${EVAL[@]}" > "${eval_log}" 2> >(tee -a "${eval_log}" >&3); trc=$?
  fi
  [[ ${trc} -ne 0 ]] && rc=${trc}
  report_results "${t}" "${trc}" "${eval_log}"

  report_tpot "${t}"
done
echo "────────────────────────────────────────────────────────────"
echo "[done] rc=${rc}   results + logs under ${OUT_DIR}"

exit "${rc}"

# =============================================================================
#  APPENDIX A — sources for every value above. (Nothing below executes.)
# =============================================================================
#
# A.1  SOURCES
# -----------------------------------------------------------------------------
#  [D] DeepSeek-V4-Flash model card
#      https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
#      - Three reasoning modes, with response formats:
#          Non-think   fast, intuitive                      -> `</think>` summary
#          Think High  conscious logical analysis           -> `<think>` … `</think>` summary
#          Think Max   reasoning pushed to its fullest      -> special system prompt + `<think>` …
#      - "For local deployment, we recommend setting the sampling parameters to
#         temperature = 1.0, top_p = 1.0."
#      - "For the Think Max reasoning mode, we recommend setting the context window to at
#         least 384K tokens."
#      - "This release does not include a Jinja-format chat template." It ships an
#         `encoding/` folder instead (encode_messages(messages, thinking_mode=...)), which
#         is why vLLM needs --tokenizer-mode deepseek_v4.
#  [R] vLLM recipe for DeepSeek-V4-Flash
#      https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash
#      - "For DeepSeek-V4, keep reasoning controls in chat_template_kwargs, as it exposes a
#         custom Think Max mode via "reasoning_effort": "max"."
#           Think High -> {"thinking": True, "reasoning_effort": "high"}
#           Think Max  -> {"thinking": True, "reasoning_effort": "max"}
#      - "Think Max … requires --max-model-len >= 393216 (384K tokens) to avoid truncation."
#      - VALIDATED gsm8k eval (MI355X section), which the gsm8k preset reproduces:
#           lm_eval --model local-completions \
#             --model_args model=$MODEL,base_url=…/v1/completions,num_concurrent=128,
#                          max_retries=10,max_gen_toks=2048,timeout=60000 \
#             --batch_size auto --tasks gsm8k --num_fewshot 8
#           gen_kwargs: ({})   limit: None   num_fewshot: 8
#           |gsm8k|3|flexible-extract|8|exact_match|0.9439|±0.0063|
#           |     | |strict-match    |8|exact_match|0.9431|±0.0064|
#  [N] NVIDIA build page, deepseek-v4-flash
#      https://build.nvidia.com/deepseek-ai/deepseek-v4-flash
#      - Reference client, Think High: temperature=1, top_p=0.95, max_tokens=16384,
#        extra_body={"chat_template_kwargs":{"thinking":True,"reasoning_effort":"high"}}
#      - The only published output budget for Think High -> THINK_MAX_GEN_TOKS=16384.
#  [P] ShopX, "A Foundation Model for Intent-to-Item Fulfillment in Agentic Shopping",
#      arXiv 2606.31693, Table B.6 (lm-eval task choices for general-capability rows):
#        MMLU-Pro      -> task mmlu_pro,                       5 shots
#        GPQA-Diamond  -> task gpqa_diamond_generative_n_shot, 0 shots
#        GSM8K Flex    -> task gsm8k_cot,                      8 shots, flexible extraction
#  [Y] The task's own lm-eval yaml. Never restated in this script: when the harness
#      documents a value, the flag is simply not passed, so the yaml applies.
#        gsm8k     5-shot; doc_to_text "Question: {{question}}\nAnswer:";
#                  until ["Question:","</s>","<|im_end|>"]; do_sample false; temperature 0.0;
#                  no max_gen_toks -> harness default 256
#                  strict-match  regex "#### (\-?[0-9\.\,]+)"   <- needs the few-shot format
#                  flexible      regex "(-?[$0-9.,]{2,})|(-?[0-9]+)", group_select -1
#        mmlu_pro  GROUP of 14 per-subject subtasks (12,032 docs total); 5-shot CoT
#                  (fewshot_config sampler first_n); until ["Question:"]; do_sample false;
#                  temperature 0.0; max_gen_toks 2048; filter "custom-extract"
#        gpqa_*    0-shot generative; do_sample false
#
# A.2  WHY EACH PRESET RUNS THE MODE IT RUNS
# -----------------------------------------------------------------------------
#    benchmark            Non-think   Think High   Think Max   published under
#    -------------------  ----------  -----------  ----------  ----------------------------
#    MMLU-Pro (EM)            83.0       86.4         86.2     instruct, per-mode comparison
#    GPQA Diamond (Pass@1)    71.2       87.4         88.1     instruct, per-mode comparison
#    GSM8K (EM, 8-shot)         —          —            —      BASE model table only (90.8)
#
#  gsm8k        -> Non-think + completions.
#                  GSM8K appears in no instruct or per-mode table for this model; [D] reports
#                  it only for the base checkpoint at 8-shot. [R]'s validated run follows that
#                  same base-style protocol on the instruct checkpoint (raw completions, no
#                  chat template, empty gen_kwargs) and lands at 0.9431/0.9439. Nobody
#                  documents gsm8k with thinking on for V4 — it is a saturated arithmetic
#                  benchmark, and the model card treats it as a pretraining signal.
#                  Cheapest sanity check on the offload path: it has a published target.
#  mmlu_pro     -> Think High + chat.
#                  86.4 (High) is the headline and beats Max (86.2) — the extra effort buys
#                  nothing here, so High is the best config, not a compromise. 5-shot from
#                  its yaml [Y], matching [P].
#  gpqa_diamond -> Think High + chat.
#                  Thinking is worth +16.2 points (71.2 -> 87.4), so Non-think is not a
#                  meaningful configuration here. Max (88.1) is the card's headline but sits
#                  0.7 above High while wanting --max-model-len >= 393216 [R] and a far larger
#                  budget across 198 questions; THINK_EFFORT=max when you want it. Task name
#                  and 0-shot from [P]. (The checkpoint's own max_model_len is 1,048,576, so
#                  the 384K requirement is already satisfied without passing the flag.)
#
# A.3  SAMPLING — WHY IT SPLITS BY PATH
# -----------------------------------------------------------------------------
#  lm-eval's API backends always write `temperature` into the payload
#  (gen_kwargs.pop("temperature", 0)), so a request can never omit it and inherit a server
#  default. Serve-side --override-generation-config therefore cannot control the eval, and
#  the honest options are: send an explicit value, or let lm-eval quietly send 0.
#    chat path (mmlu_pro, gpqa)  -> temperature=1.0 [D], top_p=0.95 [N]. (DeepSeek's hosted
#        API ignores temperature/top_p in thinking mode entirely.) Expect Pass@1 variance at
#        198 questions.
#    completions path (gsm8k)    -> no gen_kwargs at all, exactly as [R] ran it, so lm-eval's
#        own temperature=0 applies. That is also what the task yaml documents (do_sample:
#        false), and it is the only way to reproduce 0.9431/0.9439.
#  Neither --generation-config nor --override-generation-config is passed: neither appears in
#  [R]'s serve command, and vLLM's default (--generation-config auto) reads the checkpoint's
#  own generation_config.json.
#
# A.4  WHY THINKING FORCES THE CHAT ENDPOINT
# -----------------------------------------------------------------------------
#  --reasoning-parser deepseek_v4 does not enable thinking; it teaches vLLM how to SPLIT the
#  trace out into message.reasoning_content, and it only runs on /v1/chat/completions.
#  --tokenizer-mode deepseek_v4 is the custom message encoder (needed because there is no
#  Jinja template), and --tool-call-parser is for tool calls. The switch itself is
#  chat_template_kwargs, i.e. --default-chat-template-kwargs here.
#  On /v1/completions there is no chat template, so thinking is unreachable — and if it did
#  happen, the raw <think> trace would land in the text and break the answer regex, because
#  lm-eval's own think-stripping (enable_thinking / think_end_token) exists for the hf, vllm
#  and sglang backends only, not for API models.
#
# A.5  THE ONE DELIBERATE DEVIATION — THINK_DROP_STOPS
# -----------------------------------------------------------------------------
#  Documented nowhere; it is a judgement call, flagged so you can reverse it.
#  Each task's `until` strings ("Question:" for gsm8k and mmlu_pro, "</s>" for gpqa) are
#  matched by the sampler on the RAW stream — including the reasoning trace. A chain of
#  thought that restates the question therefore truncates the answer away and scores 0.
#  THINK_DROP_STOPS=1 sends until=[] whenever thinking is on. Set 0 to keep the yaml's stop
#  strings and compare. (On lm-eval 0.4.x, handle_arg_string returns the string "[]" rather
#  than a list, so the request carries one stop string "[]" that never matches — same effect,
#  odd-looking samples log.) The tokenizer EOS is still appended to `stop` regardless (A.8),
#  so the request always keeps a terminator.
#
# A.6  READING THE RESULTS
# -----------------------------------------------------------------------------
#  strict-match vs flexible-extract on gsm8k: strict requires the literal "#### <number>";
#  flexible takes the last number. A large gap is a FORMATTING signal, not an accuracy one.
#  The "#### N" convention exists ONLY in the few-shot examples (doc_to_target is the full
#  answer string, which ends in "#### N"), so at --num_fewshot 0 strict-match is structurally
#  0.0 on every document while flexible still scores. On the completions path with shots the
#  model imitates the convention and the two agree — [R] measured 0.9431 vs 0.9439.
#  Through the chat template it answers in prose ("The answer is 18"), strict collapses while
#  flexible stays high, and papers evaluating chat models on gsm8k report flexible-extract for
#  exactly this reason. Observed here: V4-Flash-w8a8, Think High via chat, 200 samples ->
#  strict 54% / flexible 97%. The 97% is the accuracy; the 54% is format adherence.
#
# A.7  OPERATIONAL NOTES
# -----------------------------------------------------------------------------
#  - Mixed runs work: gsm8k goes to /v1/completions and never touches the chat template,
#    while mmlu_pro/gpqa use it. Two CHAT tasks wanting different modes cannot share one
#    server, and resolve_run_mode dies up front rather than mislabelling a multi-day run.
#  - Idavidrein/gpqa is a GATED dataset: accept the terms and export HF_TOKEN when seeding
#    the offline cache, or preflight passes and lm-eval fails.
#  - LIMIT is PER LEAF TASK. mmlu_pro is a group of 14 subtasks, so LIMIT=5 runs 70 docs and
#    LIMIT=100 runs 1400. LIMIT < 1.0 is a FRACTION of each task, not a count. An oversized
#    LIMIT is harmless — lm-eval truncates via islice — it just runs the full set.
#  - Runtime: mmlu_pro is 12,032 docs at CONCURRENCY=1 with a 16k thinking budget — days.
#    LIMIT for iteration, or point mmlu_pro's P_TASK at one subtask (e.g. mmlu_pro_biology),
#    which is far easier to reason about than 5 docs from each of 14 subjects.
#  - Comparability with [R]: they ran num_concurrent=128 with CUDA graphs and FP4+FP8; you
#    run 1 with --enforce-eager on w8a8/Ascend. Small deltas expected, tens of points not.
#  - lm-eval version traps: fewshot_config.doc_to_text is ignored in 0.4.9.2, which strips
#    the CoT out of MMLU-Pro's few-shot examples (EleutherAI/lm-evaluation-harness#3457).
#    Confirm the yaml values above with `lm-eval validate --tasks gsm8k,mmlu_pro`.
#
# A.8  EOS STOP STRING ON THE CHAT PATH
# -----------------------------------------------------------------------------
#  LocalChatCompletion defaults tokenizer_backend=None, which nulls the tokenizer and leaves
#  lm-eval with no EOS to add to `stop` (the "Cannot determine EOS string" warning).
#  tokenizer_backend=huggingface makes lm-eval load TOKENIZER and read eos_token from it;
#  handle_stop_sequences() appends it to `stop`. Safe only because tokenized_requests=False,
#  which keeps apply_chat_template on the JSON path — the server applies its own encoder,
#  as required since [D] ships no Jinja template. preflight() proves the tokenizer loads and
#  exposes an eos_token before the server starts; EOS_STRING=... bypasses tokenizer loading.
# =============================================================================
