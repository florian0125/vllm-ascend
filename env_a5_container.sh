#!/usr/bin/env bash
# =============================================================================
# env_a5_container.sh — environment for vllm-ascend INSIDE the Docker container
#   base image: quay.io/ascend/cann:9.0.0-950-openeuler24.03-py3.12  (A5 / 950)
#
# Fork of your host `a5_env.sh`, changed only where the container differs:
#   1. no conda — the image uses the system python3.12 (a5_env.sh's
#      `conda activate moeoffv5-keyi` line does not apply inside the image)
#   2. ASCEND_HOME_PATH -> /usr/local/Ascend  (image CANN; host driver/firmware
#      are bind-mounted under that same tree)
#
# Every perf knob / timeout / NIC line / V4 KV feature is copied verbatim from
# a5_env.sh — including SOC_VERSION=ascend950pr_9579 and VLLM_BATCH_INVARIANT=0.
# Keep in sync if a5_env.sh changes.
#
# USAGE — inside the container, before every run:
#     source /home/keyi/env_a5_container.sh
#     CARD=3 PORT=6061 MODEL=/mnt/data/DeepSeek-V4-Flash-w8a8-mtp \
#       PREFETCH=0 TIMING=1 TASKS=gsm8k LIMIT=20 ./v4_eval_bench.sh
# =============================================================================

# ---- 0. Guard: must be sourced ---------------------------------------------
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: source this file, do not run it ->  source ${BASH_SOURCE[0]}" >&2
  exit 1
fi

# ---- 1. CANN / NNAL (installed at /usr/local/Ascend inside the image) -------
export ASCEND_HOME_PATH="/usr/local/Ascend"
_TOOLKIT_ENV="${ASCEND_HOME_PATH}/ascend-toolkit/set_env.sh"
_NNAL_ENV="${ASCEND_HOME_PATH}/nnal/atb/set_env.sh"
[[ -f "${_TOOLKIT_ENV}" ]] && source "${_TOOLKIT_ENV}" || echo "WARN: missing ${_TOOLKIT_ENV}"
[[ -f "${_NNAL_ENV}" ]]    && source "${_NNAL_ENV}"    || echo "WARN: missing ${_NNAL_ENV} (libatb.so)"

# ---- 2. clear any stale container torch-npu path ---------------------------
unset TORCH_NPU_PATH

# ---- 3. SOC target — MUST equal the SOC_VERSION you built the image with ----
#   A5 has two SKUs: 950PR (ascend950pr_9579) and 950DT (ascend950dt_9582).
#   The Dockerfile DEFAULTS to 950DT — override it at build time to match this box.
export SOC_VERSION="ascend950pr_9579"   # confirm with: npu-smi info -t board -i 0

# ---- 4. Performance / runtime knobs (from a5_env.sh) -----------------------
export VLLM_USE_V1=1
export VLLM_ASCEND_ENABLE_NZ=1
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
export VLLM_SERVER_DEV_MODE=1
export ASCEND_LAUNCH_BLOCKING=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_BATCH_INVARIANT=0                # a5_env.sh value (A3 used 1; A5 uses 0)
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# CPU threading
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10

# Generous timeouts — prevent spurious timeouts on long (large-model) loads.
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export VLLM_ENGINE_READY_TIMEOUT_S=10000
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000

# V4-specific KV features (safe for DeepSeek-V4; set 0 for Qwen / V2-Lite).
export USE_MULTI_BLOCK_POOL=1
export USE_MULTI_GROUPS_KV_CACHE=1

# ---- 5. Distributed comms (auto-detect THIS box's NIC; --net=host sees it) --
export NIC_NAME=$(ip -o -4 addr show scope global up 2>/dev/null | \
  awk '$2 !~ /^(veth|docker|br-|virbr|tun|tap|cni|flannel|lo|dummy)/ {print $2; exit}')
export LOCAL_IP=$(ip -o -4 addr show dev "$NIC_NAME" scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
export HCCL_IF_IP="${LOCAL_IP}"
export GLOO_SOCKET_IFNAME="${NIC_NAME}"
export TP_SOCKET_IFNAME="${NIC_NAME}"
export HCCL_SOCKET_IFNAME="${NIC_NAME}"
echo "Using NIC=${NIC_NAME} IP=${LOCAL_IP}"

# ---- 6. jemalloc + mooncake lib path (both set up by the Dockerfile) -------
_JEMALLOC="$(find /usr/lib64 /usr/lib -name 'libjemalloc.so.2' 2>/dev/null | head -n1)"
[[ -n "${_JEMALLOC}" ]] && export LD_PRELOAD="${_JEMALLOC}:${LD_PRELOAD}" \
  || echo "WARN: libjemalloc.so.2 not found"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"   # mooncake install prefix

# ---- 7. Weights / multiproc ------------------------------------------------
export VLLM_USE_MODELSCOPE="True"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

# ---- 8. Summary ------------------------------------------------------------
echo "[env_a5_container] SOC=${SOC_VERSION} NIC=${NIC_NAME} IP=${LOCAL_IP} TORCH_NPU_PATH='${TORCH_NPU_PATH:-(unset, good)}'"
command -v npu-smi >/dev/null 2>&1 && echo "[env_a5_container] npu-smi: OK" || echo "[env_a5_container] WARN: npu-smi not on PATH (is /usr/local/bin/npu-smi mounted?)"