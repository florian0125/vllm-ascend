#!/usr/bin/env bash
# a5_env.sh — single environment file for vllm-ascend on Atlas A3 (Ascend 910).
# Fuses install-time + runtime + performance vars. Perf knobs adopted from a
# known-good colleague config (see SESSION_SPEC.md for the classification).
#
# Adapted from env_a5.sh. Only the SOC target and the vLLM line move; the
# torch / torch-npu / triton / CANN stack is identical to A5 because vLLM 0.21.0
# is tracked by the vllm-ascend MAIN branch, whose matched stack is still
# CANN 9.0.0 + torch/torch-npu 2.10.0 + triton-ascend 3.2.1.
#
# USAGE: SOURCE it (do NOT execute), every shell, before building or running:
#     conda activate moeoffv5
#     source a5_env.sh
# Then pick a free card per run:  ASCEND_RT_VISIBLE_DEVICES=5 python run_infer.py ...

# ---- 0. Guard: must be sourced ---------------------------------------------
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: source this file, do not run it ->  source ${BASH_SOURCE[0]}" >&2
  exit 1
fi

conda activate moeoffv5-keyi
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ---- 1. CANN / NNAL (system-level) -----------------------------------------
# export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend}"
export ASCEND_HOME_PATH="/home/keyi/Ascend"
_TOOLKIT_ENV="${ASCEND_HOME_PATH}/ascend-toolkit/set_env.sh"
_NNAL_ENV="${ASCEND_HOME_PATH}/nnal/atb/set_env.sh"
[[ -f "${_TOOLKIT_ENV}" ]] && source "${_TOOLKIT_ENV}" || echo "WARN: missing ${_TOOLKIT_ENV}"
[[ -f "${_NNAL_ENV}" ]]    && source "${_NNAL_ENV}"    || echo "WARN: missing ${_NNAL_ENV} (libatb.so)"

# ---- 2. CRITICAL FIX: clear stale container path ---------------------------
# Leftover TORCH_NPU_PATH=/torch_npu breaks the C++ build (NPUStream.h not found).
unset TORCH_NPU_PATH

# ---- 3. A3 build target (DETECTED on this machine) -------------------------
export SOC_VERSION="ascend950pr_9579"     # confirm with `npu-smi info -t board -i 0`
# NOTE: on A3 this value really matters — the full custom AscendC kernels are
# compiled against it (unlike A5, which skipped them). A wrong SOC here = a
# kernel build that succeeds but is wrong for the silicon.

# ---- 4. Pinned versions ----------------------------------------------------
# vLLM moves to 0.21.0. vllm-ascend itself is NOT pinned to a tag here — it is
# built from the `main` branch (no v0.21.0 vllm-ascend release exists yet;
# main carries the CI commitment for the vLLM v0.21.0 tag). See INSTALL_A3.md.
export VLLM_VERSION="0.21.0"
export TORCH_VERSION="2.10.0"
export TORCH_NPU_VERSION="2.10.0"
export TORCHVISION_VERSION="0.25.0"
export TORCHAUDIO_VERSION="2.10.0"
export TRITON_ASCEND_VERSION="3.2.1"

# ---- 5. Pip indexes --------------------------------------------------------
# Global primary index is set to Huawei via `pip config` (see INSTALL_A3.md
# Phase 0): https://mirrors.huaweicloud.com/repository/pypi/simple/
# The two below are passed PER-COMMAND as --extra-index-url. Do NOT set a global
# extra-index-url with multiple URLs (pip loops on resolution).
# NOTE: TORCH_CPU_IDX is the ONLY host of the torch +cpu wheel (no PyPI mirror
# carries it) — on a slow link, fetch that wheel with aria2c instead (Phase 2).
export HUAWEI_PYPI="https://mirrors.huaweicloud.com/repository/pypi/simple/"
export TORCH_CPU_IDX="https://download.pytorch.org/whl/cpu/"
export ASCEND_IDX="https://mirrors.huaweicloud.com/ascend/repos/pypi"   # torch-npu, triton-ascend

# ---- 6. Performance / runtime knobs (adopted from colleague's good config) --
export VLLM_USE_V1=1                                  # force V1 engine (already default)
export VLLM_ASCEND_ENABLE_NZ=1                        # NZ weight layout (note: deprecating -> additional_config.weight_nz_mode)
export HCCL_OP_EXPANSION_MODE="AIV"                   # expand HCCL ops onto AI Vector cores
export HCCL_BUFFSIZE=1024                             # HCCL buffer (MB)
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"  # reduce NPU mem fragmentation
export VLLM_ASCEND_ENABLE_FUSED_MC2=0                 # safe off on single card (no comm to fuse)
export VLLM_SERVER_DEV_MODE=1                         # extra dev endpoints (only affects `vllm serve`)
export ASCEND_LAUNCH_BLOCKING=0                       # async NPU dispatch (set 1 only to debug)
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_BATCH_INVARIANT=0                         # Enable batch invariance to get consistent results regardless of scheduling for online serve mode
export VLLM_ENABLE_V1_MULTIPROCESSING=0               # Turn off multiprocessing to make the scheduling deterministic
export COMPILE_CUSTOM_KERNELS=1
export COPY_NON_BLOCKING=1

# CPU threading
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10                             # cap per-process threads; tune to your core count

# Generous timeouts — prevent spurious timeouts on long (large-model) loads.
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export VLLM_ENGINE_READY_TIMEOUT_S=10000
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000

# --- V4-SPECIFIC KV features: leave OFF for Qwen/V2-Lite ablations. ---
# These come from the DeepSeek-V4-Flash launch config and can destabilize
# standard-attention models. serve_deepseek_v4.sh enables them itself.
export USE_MULTI_BLOCK_POOL=1
export USE_MULTI_GROUPS_KV_CACHE=1

# ---- 7. Distributed comms (auto-detected to THIS box; override if wrong) ----
# Colleague used enp189s0f0 / 80.48.37.147 — those are THEIR machine. We detect ours.
# export NIC_NAME="enp194s0f0"
# Pick the first real, non-virtual interface with a global-scope IPv4 address.
export NIC_NAME=$(ip -o -4 addr show scope global up 2>/dev/null | \
  awk '$2 !~ /^(veth|docker|br-|virbr|tun|tap|cni|flannel|lo|dummy)/ {print $2; exit}')

export LOCAL_IP=$(ip -o -4 addr show dev "$NIC_NAME" scope global | awk '{print $4}' | cut -d/ -f1)
export HCCL_IF_IP="${LOCAL_IP}"
export GLOO_SOCKET_IFNAME="${NIC_NAME}"
export TP_SOCKET_IFNAME="${NIC_NAME}"
export HCCL_SOCKET_IFNAME="${NIC_NAME}"

echo "Using NIC=${NIC_NAME} IP=${LOCAL_IP}"   # sanity-check before launching

# Mooncake libs — only needed for disaggregated/KV-transfer setups; add if present.
_MOONCAKE_DIR="${ASCEND_HOME_PATH}/ascend-toolkit/latest/python/site-packages/mooncake"
[[ -d "${_MOONCAKE_DIR}" ]] && export LD_LIBRARY_PATH="${_MOONCAKE_DIR}:${LD_LIBRARY_PATH}"

# ---- 8. jemalloc + lib path ------------------------------------------------
_JEMALLOC="$(find /usr/lib64 /usr/lib -name 'libjemalloc.so.2' 2>/dev/null | head -n1)"
[[ -n "${_JEMALLOC}" ]] && export LD_PRELOAD="${_JEMALLOC}:${LD_PRELOAD}" \
  || echo "WARN: libjemalloc.so.2 not found (yum install jemalloc)"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

# ---- 9. Weights / multiproc ------------------------------------------------
export VLLM_USE_MODELSCOPE="True"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

# ---- 10. Per-run reminder --------------------------------------------------
# export ASCEND_RT_VISIBLE_DEVICES=5   # pick a free card from `npu-smi info`

# ---- 11. Summary -----------------------------------------------------------
echo "[env_a3] SOC=${SOC_VERSION} torch=${TORCH_VERSION} torch-npu=${TORCH_NPU_VERSION} vllm=${VLLM_VERSION}"
echo "[env_a3] NIC=${NIC_NAME} IP=${LOCAL_IP}  TORCH_NPU_PATH='${TORCH_NPU_PATH:-(unset, good)}'"
echo "[env_a3] perf: NZ=${VLLM_ASCEND_ENABLE_NZ} HCCL_OP=${HCCL_OP_EXPANSION_MODE} BUFF=${HCCL_BUFFSIZE} MC2=${VLLM_ASCEND_ENABLE_FUSED_MC2}"
command -v npu-smi >/dev/null 2>&1 && echo "[env_a3] npu-smi: OK" || echo "[env_a3] WARN: npu-smi not on PATH"