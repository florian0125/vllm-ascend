#!/usr/bin/env bash
# =============================================================================
# build_image_a5_local.sh — build the A5 image from your LOCAL vllm +
# vllm-ascend checkouts (Dockerfile_a5.local), WITHOUT a docker restart.
#
# The base image is NOT pulled here — load it once with skopeo (userspace, uses
# your proxy.sh) so the daemon never has to. This script fails early if it's
# missing and prints the exact skopeo command.
#
#   source /home/keyi/proxy.sh          # so the Mooncake clone + pip get the proxy
#   ./build_image_a5_local.sh
# =============================================================================
set -euo pipefail

CODE_DIR="${CODE_DIR:-/home/keyi/code}"                              # parent containing vllm/ and vllm-ascend-moe5/
VLLM_DIR="${VLLM_DIR:-vllm}"                                         # relative to CODE_DIR
VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-vllm-ascend-moe5}"              # relative to CODE_DIR
DOCKERFILE="${DOCKERFILE:-${CODE_DIR}/${VLLM_ASCEND_DIR}/Dockerfile.a5.openEuler.local}"
IMAGE="${IMAGE:-vllm-ascend-a5:moe5}"
SOC_VERSION="${SOC_VERSION:-ascend950pr_9579}"                       # confirm: npu-smi info -t board -i 0
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}"
BASE_IMAGE="quay.io/ascend/cann:9.0.0-950-openeuler24.03-py3.12"

die(){ echo "ERROR: $*" >&2; exit 1; }

[[ -d "${CODE_DIR}/${VLLM_DIR}" ]]        || die "no ${CODE_DIR}/${VLLM_DIR} (your local vLLM checkout)"
[[ -d "${CODE_DIR}/${VLLM_ASCEND_DIR}" ]] || die "no ${CODE_DIR}/${VLLM_ASCEND_DIR} (your local vllm-ascend checkout)"
[[ -f "${DOCKERFILE}" ]]                  || die "Dockerfile not found: ${DOCKERFILE}"

# base image must already be loaded (via skopeo) — do NOT let the daemon pull it
if ! docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
  cat >&2 <<EOF
ERROR: base image not present locally: ${BASE_IMAGE}
Load it first WITHOUT restarting docker (skopeo honors your proxy.sh):

  source /home/keyi/proxy.sh
  yum install -y skopeo    # if needed
  skopeo copy docker://${BASE_IMAGE} docker-daemon:${BASE_IMAGE}
  # if you get 'x509: certificate signed by unknown authority' (proxy TLS
  # interception), add:  --src-tls-verify=false   (blobs are still digest-verified)

then re-run this script.
EOF
  exit 1
fi

# submodule / artifact reminder (COPY takes the on-disk tree, so kernel sources
# and a clean build/ must already be right in the context)
if [[ -f "${CODE_DIR}/${VLLM_ASCEND_DIR}/.gitmodules" ]]; then
  echo "REMINDER: ensure kernel submodules are initialised and artifacts are clean in the context:"
  echo "  git -C ${CODE_DIR}/${VLLM_ASCEND_DIR} submodule update --init --recursive"
  echo "  rm -rf ${CODE_DIR}/${VLLM_ASCEND_DIR}/{build,csrc/build,*.egg-info}"
fi

# proxy passthrough for RUN steps (Mooncake clone + pip). NOT needed for the base
# (already loaded). Source proxy.sh before running so these are set.
PROXY_ARGS=()
_HP="${HTTPS_PROXY:-${https_proxy:-}}"; _HPP="${HTTP_PROXY:-${http_proxy:-${_HP}}}"
if [[ -n "${_HP}${_HPP}" ]]; then
  _NP="${NO_PROXY:-${no_proxy:-localhost,127.0.0.1,::1,mirrors.huaweicloud.com}}"
  PROXY_ARGS=( --build-arg http_proxy="${_HPP}" --build-arg https_proxy="${_HP:-$_HPP}"
               --build-arg HTTP_PROXY="${_HPP}" --build-arg HTTPS_PROXY="${_HP:-$_HPP}"
               --build-arg no_proxy="${_NP}"    --build-arg NO_PROXY="${_NP}" )
  echo "Build proxy: ${_HPP}  no_proxy=${_NP}"
else
  echo "WARN: no proxy env — Mooncake github clone will likely hang. source /home/keyi/proxy.sh first." >&2
fi

cat <<EOF
────────────────────────────────────────────────────────────
 Image        : ${IMAGE}
 Context      : ${CODE_DIR}          (COPYs ${VLLM_DIR}/ and ${VLLM_ASCEND_DIR}/)
 Dockerfile   : ${DOCKERFILE}
 Base (local) : ${BASE_IMAGE}
 SOC_VERSION  : ${SOC_VERSION}
────────────────────────────────────────────────────────────
Tip: add ${CODE_DIR}/.dockerignore to keep the context small, e.g.:
  **/build/  **/csrc/build/  **/*.egg-info/  **/__pycache__/  **/*.pyc
(keep .git so setuptools-scm can version vllm / vllm-ascend)
EOF

cd "${CODE_DIR}"
docker build \
  -f "${DOCKERFILE}" \
  --build-arg SOC_VERSION="${SOC_VERSION}" \
  --build-arg PIP_INDEX_URL="${PIP_INDEX_URL}" \
  "${PROXY_ARGS[@]}" \
  --progress=plain \
  -t "${IMAGE}" \
  .

echo
echo "Built ${IMAGE}. Back it up:  docker save ${IMAGE} | gzip > /mnt/data/images/${IMAGE//[:\/]/_}.tar.gz"