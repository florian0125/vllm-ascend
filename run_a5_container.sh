#!/usr/bin/env bash
# =============================================================================
# run_container_a5.sh — launch a long-lived vllm-ascend A5 container with NPU
# access, your data mounted, and (by default) your vllm-ascend SOURCE mounted
# for live Python editing.
#
# WHAT GETS MOUNTED (host path -> SAME path inside; read-write; persists on host):
#   /home      scripts, env_a5_container.sh, tokenizer, HF cache, sharegpt, results
#   /mnt/data  model weights (/mnt/data/DeepSeek-V4-Flash-w8a8-mtp)
#   /data      extra data disk (if present)
#   + EXTRA_MOUNTS (space-separated host paths)
#
# LIVE EDITING (MOUNT_SRC=1, default):
#   Bind-mounts your host vllm-ascend over the install path so you can edit its
#   PYTHON on the host and see changes with NO rebuild. The image's entrypoint
#   restores the compiled kernels into the mount at start (see Dockerfile). After
#   a .py edit, just restart `vllm serve` — no rebuild, no re-install.
#   (Requires the image built from the Dockerfile that has the entrypoint baked.)
#
# Usage:
#   ./run_container_a5.sh
#   NAME=vllm-a5-keyi IMAGE=vllm-ascend-a5:moe5 ./run_container_a5.sh
#   MOUNT_SRC=0 ./run_container_a5.sh                       # use the baked source (no live edit)
#   VA_SRC=/path/to/vllm-ascend-moe5 ./run_container_a5.sh  # non-default source path
#   EXTRA_MOUNTS="/scratch /mnt/datasets" ./run_container_a5.sh
# =============================================================================
set -euo pipefail

IMAGE="${IMAGE:-vllm-ascend-a5:moe5}"
NAME="${NAME:-vllm-a5-${USER:-dev}}"
SHM="${SHM:-1000g}"

DATA_MOUNTS="${DATA_MOUNTS:-/home /data /mnt/data}"
EXTRA_MOUNTS="${EXTRA_MOUNTS:-}"

# Live-edit: mount host sources over the editable install paths.
MOUNT_SRC="${MOUNT_SRC:-1}"                     # vllm-ascend (the one you edit)
VA_SRC="${VA_SRC:-/home/keyi/code/vllm-ascend-moe5}"
VA_DST="/vllm-workspace/vllm-ascend"
MOUNT_VLLM="${MOUNT_VLLM:-1}"                   # vllm too, so the pair can't drift
VLLM_SRC="${VLLM_SRC:-/home/keyi/code/vllm}"    # empty-target build = pure python, safe to mount
VLLM_DST="/vllm-workspace/vllm"

# --- guards -----------------------------------------------------------------
if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  echo "Container '${NAME}' already exists."
  echo "  enter it  :  docker exec -it ${NAME} bash"
  echo "  remove it :  docker rm -f ${NAME}"
  exit 1
fi
docker image inspect "${IMAGE}" >/dev/null 2>&1 || {
  echo "ERROR: image '${IMAGE}' not found locally. Build/tag it first." >&2; exit 1; }

# --- data mounts (skip non-existent) ----------------------------------------
mounts=(); shown=(); missing=()
for d in ${DATA_MOUNTS} ${EXTRA_MOUNTS}; do
  if [[ -d "$d" ]]; then mounts+=( -v "$d:$d" ); shown+=( "$d" ); else missing+=( "$d" ); fi
done
[[ ${#shown[@]} -eq 0 ]] && { echo "ERROR: none of the data paths exist: ${DATA_MOUNTS} ${EXTRA_MOUNTS}" >&2; exit 1; }
[[ ${#missing[@]} -gt 0 ]] && echo "note: skipping non-existent path(s): ${missing[*]}"
[[ " ${shown[*]} " == *" /home "*     ]] || echo "WARN: /home not mounted."
[[ " ${shown[*]} " == *" /mnt/data "* ]] || echo "WARN: /mnt/data not mounted."

# --- live-edit source mount -------------------------------------------------
src_mount=()
if [[ "${MOUNT_SRC}" == "1" ]]; then
  if [[ -d "${VA_SRC}" ]]; then
    src_mount=( -v "${VA_SRC}:${VA_DST}" )
    live_msg="${VA_SRC} -> ${VA_DST}  (edit .py on host; restart 'vllm serve' to apply)"
  else
    echo "WARN: MOUNT_SRC=1 but VA_SRC not found: ${VA_SRC} — launching WITHOUT live-edit mount."
    live_msg="(none — VA_SRC missing)"
  fi
else
  live_msg="off (using the vllm-ascend baked into the image)"
fi

# --- live-edit vllm mount (keeps vllm + vllm-ascend versions in sync) --------
if [[ "${MOUNT_VLLM}" == "1" ]]; then
  if [[ -d "${VLLM_SRC}" ]]; then
    src_mount+=( -v "${VLLM_SRC}:${VLLM_DST}" )
    vllm_msg="${VLLM_SRC} -> ${VLLM_DST}"
  else
    echo "WARN: MOUNT_VLLM=1 but VLLM_SRC not found: ${VLLM_SRC} — using baked vllm."
    vllm_msg="(none — VLLM_SRC missing)"
  fi
else
  vllm_msg="off (using the vllm baked into the image)"
fi

echo "Launching '${NAME}' from '${IMAGE}'"
echo "  data mounts : ${shown[*]}"
echo "  live edit   : ${live_msg}"
echo "  vllm mount  : ${vllm_msg}"

docker run -dit -u root \
  --name "${NAME}" \
  -e ASCEND_RUNTIME_OPTIONS=NODRV \
  --privileged=true \
  -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ \
  -v /usr/local/Ascend/firmware/:/usr/local/Ascend/firmware/ \
  -v /usr/local/sbin/:/usr/local/sbin \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  "${mounts[@]}" \
  "${src_mount[@]}" \
  --shm-size="${SHM}" \
  --net=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  "${IMAGE}" \
  /bin/bash

echo
echo "Started '${NAME}'. Verify:"
echo "  docker exec -it ${NAME} bash"
echo "  python -c 'import torch,vllm_ascend; print(vllm_ascend.__file__); print(hasattr(torch.ops._C_ascend,\"npu_hc_pre\"))'"
[[ "${MOUNT_SRC}" == "1" && -d "${VA_SRC}" ]] && \
  echo "  # vllm_ascend.__file__ should be under ${VA_DST}; the hasattr check should print True"