#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
nproc_per_node=${NPROC_PER_NODE:-8}
dist_backend=${DIST_BACKEND:-gloo}
hold_seconds=${HOLD_SECONDS:-60}
shared_file=${SHARED_FILE:-/dev/shm/vllm_ascend_torch_shared_cpu_h2d_$$.bin}

echo "=== TorchNPU environment ==="
python3 -c '
import torch
import torch_npu

print("torch=", torch.__version__)
print("torch_npu=", getattr(torch_npu, "__version__", "unknown"))
print("npu_count=", torch.npu.device_count())
print("devices=", [
    torch.npu.get_device_name(i) for i in range(torch.npu.device_count())
])
'

if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info
fi

echo
echo "=== Shared CPU H2D validation ==="
echo "nproc_per_node=$nproc_per_node"
echo "dist_backend=$dist_backend"
echo "shared_file=$shared_file"
echo "hold_seconds=$hold_seconds"
echo
echo "During HOLD, inspect physical sharing from another terminal:"
echo "  bash $script_dir/inspect_torch_shared_cpu_pss.sh $shared_file"
echo

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="$nproc_per_node" \
  "$script_dir/validate_torch_shared_cpu_h2d_950dt.py" \
  --backend "$dist_backend" \
  --shared-file "$shared_file" \
  --hold-seconds "$hold_seconds" \
  "$@"
