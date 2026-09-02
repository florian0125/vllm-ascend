#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
nproc_per_node=${NPROC_PER_NODE:-2}

echo "=== Torch/HCCL prefill P2P smoke test ==="
echo "nproc_per_node=$nproc_per_node"

python3 -c '
import torch
import torch_npu

print("torch=", torch.__version__)
print("torch_npu=", getattr(torch_npu, "__version__", "unknown"))
print("npu_count=", torch.npu.device_count())
'

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="$nproc_per_node" \
  "$script_dir/validate_torch_prefill_p2p_950dt.py" \
  "$@"
