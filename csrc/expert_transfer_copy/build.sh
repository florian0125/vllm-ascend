#!/usr/bin/env bash
# Standalone mapped-DRAM copy kernel; source CANN before invoking.
set -euo pipefail
SRC=$(cd "$(dirname "$0")" && pwd)
BUILD=${1:?Usage: build.sh /absolute/build/directory}
SOC=${2:-ascend950pr_9599}
: "${ASCEND_HOME_PATH:?Source the CANN set_env.sh first}"
cmake -S "$SRC" -B "$BUILD" -DSOC_VERSION="$SOC" -DCMAKE_BUILD_TYPE=Release -DASCEND_HOME_PATH="$ASCEND_HOME_PATH"
cmake --build "$BUILD" -j4
TD=$(python -c 'import torch,os; print(os.path.dirname(torch.__file__))')
ND=$(python -c 'import torch_npu,os; print(os.path.dirname(torch_npu.__file__))')
ABI=$(python -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
g++ -shared -fPIC -std=c++17 -O2 -D_GLIBCXX_USE_CXX11_ABI="$ABI" \
 -I"$ASCEND_HOME_PATH/include" -I"$TD/include" -I"$TD/include/torch/csrc/api/include" \
 -I"$ND/include" -I"$SRC" "$SRC/dispatch.cpp" \
 -L"$BUILD/lib" -lnpu_expert_copy -L"$TD/lib" -ltorch -lc10 -ltorch_python -L"$ND/lib" -ltorch_npu \
 -Wl,-rpath,'$ORIGIN' -Wl,-rpath,"$BUILD/lib" -Wl,-rpath,"$TD/lib" -Wl,-rpath,"$ND/lib" \
 -o "$BUILD/lib/libnpu_expert_copy_dispatch.so"

# Install only into this source checkout; no system library changes.
PACKAGE_LIB="$SRC/../../vllm_ascend/expert_offload/lib"
mkdir -p "$PACKAGE_LIB"
cp "$BUILD/lib/libnpu_expert_copy.so" "$BUILD/lib/libnpu_expert_copy_dispatch.so" "$PACKAGE_LIB/"
