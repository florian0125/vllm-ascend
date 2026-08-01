#!/bin/bash
# A5 DP2 V4 offload 验证脚本 (2026-08-01)
# V4 多卡不支持 TP(DSA 稀疏 attention 的 KvQuantSparseAttnSharedkvMetadata 在 TP>1 崩)。
# 改用 DP2: 2 个独立副本,每副本 tp1(card 各一),每副本跑单卡 offload(enable_multi_card=false)。
# 每副本 = 已验过的单卡 offload 配置(GSM8K 5/5),DP 只是开两份提吞吐。
set -e
unset ftp_proxy; unset https_proxy; unset http_proxy
export LD_PRELOAD=/lib64/libsqlite3.so.0   # sqlite3 v3.42 修复(torch_npu 导入)

export VLLM_VERSION=0.26.0
export VLLM_ASCEND_ENABLE_NZ=1
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_SERVER_DEV_MODE=1
export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3"     # DP2: 2 卡各跑一个副本

export USE_MULTI_BLOCK_POOL=1
export USE_MULTI_GROUPS_KV_CACHE=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=0

local_ip="141.61.141.58"; nic_name="enp35s0f2"
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
export PYTHONPATH=/mnt/share/l00889328/moe_offload_paper/vllm:/mnt/share/l00889328/moe_offload_paper/vllm-ascend:$PYTHONPATH
export HCCL_IF_IP=$local_ip; export GLOO_SOCKET_IFNAME=$nic_name; export TP_SOCKET_IFNAME=$nic_name; export HCCL_SOCKET_IFNAME=$nic_name
export ASCEND_CONNECT_TIMEOUT=10000; export ASCEND_TRANSFER_TIMEOUT=10000
export VLLM_ENGINE_READY_TIMEOUT_S=10000; export VLLM_RPC_TIMEOUT=3600000; export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null || true
sysctl -w vm.swappiness=0 2>/dev/null || true
sysctl -w kernel.numa_balancing=0 2>/dev/null || true

vllm serve /mnt/weight/A5-weights/DeepSeek-V4-Flash \
    --host 0.0.0.0 --port 8150 \
    --data-parallel-size 2 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 2 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name "glm-5" \
    --max-num-seqs 1 \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --enforce-eager \
    --reasoning-parser deepseek_v4 \
    --aggregate-engine-logging \
    --quantization fp8 --api-server-count 1 --safetensors-load-strategy 'prefetch' \
    --additional-config '{"enable_cpu_binding":true, "expert_offload_config": {"expert_offload": true, "enable_multi_card": true, "num_device_experts": 32, "num_device_layers": 1, "cache_policy_enabled": true, "expert_prefetch_enabled": false, "expert_prefetch_num":1, "moe_offload_debug": false}}'
