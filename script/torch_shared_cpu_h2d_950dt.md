# Ascend 950DT Torch 共享 CPU 专家权重验证

## 目标和边界

这组脚本验证：在同一台 950DT 服务器上，全局只保留一份文件映射的 CPU 专家权重，所有 EP rank 都能通过 Torch 从任意专家切片加载到本 rank 的 NPU。生产代码还支持 `exclusive_dynamic`：共享 CPU 只保存当前不在任何 NPU 上的紧凑专家集合，全部 rank 的 decode NPU 槽与该 CPU 集合共同组成唯一一份 canonical 权重。

这不是完整的 `ExpertOffloadManager` 模型链路测试。脚本除独立 API smoke test 外，还会直接调用仓库中的 `TorchSharedCPUWeightPool` 和 `TorchSharedCPUH2DTransport`，验证每个 rank 只写自己的 checkpoint shard 后，其他 rank 能通过生产共享池/transport 加载该专家。它仍不覆盖：

- 路由、LRC、`log2phy` 和专家替换；
- prefill 的 EP shard/ALLTOALL 路径；
- safetensors 加载和权重后处理生命周期；
- NZ、FP4 等格式元数据及真实 GMM 精度；
- ACL Graph host callback、预取流和计算重叠。

仓库现已提供对应的生产调用链实现，包括 `exclusive_dynamic` 的跨 rank 原子 swap 和 prefill PyHCCL raw-pointer P2P。共享 CPU 脚本本身仍只是独立的前置条件验证，不能替代真实模型精度和性能测试；跨卡 D2D 另由 `validate_torch_prefill_p2p_950dt.py` 做轻量原语验证。

950DT 文档中的“不支持内存共享（IPC）”是指 NPU Device Tensor IPC。本测试不共享 NPU Tensor，只使用 Linux `MAP_SHARED` CPU 文件映射，然后执行 CPU 到本地 NPU 的 `copy_`。

## 文件

- `validate_torch_shared_cpu_h2d_950dt.py`：多 rank 功能、并发、pinned staging、生产 pool/transport 切片和时延测试。
- `inspect_torch_shared_cpu_pss.sh`：从 `/proc/<pid>/smaps` 汇总共享映射的 PSS。
- `run_torch_shared_cpu_h2d_950dt.sh`：环境打印和 `torchrun` 启动入口。
- `validate_torch_prefill_p2p_950dt.py`：验证 raw Storage 权重、非连续 MXFP4 scale、零偏移 `uint8` buffer、`HcclBatchSendRecv` 和同 stream 分块复用。
- `run_torch_prefill_p2p_950dt.sh`：跨卡 prefill P2P 快速启动入口。

## 前置条件

1. 已安装匹配版本的 PyTorch、TorchNPU、CANN、驱动和固件。
2. 每个进程对应一张 NPU。
3. `/dev/shm` 有足够空间。默认 8 rank、每 rank 4 个逻辑专家、每专家 12.75 MiB，共约 408 MiB。
4. PyTorch 构建支持 Gloo。共享 mmap 的控制面以及 PyHCCL root info 交换都使用 CPU group，不能用 HCCL device group 代替。

先记录环境：

```bash
npu-smi info

python3 - <<'PY'
import torch
import torch_npu

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu_count:", torch.npu.device_count())
for index in range(torch.npu.device_count()):
    print(index, torch.npu.get_device_name(index))
PY

cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
```

## 快速运行

推荐显式指定共享文件名，便于在另一个终端检查：

```bash
cd /path/to/vllm-ascend

export NPROC_PER_NODE=8
export DIST_BACKEND=gloo
export HOLD_SECONDS=60
export SHARED_FILE=/dev/shm/torch_shared_cpu_h2d_run1.bin

bash script/run_torch_shared_cpu_h2d_950dt.sh
```

## vLLM Ascend 生产配置

同一台机器上的多 rank replicated expert offload 可使用：

```json
{
  "expert_offload_config": {
    "expert_offload": true,
    "enable_multi_card": true,
    "storage_partition_mode": "replicated",
    "h2d_backend": "torch",
    "shard_per_rank": true,
    "shared_cpu_weights": true,
    "shared_cpu_weights_dir": "/dev/shm"
  }
}
```

该组合的语义是：

- checkpoint 加载仍由每个 rank 只写自己的 EP shard；
- w13、w2、scale、offset 和 scale-bias 的底层 CPU 存储由所有 rank 共同映射，全局常驻数据是一份完整专家权重；
- target model finalize 完成并经过 CPU group barrier 后，decode 可以把任意专家放到任意 rank 的 NPU slot；
- Torch H2D 先复制到 rank 本地 pinned staging，再执行 `non_blocking=True` 的 NPU `copy_`；staging 是有界临时工作集，不是第二份常驻完整权重；
- multi-card prefill 仍只加载本 rank 的静态 EP shard，并走 ALLTOALL/标准 EP 计算边界；
- target finalize 之后才注册的 draft/MTP 层仍保持 owner-shard placement；
- 该模式只支持同机 rank，不提供跨机器 CPU 共享。

如果要求 CPU 与全部 decode NPU 在全局共同只保存一份 canonical 权重，使用方案 A 的正式配置：

```json
{
  "expert_offload_config": {
    "expert_offload": true,
    "enable_multi_card": true,
    "storage_partition_mode": "exclusive_dynamic",
    "num_device_experts": 32,
    "h2d_backend": "torch",
    "shard_per_rank": true,
    "shared_cpu_weights": true,
    "shared_cpu_weights_dir": "/dev/shm",
    "moe_offload_debug": true
  }
}
```

其中 `num_device_experts` 是每层全局 decode NPU 容量。例如 EP=8、配置值为 32 时，每 rank 分配 4 个 decode 专家槽；假设模型有 128 个专家，共享 CPU 紧凑池只分配 `128 - 32 = 96` 个专家槽。

该模式的正式数据流是：

1. 初始化时，全局初始 NPU 专家按 physical slot 切到各 rank，checkpoint 直接写入对应 rank 的 decode NPU 槽；剩余专家按紧凑 CPU slot 分配唯一 writer rank，写入所有 rank 共同映射的 mmap。
2. Decode placement 保留未迁移专家的 rank 和 slot。miss 发生时，槽所在 rank 先把 victim D2H 到小型 pinned scratch，同时把 incoming 从共享 CPU H2D 到该槽；设备复制成功后才把 victim 写回 incoming 原来的紧凑 CPU slot。
3. 全部 rank 通过 CPU group 汇总 device-copy 和 CPU-commit 状态；任一 rank 失败都会显式报错，所有权表和 `log2phy` 不会提交新版本。
4. Prefill 仍使用静态 EP shard。CPU-owned 专家从共享 mmap H2D；本 rank NPU-owned 专家本地 D2D；其他 rank NPU-owned 专家先 pack 到零偏移 `uint8` NPU buffer，再通过独立 PyHCCL communicator 的 raw-pointer `HcclBatchSendRecv` 直接传到目标 rank。
5. 跨卡 pack、HCCL send/recv 和 unpack 都提交到同一个 `load_stream`，由 stream 顺序保证接收完成后才读取 recv buffer；不再依赖 `torch.distributed` Work 对另一个 stream 的隐式完成语义。
6. Prefill 只是向 prefill pool 写快照，不改变 canonical 所有权或 decode `log2phy`。正式路径不创建 CPU snapshot，也不提供远端 NPU→CPU→NPU 的静默回退。

配置校验会拒绝非 Torch backend、`enable_multi_card=false` 或 `shard_per_rank=false` 的共享 CPU 组合。全局共享模式中的共享目录、共享源、HCCL P2P 或分布式 swap 任一环节不可用时都会显式报错。

如果机器不支持或暂时不希望启用跨卡专家权重传输，可以使用 rank-local exclusive 配置：

```json
{
  "expert_offload_config": {
    "expert_offload": true,
    "enable_multi_card": true,
    "storage_partition_mode": "exclusive_dynamic",
    "num_device_experts": 32,
    "h2d_backend": "torch",
    "shard_per_rank": true,
    "shared_cpu_weights": false,
    "moe_offload_debug": true
  }
}
```

该组合不创建 `/dev/shm` mmap，也不读取 `shared_cpu_weights_dir`。全局专家先按 EP rank 静态切分；每个 rank 的分片再拆成互斥的私有 CPU 集合和本地 decode NPU 槽。因此每层满足：

```text
所有 rank 的 (private CPU experts + local decode NPU experts)
    = 一份完整全局专家权重
```

Decode planner 使用 owner-shard 约束，miss 只在同一 rank 内执行 victim D2H 和 incoming H2D；prefill 的 CPU-owned 专家做本地 H2D，已在本地 decode NPU 的专家做本地 D2D。专家权重不会跨 rank 传输；MC2/ALLTOALL 仍可跨 rank 传递 token。配置中的 `num_device_experts` 仍是全局 NPU 槽数，例如 EP=8、值为 32 时每 rank 有 4 个 NPU 槽。

初始化完成后应看到 `cpu_mode=exclusive_sharded`、`expected_full_weight_copies_across_ep=1`，以及：

```text
[EXCLUSIVE-SHARDED] ... cpu_experts_per_layer=... npu_experts_per_layer=... cross_rank_weight_transfer=False
```

多卡 `exclusive_dynamic` 的配置边界为：`shared_cpu_weights=true` 使用全局 ownership 和跨卡权重 P2P；`shared_cpu_weights=false` 必须同时设置 `shard_per_rank=true`，使用上述 rank-local ownership。后一种配置不能做跨 rank 的专家负载均衡，只能在每个专家的静态 owner rank 内换入换出。

## 先验证 prefill 跨卡原语

本节只面向 `shared_cpu_weights=true` 的全局 ownership 路径；rank-local exclusive 配置不使用该 P2P 原语。完整模型初始化前，先用两张卡运行针对共享模式通信路径的轻量测试：

```bash
export NPROC_PER_NODE=2
bash script/run_torch_prefill_p2p_950dt.sh
```

该脚本使用与日志中相同的 MXFP4 scale 逻辑形状 `(64, 4096, 2)` 和 stride `(2, 128, 1)`，验证 raw Storage 权重、typed scale、PyHCCL `uint8` raw pointer、显式 `load_stream` 以及顺序分块。它会在启动时检查当前 `libhccl.so` 是否导出 `HcclBatchSendRecv`；全部 rank 成功时会输出：

```text
TORCH_PREFILL_P2P_RESULT PASS_ALL_RANKS
```

这是独立原语 smoke test，不会构造 `ExpertOffloadManager`、真实 internal-format 权重或运行 GMM；它通过后仍必须执行完整模型的初始化、prefill 精度和 decode 回归。

如果只验证共享 CPU mmap，而 Gloo 初始化失败，可以让基础 H2D 脚本改用
HCCL 控制组：

```bash
export DIST_BACKEND=hccl
bash script/run_torch_shared_cpu_h2d_950dt.sh
```

这只能验证共享 CPU H2D 前置条件；PyHCCL root info 交换和完整模型的
`exclusive_dynamic + shared_cpu_weights=true` 路径仍需要 Gloo，不能用该设置
绕过。

脚本使用独占创建。若 `SHARED_FILE` 已存在，会直接失败而不会覆盖。确认它只是上一次异常退出遗留的测试文件后，才能显式删除：

```bash
rm -f /dev/shm/torch_shared_cpu_h2d_run1.bin
```

## 验证物理内存只有一份

主测试打印 `[HOLD]` 后，在第二个终端运行：

```bash
cd /path/to/vllm-ascend
bash script/inspect_torch_shared_cpu_pss.sh \
  /dev/shm/torch_shared_cpu_h2d_run1.bin
```

不要把每个进程的 RSS 相加判断物理内存。共享页会同时计入多个进程的 RSS。应查看脚本汇总的 PSS：

- `pss_to_file_ratio` 接近 `1`：共享映射驻留物理页约为一份文件大小；
- 接近 rank 数：可能发生了私有复制，需要检查 `clone()`、`pin_memory()` 或映射方式；
- 明显小于 `1`：部分页未驻留，或测试还没进入 HOLD 阶段。

完整模型在设置 `moe_offload_debug=true` 后，会在共享专家源发布完成时自动汇总所有 EP rank 的共享映射 PSS，并由 rank 0 输出一行：

```text
[TORCH_SHARED_CPU_PSS] ranks=2 expected_shared_expert_mib=140352.0 mapped_size_per_rank_mib=140352.1..140352.1 mappings_per_rank=258..258 mapping_total_rss_mib=140352.0 mapping_total_pss_mib=140351.8 mapping_total_swap_pss_mib=0.0 pss_to_expected_ratio=1.000 result=PASS_ONE_PHYSICAL_COPY
```

- `PASS_ONE_PHYSICAL_COPY`：每个 rank 都映射了完整的共享 CPU pool（replicated 为全部专家，exclusive 为紧凑 CPU-owned 专家），汇总 PSS 在该 pool 一份数据的 ±5% 内且没有 Swap；
- `FAIL_MAPPING_SIZE_MISMATCH`：至少一个 rank 的映射不完整或存在多余映射；
- `FAIL_EXTRA_PHYSICAL_PAGES`：汇总 PSS 超过一份完整数据的允许误差；
- `INCONCLUSIVE_NOT_FULLY_RESIDENT`：部分权重页尚未驻留，暂时不能判断；
- `INCONCLUSIVE_SWAPPED`：共享权重发生换页，不能只用驻留 PSS 下结论；
- `UNAVAILABLE`：当前系统不允许读取 `/proc/self/smaps`。

`exclusive_dynamic` 初始化还会输出：

```text
[EXCLUSIVE-SHARED-AUDIT] ranks=8 layers=61 compact_shared_cpu_mib=... global_decode_npu_mib=... canonical_total_mib=... expected_one_model_mib=... cpu_experts_per_layer=[96, ...] global_npu_experts_per_layer=[32, ...] result=PASS_ONE_CANONICAL_MODEL
```

这里的 `PASS_ONE_CANONICAL_MODEL` 验证逻辑容量和实际 storage 字节：紧凑共享 CPU 加全部 rank 的 decode NPU canonical storage 等于一份完整模型专家权重。它不包含 prefill pool、pinned swap scratch、Torch staging 和派生 fp32 scale；这些都是运行工作区或派生数据，不属于 canonical 权重。CPU 是否确实只有一份物理页仍以同一次运行中的 `TORCH_SHARED_CPU_PSS` 为准。

发生 decode 迁移和 prefill 跨卡直传时可分别检查：

```text
[EXCLUSIVE-SHARED-SWAP] layer=... rank=... global_swaps=... local_swaps=...
[PREFILL-D2D-PLAN] layer=... rank=... typed_pack_tasks=... rounds=... chunk_limit=134217728 peak_send_chunk=... peak_recv_chunk=...
[PREFILL-D2D] layer=... rank=... local_tasks=... p2p_ops=... global_remote_experts=...
```

MXFP4 模型的跨卡元数据路径应看到 `typed_pack_tasks > 0`；每个普通通信块的峰值应不超过 `chunk_limit`。发送端和接收端的专家、组件、dtype、shape、offset 或字节数不一致时，会在进入正式 HCCL 传输前报 `Cross-rank prefill communication plans do not match`。要证明“直接跨卡 D2D”实际生效，至少应看到 `p2p_ops > 0`，并结合 HCCL profiler/trace 确认对应 send/recv；只有源码、单测或独立 smoke test 不能证明完整模型在 950DT 上已经通过。

## 测试内容和通过标准

### 1. CPU 共享可见性

Rank 0 修改共享池的第一个字节，所有 rank 必须读取到同一个值。每个 rank 输出的 `shared_inode` 必须相同。

### 2. 任意专家直接 H2D

每个 rank 选择下一 rank 逻辑拥有的专家，通过：

```python
npu_destination.copy_(shared_pageable_source, non_blocking=False)
```

将 `w13`、`w2` 和量化元数据的原始字节复制到本地 NPU并逐字节校验。通过标志：

```text
[PASS] direct_remote
```

这里使用阻塞 copy，因为 PyTorch 的 file-backed mmap storage 不是 pinned memory。功能通过不代表能够与计算异步重叠。

### 3. 多 rank 并发读取同一专家

所有 rank 同时读取最后一个专家，必须全部逐字节校验通过：

```text
[PASS] concurrent_same_expert
```

### 4. 每 rank 小型 pinned staging

数据流为：

```text
全局共享 pageable CPU 权重
  -> rank 本地 pinned staging（CPU memcpy）
  -> rank 本地 NPU（torch.copy_, non_blocking=True）
```

通过标志：

```text
[PASS] pinned_staging
```

该模式仍只有一份完整 CPU 专家池，但 H2D 期间会存在每 rank 少量临时专家副本。

### 5. 时延

每个 rank 输出一行 JSON：

```text
TORCH_SHARED_CPU_RESULT {...}
```

其中：

- `direct_pageable`：共享 mmap 直接阻塞 H2D；
- `pinned_staging_end_to_end`：CPU memcpy 加 pinned H2D 的端到端时间；
- `p50_ms`、`p99_ms`：只能视为当前独立测试的数据，不能直接当作 vLLM decode 性能。

可调迭代次数和专家字节数：

```bash
bash script/run_torch_shared_cpu_h2d_950dt.sh \
  --warmup 10 \
  --iterations 100 \
  --w13-bytes 8388608 \
  --w2-bytes 4194304 \
  --quant-bytes 786432
```

### 6. 生产 pool/transport 切片

每个 rank 只填充共享池中自己的逻辑专家 shard，然后选择下一 rank 拥有的专家，通过仓库的 `H2DCopyTask` 和 `TorchSharedCPUH2DTransport` 加载到本 rank NPU。通过标志：

```text
[PASS] production_pool_transport
```

这一项覆盖本次实现的共享文件协调、共享 tensor 解析、pinned staging 生命周期和 Torch H2D，但仍不覆盖完整模型加载、路由、`log2phy`、量化 GMM 精度或 ACL Graph。

## pinned_mem_register A/B 测试

950DT 支持 TorchNPU pinned allocator 的 `pinned_mem_register`。其要求包括 TorchNPU 26.0.0 及以上、HDK 26.0.RC1 及以上、CANN 8.5.0 及以上，并且不能和 `pin_memory_expandable_segments` 同时启用。

满足版本要求后，可对 pinned staging 做 A/B：

```bash
export PYTORCH_NPU_ALLOC_CONF=pinned_mem_register:True
export SHARED_FILE=/dev/shm/torch_shared_cpu_h2d_register.bin
bash script/run_torch_shared_cpu_h2d_950dt.sh
```

该配置只作用于 `pin_memory` 分配器，不应将它解释为 `torch.from_file` 的共享 mmap 已经变成 pinned storage。

## 结果解释

| 结果 | 结论 |
|---|---|
| direct、concurrent、staging、production slice 全通过，PSS 约一份 | 共享 CPU 与生产 pool/transport 前置验证通过，可继续跑完整模型验证 |
| direct 失败，staging 通过 | mmap 可共享，但当前栈不能直接 H2D；使用小型 pinned staging |
| direct 通过，`pool_is_pinned=false` | 功能成立，但不证明异步传输或计算重叠 |
| CPU 共享通过，所有 H2D 失败 | CPU mmap 正常，TorchNPU/CANN 不接受当前 Host 源 |
| PSS 接近 rank 数倍 | 未实现单份物理专家池，需排查私有副本 |

## 在 950DT 上启用生产配置前还需验证

1. 用真实 checkpoint 检查 w13/w2、scale/offset/scale-bias 格式转换后的跨 rank 字节一致性。
2. 验证 W8A8、W4A8 dynamic、MXFP4 的 GMM 精度，而不只是原始字节一致。
3. 检查 `/dev/shm` 容量、进程 PSS、NUMA 位置以及 pinned staging 峰值。
4. 验证初始化 barrier、异常退出、target/draft 生命周期和重复启动清理。
5. 测量 decode miss 的 H2D p50/p99、吞吐和计算重叠。
6. 对 `exclusive_dynamic + shared_cpu_weights=true` 验证 MC2 decode 的全局 ownership、原子 swap、`log2phy` 一致性，以及 ALLTOALL prefill 的 HCCL P2P；同时确认没有 NPU→CPU→NPU 的远端回退。
7. 对 `exclusive_dynamic + shared_cpu_weights=false` 验证每个 rank 的 CPU/NPU ownership 并集恰好等于本 rank EP shard、不同 rank 互不重叠，且日志始终显示 `cross_rank_weight_transfer=False`。

## 官方参考

- [PyTorch `torch.from_file`](https://docs.pytorch.org/docs/stable/generated/torch.from_file.html)
- [PyTorch multiprocessing](https://docs.pytorch.org/docs/stable/multiprocessing.html)
- [TorchNPU pinned allocator 配置](https://github.com/Ascend/pytorch/blob/master/docs/zh/api/environment_variable/memory_management/PYTORCH_NPU_ALLOC_CONF.md)
- [TorchNPU NPU Tensor IPC 及 950DT 约束](https://github.com/Ascend/pytorch/blob/master/docs/zh/developer_notes/memory_management/memory_sharing_ipc.md)
