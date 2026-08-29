# Ascend 950DT Torch 共享 CPU 专家权重验证

## 目标和边界

这组脚本验证：在同一台 950DT 服务器上，全局只保留一份文件映射的 CPU 专家权重，所有 EP rank 都能通过 Torch 从任意专家切片加载到本 rank 的 NPU。

这是独立 API smoke test，不是 `ExpertOffloadManager` 生产调用链测试。它不覆盖：

- 路由、LRC、`log2phy` 和专家替换；
- prefill 的 EP shard/ALLTOALL 路径；
- safetensors 加载和权重后处理生命周期；
- NZ、FP4 等格式元数据及真实 GMM 精度；
- ACL Graph host callback、预取流和计算重叠。

950DT 文档中的“不支持内存共享（IPC）”是指 NPU Device Tensor IPC。本测试不共享 NPU Tensor，只使用 Linux `MAP_SHARED` CPU 文件映射，然后执行 CPU 到本地 NPU 的 `copy_`。

## 文件

- `validate_torch_shared_cpu_h2d_950dt.py`：多 rank 功能、并发、pinned staging 和时延测试。
- `inspect_torch_shared_cpu_pss.sh`：从 `/proc/<pid>/smaps` 汇总共享映射的 PSS。
- `run_torch_shared_cpu_h2d_950dt.sh`：环境打印和 `torchrun` 启动入口。

## 前置条件

1. 已安装匹配版本的 PyTorch、TorchNPU、CANN、驱动和固件。
2. 每个进程对应一张 NPU。
3. `/dev/shm` 有足够空间。默认 8 rank、每 rank 4 个逻辑专家、每专家 12.75 MiB，共约 408 MiB。
4. PyTorch 构建支持 Gloo；如果不支持，改用 HCCL 控制组。

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

如果 Gloo 初始化失败：

```bash
export DIST_BACKEND=hccl
bash script/run_torch_shared_cpu_h2d_950dt.sh
```

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
| direct、concurrent、staging 全通过，PSS 约一份 | 可继续设计 Torch shared Host 专家池 |
| direct 失败，staging 通过 | mmap 可共享，但当前栈不能直接 H2D；使用小型 pinned staging |
| direct 通过，`pool_is_pinned=false` | 功能成立，但不证明异步传输或计算重叠 |
| CPU 共享通过，所有 H2D 失败 | CPU mmap 正常，TorchNPU/CANN 不接受当前 Host 源 |
| PSS 接近 rank 数倍 | 未实现单份物理专家池，需排查私有副本 |

## 进入生产集成前还需验证

1. 用真实 checkpoint 的 `w13/w2`、scale/offset/scale-bias 和格式转换后字节建立共享池。
2. 复用生产 `H2DCopyTask`、NPU连续 slot storage 和 load stream。
3. 验证 W8A8、W4A8 dynamic、MXFP4 的 GMM 精度，而不只是原始字节一致。
4. 验证初始化 barrier、只读阶段、异常退出清理和 MTP 后注册层。
5. 分别测 direct、staging 的 H2D 带宽、p50/p99、NUMA影响以及计算重叠。
6. 再验证 MC2 decode 的全局 placement、`log2phy` 和 ALLTOALL prefill 边界。

## 官方参考

- [PyTorch `torch.from_file`](https://docs.pytorch.org/docs/stable/generated/torch.from_file.html)
- [PyTorch multiprocessing](https://docs.pytorch.org/docs/stable/multiprocessing.html)
- [TorchNPU pinned allocator 配置](https://github.com/Ascend/pytorch/blob/master/docs/zh/api/environment_variable/memory_management/PYTORCH_NPU_ALLOC_CONF.md)
- [TorchNPU NPU Tensor IPC 及 950DT 约束](https://github.com/Ascend/pytorch/blob/master/docs/zh/developer_notes/memory_management/memory_sharing_ipc.md)
