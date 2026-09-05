# vLLM Ascend Profiling 采集操作指南

本文适用于当前 DeepSeek-V4-Flash、TP1、DP2、EP/Expert Offload 服务，用于解决以下 profiling 日志：

```text
Incorrect schedule: Stop profiler while current state is RECORD
The profiling data cannot be parsed during the daemon process
```

## 1. 问题说明

这两条信息不表示推理请求失败。已有日志中：

- `/start_profile` 返回了 HTTP 200。
- `/v1/completions` 请求返回了 HTTP 200。
- worker 输出了 `Profiler stopped successfully`。

实际问题分为两部分：

1. vLLM 的 NPU worker 是 daemon 子进程，`torch_npu` 不能在该进程内完成 profiling 数据的在线解析，需要在普通前台进程中离线解析。
2. 当前 Ascend profiler wrapper 没有把 `warmup_iterations`、`active_iterations` 和 `wait_iterations` 接入底层 `torch_npu.profiler.schedule`。`max_iterations` 只是 vLLM 外层的 worker-step 计数，DP2 下两个 worker 可能在不同时间达到上限，造成采集窗口不一致。

因此，当前版本推荐使用：

> profiling 外预热 + 手动 start/stop + 独立目录保存原始数据 + 服务外离线解析

## 2. 推荐的服务启动配置

每轮采集使用一个新的绝对路径。例如：

```bash
PROFILE_RUN_DIR=/home/g00619970/moeoffload_multi/profile_run_0905
mkdir -p "${PROFILE_RUN_DIR}"
```

把服务的 profiler 参数设置为：

```bash
--profiler-config '{"profiler":"torch","torch_profiler_dir":"/home/g00619970/moeoffload_multi/profile_run_0905","torch_profiler_with_stack":false,"ignore_frontend":true,"delay_iterations":0,"max_iterations":0}'
```

配置说明：

- `max_iterations=0`：关闭各 DP worker 的独立自动停止，由 `/stop_profile` 统一停止。
- `delay_iterations=0`：收到 `/start_profile` 后立即开始采集。
- `ignore_frontend=true`：只关注 worker/NPU 执行，减少前端采集开销。
- `torch_profiler_with_stack=false`：先关闭栈信息，降低采集开销；确实需要模块栈时再打开。
- 当前版本不要依赖 `warmup_iterations`、`active_iterations`、`wait_iterations` 控制采集窗口。

启动后先确认服务正常：

```bash
curl -sS http://127.0.0.1:8155/health
```

## 3. 先在 profiling 外完成预热

在调用 `/start_profile` 之前，先发送 2～3 轮和正式测试相同形状的请求。这样可以排除首次执行、算子初始化和缓存建立等一次性开销。

示例：

```bash
curl -sS -X POST http://127.0.0.1:8155/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5","prompt":"profiling warmup","max_tokens":32,"temperature":0}'
```

确认每次请求都返回 HTTP 200，再开始正式采集。

如果需要保留 HTTP 状态码，可以使用：

```bash
curl -sS -o /tmp/profile_warmup_response.json \
  -w 'HTTP %{http_code}\n' \
  -X POST http://127.0.0.1:8155/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5","prompt":"profiling warmup","max_tokens":32,"temperature":0}'
```

## 4. 正式采集

### 4.1 开始 profiling

```bash
curl -sS -X POST http://127.0.0.1:8155/start_profile
```

必须确认该接口返回 HTTP 200，并在服务日志中看到：

```text
Starting profiler...
Profiler started.
```

### 4.2 发送 DP2 并发请求

DP2 下建议至少发送两个并发请求，让两个 DP worker 都进入真实模型执行路径。请求的 prompt、`max_tokens`、采样参数应固定，方便不同轮次比较。

```bash
curl -sS -o /tmp/profile_response_dp0.json \
  -w 'request-1 HTTP %{http_code}\n' \
  -X POST http://127.0.0.1:8155/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5","prompt":"fixed profiling request","max_tokens":50,"temperature":0}' &
PROFILE_REQUEST_PID_1=$!

curl -sS -o /tmp/profile_response_dp1.json \
  -w 'request-2 HTTP %{http_code}\n' \
  -X POST http://127.0.0.1:8155/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5","prompt":"fixed profiling request","max_tokens":50,"temperature":0}' &
PROFILE_REQUEST_PID_2=$!

wait "${PROFILE_REQUEST_PID_1}" "${PROFILE_REQUEST_PID_2}"
```

注意：

- 一定要等待两个请求都结束后再停止 profiling。
- 两个请求都应返回 HTTP 200。
- `max_tokens` 不宜过大，否则原始 trace 会很大，解析时间和磁盘占用都会明显增加。
- 如果只分析单请求延迟，可以只发送一个请求，但 DP2 中可能只有一个 DP worker 有完整的模型执行数据。

### 4.3 停止 profiling

```bash
curl -sS -X POST http://127.0.0.1:8155/stop_profile
```

等待日志出现：

```text
Profiler stopped successfully.
Profiler stopped.
```

当前版本仍可能同时输出：

```text
Incorrect schedule: Stop profiler while current state is RECORD
The profiling data cannot be parsed during the daemon process
```

只要满足以下条件，这里的 daemon parsing `ERROR` 可以按“在线解析失败”处理，不应直接判断为采集失败：

- 推理请求返回 HTTP 200。
- `/stop_profile` 返回 HTTP 200。
- 输出目录下生成了 `*_ascend_pt` 原始数据目录。

## 5. 检查原始数据

在部署机器的普通 shell 中执行：

```bash
find /home/g00619970/moeoffload_multi/profile_run_0905 \
  -maxdepth 2 -type d -name '*_ascend_pt' -print
```

DP2 正常情况下通常会看到带不同 rank/worker 名称的多个目录。继续检查是否有实际文件：

```bash
find /home/g00619970/moeoffload_multi/profile_run_0905 \
  -maxdepth 4 -type f | head -n 50
```

如果没有生成 `*_ascend_pt` 目录，先不要执行离线解析，转到第 8 节排查。

## 6. 在服务进程外离线解析

必须从普通前台 shell 执行，不要在 vLLM worker 或其他 daemon 子进程中执行。

```bash
python3 -c 'from torch_npu.profiler.profiler import analyse; analyse(profiler_path="/home/g00619970/moeoffload_multi/profile_run_0905", max_process_number=8)'
```

解析完成后检查 JSON、CSV 或数据库文件：

```bash
find /home/g00619970/moeoffload_multi/profile_run_0905 \
  -type f \( -name '*.json' -o -name '*.csv' -o -name '*.db' \) \
  | head -n 100
```

说明：

- `profiler_path` 指向本轮 profiling 的父目录，不要直接指向服务日志文件。
- 如果解析机器 CPU 较少，可把 `max_process_number` 调整为 2 或 4。
- 不要把多个不同实验的原始数据长期混在同一个目录中。

## 7. 尝试解析本次已经采集的数据

本次日志使用的目录是：

```text
/home/g00619970/moeoffload_multi/prefile
```

可以先检查现有原始数据：

```bash
find /home/g00619970/moeoffload_multi/prefile \
  -maxdepth 2 -type d -name '*_ascend_pt' -print
```

如果目录存在，先尝试离线解析，不一定需要立刻重新采集：

```bash
python3 -c 'from torch_npu.profiler.profiler import analyse; analyse(profiler_path="/home/g00619970/moeoffload_multi/prefile", max_process_number=8)'
```

由于本次配置了 `max_iterations=3`，而且 DP0/DP1 的停止时间不一致，现有数据可能只包含请求最开始的几个 worker step。它可以用来确认解析链路，但不适合作为完整请求或两个 DP rank 的严格性能对比结果。

## 8. 常见问题排查

### 8.1 没有生成 `*_ascend_pt` 目录

依次检查：

1. `torch_profiler_dir` 是否为绝对路径。
2. 启动服务的用户是否有该目录的写权限。
3. `/start_profile` 是否返回 HTTP 200。
4. `/start_profile` 和 `/stop_profile` 之间是否确实执行了推理请求。
5. `MSMONITOR_USE_DAEMON` 或 `additional_config.msmonitor_use_daemon` 是否被启用。它不能与当前 torch profiler 同时使用。
6. 磁盘空间是否充足。

### 8.2 离线解析仍然失败

检查：

1. `analyse()` 是否在服务外的普通前台进程中运行。
2. 传入的是包含 `*_ascend_pt` 的父目录，而不是日志路径。
3. 目录是否为真实目录而不是软链接。
4. 当前 shell 是否正确加载了与服务相同的 CANN、PyTorch 和 torch_npu 环境。
5. 尝试降低并行解析进程数：

```bash
python3 -c 'from torch_npu.profiler.profiler import analyse; analyse(profiler_path="/home/g00619970/moeoffload_multi/profile_run_0905", max_process_number=2)'
```

### 8.3 数据不完整或只看到少量 decode step

检查：

1. 是否仍配置了 `max_iterations=3` 等较小上限。
2. 是否在推理请求返回之前调用了 `/stop_profile`。
3. DP2 的两个请求是否真正并发执行。
4. 是否复用了包含旧数据的 profiling 目录。

### 8.4 profiling 对性能影响很大

优先使用：

```json
{
  "torch_profiler_with_stack": false,
  "torch_profiler_with_memory": false
}
```

同时缩短固定请求的输出长度，仅采集足够分析的请求数量。profiling 数据用于定位瓶颈，不应直接作为无采集开销时的最终吞吐或时延成绩。

## 9. 如果希望从代码层彻底解决

当前无代码修改方案仍可能输出 `RECORD` 和 daemon parsing 提示。要让 schedule 和解析流程完整，需要修改 `vllm_ascend/profiler/torch_npu_profiler.py`：

1. 给 `tensorboard_trace_handler` 设置 `analyse_flag=False`，只在 worker 中落原始数据。
2. 给 `torch_npu.profiler.profile()` 接入 `schedule=`。
3. 在 `_profiler_step()` 中调用底层 `profiler.step()`。
4. 协调 `max_iterations` 与 schedule 周期，避免外层计数在 `RECORD` 阶段提前停止。
5. 增加 DP2、多 rank、手工停止和自动停止的测试。

代码修改完成后仍需要在真实 Ascend NPU 上验证：

- 两个 DP rank 都生成数据。
- 请求返回 HTTP 200。
- schedule 完整进入保存阶段。
- 离线解析能够生成可查看的结果文件。
- profiling 关闭后服务仍可继续正常处理请求。

