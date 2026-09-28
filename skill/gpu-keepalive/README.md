# GPU 健康分维护

在空闲 GPU 上跑真实训练任务，保持有效利用率，避免被退卡。

## 健康分定义（对齐 RAI 平台规则）

### 计算规则

**训练健康分** = 基于 ETTR（Estimated Training Time Ratio）× SM 利用率（或 MFU），按任务类别/卡型/模型大小分桶划定 75 分位基线，统一折算到 0-100 分。

核心逻辑：
- **不是简单的 GPU 利用率高就行**——通信阻塞、空转 kernel、无效计算都可能表现为高 Util，但不算有效计算
- 健康分看的是**有效计算占比**：GPU 真正在做有意义的矩阵运算的时间比例
- batch_size 越大，MFU 越高（每次前向/反向的计算量更大，通信和数据加载的相对开销更小）

**队列健康分** = Σ(任务H200标准卡时 × 任务健康分) / Σ(总H200标准卡时)

### 考核目标（9-10月双月）

| 指标 | M9W3-4 | M10W1-2 | M10W3-4 |
|---|---|---|---|
| 训练健康分 | ≥ 62 | ≥ 63 | ≥ 65 |
| 推理健康分 | ≥ 53 | ≥ 59 | ≥ 65 |

### 本 skill 的准出条件

训练任务持续运行，直到以下任一条件满足才结束：
1. **用户显式调用 `stop.py`** —— 唯一的正常停止方式
2. **所有 GPU 被其他任务占用** —— 脚本自动检测并退出，不再占用资源
3. **训练进程连续异常退出超过阈值** —— 防止死循环重启

**不会因为以下原因停止：**
- loss 收敛（训练目标不是收敛，是保持有效计算）
- 单轮训练结束（supervisor 自动重启下一轮）
- 临时性错误（supervisor 自动重试）

## 快速开始

```bash
# 启动（自动检测空闲卡，每张卡独立跑训练）
python start.py start

# 指定卡
python start.py start --gpus 4,5,6,7

# 查看状态
python start.py status

# 停止（唯一的正常停止方式）
python stop.py
```

## 工作原理

### 1. 检测空闲 GPU

```
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu
```

显存占用 < 80GB 的卡视为空闲。每张空闲卡独立启动一个训练进程（不依赖多卡 DDP，避免异构卡兼容问题）。

### 2. 自适应 batch_size

| 显存 | batch_size |
|---|---|
| ≥ 80GB | 4 |
| ≥ 40GB | 2 |
| < 40GB | 1 |

### 3. Supervisor 永不停止

每张卡的训练进程由一个 supervisor 进程监控：
- 训练正常退出（loss 收敛）→ supervisor 等待 10 秒后自动重启
- 训练异常退出（OOM/错误）→ supervisor 等待 10 秒后自动重启
- 只有 `stop.py` 会 kill supervisor 和训练进程

### 4. 健康分监控

训练日志写在 `logs/keepalive_gpu{N}.log`，可以通过日志确认：
- loss 是否正常下降（有效计算的证明）
- 每步耗时是否稳定（没有通信阻塞或数据加载瓶颈）

## 使用自己的训练脚本

```bash
python start.py start \
  --train-script /path/to/your/train.py \
  --model-dir /path/to/your/model \
  --data-files /path/to/data1.jsonl /path/to/data2.jsonl \
  --venv-python /path/to/your/venv/bin/python
```

## 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--gpus` | 指定 GPU 编号（逗号分隔），不填则自动检测空闲卡 | 自动检测 |
| `--train-script` | 训练脚本路径 | Raya 全量 FT |
| `--model-dir` | 模型目录 | Raya rlcd_v1 |
| `--data-files` | 训练数据文件（可多个） | Raya 训练集 |
| `--media-root` | 媒体文件根目录 | /data/temp/raya_scratch |
| `--venv-python` | Python 虚拟环境路径 | Raya venv |

## 注意事项

- 训练任务是**真实训练**（loss 会正常下降），不是空转
- 每张卡独立运行，互不影响——某张卡被占用了，其他卡不受影响
- 如果卡被别人占用了（显存不够），脚本会自动跳过
- supervisor 保证训练永不停止，除非显式调用 `stop.py`
- 日志按卡分开写在 `logs/keepalive_gpu{N}.log`
