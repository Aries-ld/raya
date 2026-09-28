# GPU 健康分维护

在空闲 GPU 上跑真实训练任务，保持有效利用率，避免被退卡。

## 健康分定义（对齐 RAI 平台规则）

**训练健康分** = 基于 ETTR × SM 利用率（或 MFU），按任务类别/卡型/模型大小分桶划定 75 分位基线，统一折算到 0-100 分。

核心逻辑：**不是简单的 GPU 利用率高就行**——通信阻塞、空转 kernel、无效计算都可能表现为高 Util，但不算有效计算。健康分看的是**有效计算占比**。

**考核目标**（9-10月双月）：训练健康分 ≥ 62 → 63 → 65

## 快速开始

```bash
# 启动（自动检测空闲卡，自动准备模型和数据）
python start.py start

# 指定卡
python start.py start --gpus 4,5,6,7

# 查看状态
python start.py status

# 停止
python start.py stop
```

## 设计原则

### 1. 路径可配置，不硬编码

- `--project-dir` 指定项目目录，不填则自动检测
- 所有数据/模型/日志路径都基于 project-dir 推导
- 不同容器的云盘路径不同，不影响使用

### 2. 自动准备模型和数据

- 模型：从 HuggingFace 下载（走 hf-mirror，内网可达）
- 数据集：自动跑数据准备脚本，hf-mirror 可达的自动下，外网拿不到的**自动跳过**（不会阻塞）
- 已有数据直接用，不重复下载

### 3. 异构 GPU 适配

| 显存 | batch_size | 说明 |
|---|---|---|
| ≥ 80GB | 4 | H20/A100 等大显存卡 |
| ≥ 40GB | 2 | A10/V100 等中显存卡 |
| ≥ 20GB | 1 | T4 等小显存卡 |
| < 20GB | 1 | 小显存也能跑，只是慢 |

每张空闲卡**独立跑一个训练进程**（不依赖多卡 DDP），不同规格的卡互不影响。

### 4. 永不停止

supervisor 模式：训练正常退出/异常退出都自动重启，只有显式调用 `stop` 才会真正停止。

### 5. 准出条件

训练任务持续运行，直到以下任一条件满足才结束：
1. **用户显式调用 `stop`** —— 唯一的正常停止方式
2. **所有 GPU 被其他任务占用** —— 自动检测并退出
3. **训练进程连续异常退出超过阈值** —— 防止死循环重启

**不会因为以下原因停止：**
- loss 收敛（目标是保持有效计算，不是收敛）
- 单轮训练结束（supervisor 自动重启）
- 临时性错误（supervisor 自动重试）

## 使用自己的训练脚本

```bash
python start.py start \
  --project-dir /path/to/your/project \
  --train-script /path/to/your/train.py \
  --model-dir /path/to/your/model \
  --data-files /path/to/data1.jsonl /path/to/data2.jsonl \
  --venv-python /path/to/your/venv/bin/python
```

## 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--gpus` | 指定 GPU 编号（逗号分隔），不填则自动检测空闲卡 | 自动检测 |
| `--project-dir` | 项目目录 | 自动检测 |
| `--model-id` | 模型 ID | Qwen/Qwen3.5-2B |
| `--train-script` | 训练脚本路径 | 项目内 train_full_ft.py |
| `--model-dir` | 模型目录 | 项目内 checkpoints/full_ft_v1 |
| `--data-files` | 训练数据文件（可多个） | 项目内训练集 |
| `--venv-python` | Python 虚拟环境路径 | 项目内 .venv |

## 注意事项

- 训练任务是**真实训练**（loss 会正常下降），不是空转
- 每张卡独立运行，互不影响——某张卡被占用了，其他卡不受影响
- 数据集拉取失败（外网不可达）会自动跳过，不会阻塞整体启动
- supervisor 保证训练永不停止，除非显式调用 `stop`
- 日志按卡分开写在 `logs/keepalive_gpu{N}.log`
