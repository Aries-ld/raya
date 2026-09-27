<h1 align="center">Raya</h1>
<p align="center"><strong>面向文字、图片和视频的 System 1 决策服务</strong></p>

<div align="center">

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Raya%20v1-FFD21E)](https://huggingface.co/yuyu199741/raya-decision-v1)
[![Base Model](https://img.shields.io/badge/Base%20Model-Qwen3.5--2B-7C3AED)](https://huggingface.co/Qwen/Qwen3.5-2B)
[![API Docs](https://img.shields.io/badge/Docs-API-2ea44f)](docs/api.md)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](pyproject.toml)

[English](README.md) | **简体中文**

[快速启动](#快速启动) · [接口文档](docs/api.md) · [调用示例](docs/examples/)

</div>

基于 Qwen3.5-2B 后训练模型，提供 **候选选择、等级评分与是/否概率判断**。每个问题通过一次模型前向得到候选概率，直接返回结构化 JSON，无需生成或解析自由文本。

## 评测结果

所有评测集均为 held-out（训练时从未见过）。每个问题通过单次前向传播和候选标签掩码投影完成决策。完整评测方法和细节见 [eval/README.md](eval/README.md)。

| 评测集 | 模态 | 任务类型 | 准确率 | ECE | p50 延迟 |
|---|---|---|---|---|---|
| GQA val | 图片 | 视觉问答 | **100.0%** | 0.015 | 104ms |
| KonIQ val | 图片 | 质量评分（5级） | **90.0%** | 0.194 | 59ms |
| NExT-QA val | 视频 | 视频内容理解 | **90.0%** | 0.150 | 155ms |
| POPE | 图片 | 对象存在判断（是/否） | **88.5%** | 0.061 | 60ms |
| AG News | 文本 | 新闻分类（4类） | **78.5%** | 0.082 | 45ms |
| BoolQ | 文本 | 是/否阅读理解 | **74.0%** | 0.169 | 86ms |
| SST-5 | 文本 | 情感分析（5级） | 39.5% | 0.210 | 45ms |
| MMLU | 文本 | 研究生级知识题 | 20.0% | 0.367 | 86ms |
| SuperGPQA | 文本 | 研究生级推理题 | 9.5% | 0.250 | 87ms |

**关键结论：**
- 图片和视频决策任务表现最强（88–100%）
- 高置信度预测的校准质量很好（POPE/GQA 的 ECE 仅 0.015–0.061）
- 纯知识推理题（MMLU、SuperGPQA）不是目标场景——Raya 是决策模型，不是知识问答模型
- 文本分类和阅读理解任务表现扎实（74–78%）

## 快速启动

需要 Python 3.12 或 3.13，以及 [uv](https://docs.astral.sh/uv/)。CUDA、Apple Silicon MPS 和 CPU 使用同一份代码。

```bash
git clone https://github.com/Aries-ld/raya.git
cd raya

uv sync --frozen --extra dev
uv run raya-download

cp .env.example .env
# 编辑 .env，将 RAYA_API_KEYS 替换为自己的随机密钥。

uv run raya-serve
```

已有 `.env` 时不要覆盖。客户端与服务端必须使用匹配的密钥；密钥、权重和本地产物均应留在 Git 之外。监听地址与端口通过 `RAYA_HOST` / `RAYA_PORT` 配置。

服务启动后检查就绪状态：

```bash
curl http://127.0.0.1:8000/readyz
# {"status":"ready"}
```

模型初始化完成前，就绪探针可能返回 503或尚无法连接。对外使用时由部署方提供可访问的服务地址与 API key；`127.0.0.1`只指向调用方本机。

## 调用示例

设置调用方环境变量。`RAYA_BASE_URL` 是服务根地址，不包含 `/v1`。

```bash
export RAYA_BASE_URL='http://127.0.0.1:8000'
export RAYA_API_KEY='replace-with-your-issued-key'

curl --fail-with-body --max-time 150 \
  "$RAYA_BASE_URL/v1/systemone" \
  -H "Authorization: Bearer $RAYA_API_KEY" \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "raya-decision-v1",
    "state": "我忘记了登录密码，无法登录账号。",
    "questions": {
      "intent": {
        "type": "choice",
        "instructions": "用户请求的主要意图是什么？",
        "criteria": {
          "reset_password": "找回或重置登录密码",
          "register": "注册新账号",
          "other": "其他请求"
        }
      }
    }
  }'
```

示意响应，概率和 token 数仅用于展示结构：

```json
{
  "model": "raya-decision-v1",
  "answers": {
    "intent": {
      "type": "choice",
      "choice": "reset_password",
      "probabilities": {"reset_password": 0.98, "register": 0.01, "other": 0.01},
      "confidence": 0.98
    }
  },
  "usage": {"input_tokens": 96, "output_tokens": 0}
}
```

返回调用方定义的选项 ID，无需解析生成文本。`confidence` 当前定义为最大候选概率；候选集合不完整时，高置信度也不保证判断正确。

完整的鉴权、图片/视频上传、三种问题类型、限制、错误码和重试说明见 **[API 接入文档](docs/api.md)**。

### 独立 Python 客户端

仅依赖 Python标准库，调用方不需要安装模型或推理依赖。

```bash
python3 docs/examples/call_raya.py --request docs/examples/text-request.json
python3 docs/examples/call_raya.py --request docs/examples/image-request.json --image ./photo.jpg
python3 docs/examples/call_raya.py --request docs/examples/video-request.json --video ./clip.mp4
```

图片与视频脚本会自动生成 Base64 data URL。HTTP 接口不接受本地路径；HTTPS 媒体域名需要由维护者加入白名单。

## 文档

| 入口 | 内容 |
| --- | --- |
| [API 接入文档](docs/api.md) | 鉴权、问题类型、媒体上传、响应格式、错误与重试 |
| [OpenAPI 定义](docs/openapi.json) | 机器可读的请求与响应协议 |
| [调用示例](docs/examples/) | 仅依赖 Python 标准库的客户端与三种模态请求模板 |
| [配置示例](.env.example) | 设备、模型路径、请求限制与解码参数 |

## 工作原理与能力

| 能力 | 行为 |
| --- | --- |
| 文字 | 自然语言或结构化 JSON 材料 |
| 图片 | 文字上下文与图片共同判断 |
| 视频 | 3–20 秒，Qwen `fps=1` 采样，仅处理画面 |
| `choice` | 从 2–26 个选项中选择，返回完整概率分布 |
| `score` | 对 2–10 个有序等级计算期望分数 |
| `noul` | 返回“是”的概率 |
| 多问题 | 一次请求 1–16 个独立问题 |
| 运行设备 | 自动按 CUDA → MPS → CPU 选择与故障降级 |

正式接口为 **`POST /v1/systemone`**：

```text
model + state + questions  →  model + answers + usage
```

协议采用 Jev System One 风格；图片与视频通过 `state.media` 扩展。Raya 使用自己的模型、候选概率定义和执行实现，不声称复现 Jev 的全部能力、校准结果或吞吐。当前不提供聊天 completion、流式文本生成或工具调用。

```text
材料与候选 → 多模态预处理 → 每问题一次前向 → 候选概率 → JSON 答案
```

## 模型信息

| 项目 | 来源 |
| --- | --- |
| Raya v1 权重 | [yuyu199741/raya-decision-v1](https://huggingface.co/yuyu199741/raya-decision-v1) |
| 后训练基座 | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) |
| 分词器 | 来自 Raya checkpoint |
| 视觉预处理器 | 来自 Qwen3.5-2B |
| 版本锁定 | 在 [download.py](raya_maas/download.py) 中固定模型仓库提交 |

首次下载的模型权重约 4.4 GB。下载清单写入 `models/manifest.json`；运行服务时只读取本地文件。

## 默认限制

| 项目 | 限制 |
| --- | --- |
| 输入 token | 每问题 4096，包含材料、候选和视觉 token |
| 问题 | 每请求 1–16 |
| 图片和视频 | 每请求最多 4 个，最多 1 个视频 |
| 文件 | 每个最多 64 MiB |
| HTTP 请求体 | 最多 96 MiB，含 Base64开销 |
| 视频 | 3–20 秒，含边界 |
| 源像素 | 图片或视频帧最多 2000 万 |
| 排队 | 最多 8 个等待请求 |
| 超时 | 默认 120 秒 |

视频采样遵循 Qwen `fps=1` 规则：通常至少 4 帧，10 秒约 10 帧，20 秒约 20 帧；奇数帧在编码时复制末帧补齐。它不保证每个整数秒恰好取一帧，也不处理音轨。

超限会明确报错，不静默截断。当前多问题逐个前向，媒体解码可复用，但没有共享前缀并行推理或动态批处理。每个请求的处理、前向和排队耗时通过 `X-Raya-*-Ms` 响应头提供。

## 运行与部署

- 自动设备模式使用 CUDA → MPS → CPU；显式指定设备时不自动切换。
- 自动精度为 CUDA BF16（不支持时 FP16）、MPS FP16、CPU FP32。
- 单 worker持有一份模型。请求超时后，已开始的前向可能继续运行；重试可能重复计算。
- 每帧像素预算保持可控；视频默认使用 8 线程 CPU 解码，根据关键帧分布选择 seek或顺序读取。
- 服务端凭据放在 `.env`，对外部署的 TLS和调用方配额由网关负责。

```bash
# CPU 容器
docker compose up -d --build

# NVIDIA 宿主机需安装驱动和 Container Toolkit
docker compose -f compose.yaml -f compose.cuda.yaml up -d --build
```

容器使用非 root 用户和只读模型挂载。macOS 的 MPS应使用原生 Python运行；Docker Desktop 内的 Linux 容器不能使用 MPS。Docker 构建和 CUDA 实机性能尚未在本项目当前开发环境验证。

## 本地验证

```bash
uv run ruff check raya_maas scripts tests docs/examples
uv run pytest -q

# 已启动服务的真实 HTTP 检查
python3 docs/examples/call_raya.py --request docs/examples/text-request.json
```

私人本地测试目录 `local_tests/`不纳入版本控制。公开调用示例位于 `docs/examples/`；图片和视频请由调用方自行提供。

现有验证覆盖协议、鉴权、输入限制、采样一致性、设备降级、请求队列和计时。小样本结果不代表整体准确率或生产 SLA；MPS 上的 20 秒视频当前仍可能超过 2 秒。具体数据与边界见下列报告。

- [综合验收](docs/validation.md)
- [1 fps 对比与性能](docs/fps1-validation.md)
- [视频链路审查](docs/video-audit.md)

## 项目结构

```text
raya_maas/        # 推理与HTTP服务
scripts/         # 媒体准备、检查与基准
tests/           # 自动化测试
docs/            # 接入规范与验证记录
docs/examples/   # 独立客户端与请求模板
models/          # 本地下载，不进Git
artifacts/       # 本地产物，不进Git
```

## 贡献

欢迎通过 [Issues](https://github.com/Aries-ld/raya/issues) 报告可复现的问题，或提交范围明确的 Pull Request。请说明设备、依赖版本、输入模态、预期和实际行为；共享请求样例前移除密钥与私人媒体。

修改前后请运行相关测试与格式检查。协议变更需同步更新 API 文档、请求示例和 OpenAPI 定义。

## 许可证

本仓库代码采用 [Apache License 2.0](LICENSE)。模型权重、基座和外部媒体不在此代码许可证的授权范围内，其使用条款以各自发布方说明为准。

## 相关项目

- [Hugging Face 上的 Raya 模型](https://huggingface.co/yuyu199741/raya-decision-v1)
- [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
- [TypeSafe System One API](https://docs.typesafe.ai/api)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
