# Raya · Multimodal Decision MaaS

**多模态候选决策服务：输入材料与问题，返回可执行的结构化判断。**<br>
**A multimodal decision service: provide context and typed questions, receive structured answers.**

[模型 / Model](https://huggingface.co/yuyu199741/raya-decision-v1) · [基座 / Base model](https://huggingface.co/Qwen/Qwen3.5-2B) · [接口文档 / API guide](docs/api.md) · [OpenAPI](docs/openapi.json) · [调用示例 / Client examples](docs/examples/)

Raya 基于 Qwen3.5-2B 后训练权重，为文字、图片和短视频提供候选选择、等级评分与是/否概率判断。每个问题执行一次模型前向，通过候选标签的分数计算概率，并由服务代码组装 JSON；不进行自回归文本生成。

Raya serves a post-trained Qwen3.5-2B checkpoint for candidate selection, ordinal scoring, and yes/no probability judgments over text, images, and short videos. Each question uses one model forward pass. Candidate-label scores determine the probabilities; application code assembles the JSON response without autoregressive text generation.

本仓库聚焦模型下载、推理服务、协议校验和验证工具。训练代码、模型权重、原始测试媒体与密钥不随当前代码分发。

This repository focuses on model acquisition, inference serving, request validation, and verification tools. Training code, model weights, raw test media, and credentials are not distributed with the current source tree.

## 模型信息 / Model artifacts

| 项目 / Item | 来源 / Source |
| --- | --- |
| Raya v1 权重 / Decision checkpoint | [yuyu199741/raya-decision-v1](https://huggingface.co/yuyu199741/raya-decision-v1) |
| 后训练基座 / Base model | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) |
| 分词器 / Tokenizer | 来自 Raya checkpoint / Loaded from the Raya checkpoint |
| 视觉预处理器 / Vision processor | 来自 Qwen3.5-2B / Loaded from Qwen3.5-2B |
| 版本锁定 / Revision pinning | 下载模块固定仓库 commit / Repository commits pinned in [download.py](raya_maas/download.py) |

首次下载的模型权重约4.4GB。下载清单写入 `models/manifest.json`；运行服务时只读取本地文件。

The initial model weights download is approximately 4.4GB. Downloaded revisions are recorded in `models/manifest.json`; serving uses local files only.

## 能力与协议 / Capabilities and contract

| 能力 / Capability | 行为 / Behavior |
| --- | --- |
| 文字 / Text | 自然语言或结构化 JSON 材料 / Natural-language or structured JSON context |
| 图片 / Images | 文字上下文与图片共同判断 / Decisions over text context and image inputs |
| 视频 / Video | 3–20秒，Qwen `fps=1` 采样，仅处理画面 / 3–20 seconds, Qwen `fps=1` sampling, visual content only |
| `choice` | 从2–26个选项中选择，返回完整概率分布 / Select from 2–26 options with a full probability distribution |
| `score` | 对2–10个有序等级计算期望分数 / Return an expected score over 2–10 ordered levels |
| `noul` | 返回“是”的概率 / Return the probability of yes |
| 多问题 / Multiple questions | 一次请求1–16个独立问题 / 1–16 independent questions per request |
| 运行设备 / Devices | 自动按 CUDA → MPS → CPU 选择与故障降级 / Automatic CUDA → MPS → CPU selection and backend fallback |

正式接口为 **`POST /v1/systemone`**：

```text
model + state + questions  →  model + answers + usage
```

协议采用 Jev System One 风格；图片与视频通过 `state.media` 扩展。Raya 使用自己的模型、候选概率定义和执行实现，不声称复现 Jev 的全部能力、校准结果或吞吐。当前不提供聊天 completion、流式文本生成或工具调用。

The API follows the Jev System One request/response style, with images and videos added through `state.media`. Raya uses its own checkpoint, confidence definition, and execution implementation; it does not claim to reproduce Jev's full capabilities, calibration, or throughput. Chat completions, streaming text generation, and tool calling are not provided.

## 快速启动 / Quick start

需要 Python 3.12或3.13，以及 [uv](https://docs.astral.sh/uv/)。CUDA、Apple Silicon MPS 和 CPU 使用同一份代码。

Requires Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/). The same codebase supports CUDA, Apple Silicon MPS, and CPU.

```bash
git clone https://github.com/Aries-ld/raya.git
cd raya

uv sync --frozen --extra dev
uv run raya-download

cp .env.example .env
# 编辑 .env，将 RAYA_API_KEYS 替换为自己的随机密钥。
# Edit .env and replace RAYA_API_KEYS with your own random secret.

uv run raya-serve
```

已有 `.env` 时不要覆盖。客户端与服务端必须使用匹配的密钥；密钥、权重和本地产物均应留在 Git 之外。监听地址与端口通过 `RAYA_HOST` / `RAYA_PORT` 配置。

Do not overwrite an existing `.env`. Clients must use a key accepted by the server. Keep keys, weights, and local artifacts out of Git. Configure the bind address and port with `RAYA_HOST` and `RAYA_PORT`.

服务启动后检查就绪状态：<br>
Check readiness after startup:

```bash
curl http://127.0.0.1:8000/readyz
# {"status":"ready"}
```

模型初始化完成前，就绪探针可能返回503或尚无法连接。对外使用时由部署方提供可访问的服务地址与API key；`127.0.0.1`只指向调用方本机。

Before initialization completes, readiness checks may return 503 or fail to connect. For remote use, the operator must provide a reachable service URL and API key; `127.0.0.1` always refers to the caller's own machine.

## 调用示例 / API example

设置调用方环境变量。`RAYA_BASE_URL`是服务根地址，不包含`/v1`。

Set client environment variables. `RAYA_BASE_URL` is the service root URL, without `/v1`.

```bash
export RAYA_BASE_URL='http://127.0.0.1:8000'
export RAYA_API_KEY='replace-with-your-issued-key'

curl --fail-with-body --max-time 150 \
  "$RAYA_BASE_URL/v1/systemone" \
  -H "Authorization: Bearer $RAYA_API_KEY" \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "raya-decision-v1",
    "state": "I forgot my password and cannot sign in.",
    "questions": {
      "intent": {
        "type": "choice",
        "instructions": "What is the main request?",
        "criteria": {
          "reset_password": "Recover or reset a login password",
          "register": "Create a new account",
          "other": "A different request"
        }
      }
    }
  }'
```

示意响应，概率和token数仅用于展示结构：<br>
Illustrative response; probabilities and token counts below are examples, not a measured guarantee:

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

返回调用方定义的选项ID，无需解析生成文本。`confidence`当前定义为最大候选概率；候选集合不完整时，高置信度也不保证判断正确。

The response uses your option IDs, so no generated prose needs to be parsed. `confidence` is currently the maximum candidate probability; a high value does not guarantee correctness when the candidate set is incomplete.

完整的鉴权、图片/视频上传、三种问题类型、限制、错误码和重试说明见 **[API接入文档](docs/api.md)**。

See the **[API integration guide](docs/api.md)** for authentication, media inputs, all three question types, limits, errors, and retry behavior. A machine-readable [OpenAPI schema](docs/openapi.json) is included.

### 独立 Python 客户端 / Standalone Python client

仅依赖Python标准库，调用方不需要安装模型或推理依赖。

The example client uses only the Python standard library. Callers do not need model weights or inference dependencies.

```bash
python3 docs/examples/call_raya.py --request docs/examples/text-request.json
python3 docs/examples/call_raya.py --request docs/examples/image-request.json --image ./photo.jpg
python3 docs/examples/call_raya.py --request docs/examples/video-request.json --video ./clip.mp4
```

图片与视频脚本会自动生成Base64 data URL。HTTP接口不接受本地路径；HTTPS媒体域名需要由维护者加入白名单。

For media requests, the client converts local files to Base64 data URLs. The HTTP API does not accept local file paths; remote HTTPS media hosts must be allowlisted by the operator.

## 默认限制 / Default limits

| 项目 / Item | 限制 / Limit |
| --- | --- |
| 输入token / Input tokens | 每问题4096，包含材料、候选和视觉token / 4,096 per question, including context, options, and visual tokens |
| 问题 / Questions | 每请求1–16 / 1–16 per request |
| 图片和视频 / Media | 每请求最多4个，最多1个视频 / Up to 4 items, at most 1 video |
| 文件 / File size | 每个最多64MiB / Up to 64MiB per item |
| HTTP请求体 / HTTP body | 最多96MiB，含Base64开销 / Up to 96MiB, including Base64 overhead |
| 视频 / Video | 3–20秒，含边界 / 3–20 seconds, inclusive |
| 源像素 / Source pixels | 图片或视频帧最多2000万 / Up to 20 million pixels per image or video frame |
| 排队 / Queue | 最多8个等待请求 / Up to 8 waiting requests |
| 超时 / Timeout | 默认120秒 / 120 seconds by default |

视频采样遵循Qwen `fps=1`规则：通常至少4帧，10秒约10帧，20秒约20帧；奇数帧在编码时复制末帧补齐。它不保证每个整数秒恰好取一帧，也不处理音轨。

Video sampling follows Qwen's `fps=1` policy: normally at least 4 frames, around 10 for 10 seconds and 20 for 20 seconds. An odd frame count is padded by repeating the last frame during encoding. Sampling is not strictly aligned to integer-second timestamps, and audio is not processed.

超限会明确报错，不静默截断。当前多问题逐个前向，媒体解码可复用，但没有共享前缀并行推理或动态批处理。每个请求的处理、前向和排队耗时通过`X-Raya-*-Ms`响应头提供。

Limit violations return explicit errors rather than silent truncation. Questions currently run sequentially; decoded media is reused, but shared-prefix parallel inference and dynamic batching are not implemented. Per-request processing, forward-pass, and queue times are exposed through `X-Raya-*-Ms` response headers.

## 运行与部署 / Runtime and deployment

- 自动设备模式使用 CUDA → MPS → CPU；显式指定设备时不自动切换。<br>
  Automatic mode selects CUDA → MPS → CPU; explicitly selecting a device disables device fallback.
- 自动精度为 CUDA BF16（不支持时FP16）、MPS FP16、CPU FP32。<br>
  Automatic precision uses CUDA BF16 when supported, otherwise FP16; MPS FP16; CPU FP32.
- 单worker持有一份模型。请求超时后，已开始的前向可能继续运行；重试可能重复计算。<br>
  A single worker owns the model. An in-flight forward pass may continue after a request times out; retries may repeat computation.
- 每帧像素预算保持可控；视频默认使用8线程CPU解码，根据关键帧分布选择seek或顺序读取。<br>
  Per-frame pixel budgets are bounded. Video decoding defaults to 8 CPU threads and selects seeking or sequential decoding based on keyframes.
- 服务端凭据放在`.env`，对外部署的TLS和调用方配额由网关负责。<br>
  Server credentials belong in `.env`; use a gateway for public TLS termination and per-client quotas.

```bash
# CPU container / CPU容器
docker compose up -d --build

# NVIDIA host with drivers and NVIDIA Container Toolkit
# NVIDIA宿主机需安装驱动和Container Toolkit
docker compose -f compose.yaml -f compose.cuda.yaml up -d --build
```

容器使用非root用户和只读模型挂载。macOS的MPS应使用原生Python运行；Docker Desktop内的Linux容器不能使用MPS。Docker构建和CUDA实机性能尚未在本项目当前开发环境验证。

Containers use a non-root user and a read-only model mount. Use native Python for macOS MPS; Linux containers in Docker Desktop cannot use MPS. Docker builds and CUDA hardware performance have not been verified in the current development environment.

## 本地验证 / Local verification

```bash
uv run ruff check raya_maas scripts local_tests tests docs/examples
uv run pytest -q

# 已启动服务的真实HTTP检查 / Real HTTP checks against a running service
uv run python scripts/smoke.py
```

不启动HTTP服务时，也可以运行`local_tests/run_text.py`、`run_image.py`或`run_video.py`，在PyCharm中直接Run/Debug。编辑同目录JSON定义材料、问题和候选。媒体文件不随仓库分发，请提供自己的图片/视频并更新相对路径；默认文字示例无需媒体。

For offline testing, run `local_tests/run_text.py`, `run_image.py`, or `run_video.py`, including through PyCharm Run/Debug. Edit the adjacent JSON files to define context, questions, and options. Media files are not distributed with the repository: provide your own files and update the relative paths. The default text example requires no media.

现有验证覆盖协议、鉴权、输入限制、采样一致性、设备降级、请求队列和计时。小样本结果不代表整体准确率或生产SLA；MPS上的20秒视频当前仍可能超过2秒。具体数据与边界见下列报告。

Verification covers the protocol, authentication, input limits, sampling consistency, device fallback, request scheduling, and timing. Small-sample results are not overall accuracy claims or production SLAs; 20-second videos on MPS can still take more than 2 seconds. See the reports for measured results and limitations.

- [综合验收 / Verification record](docs/validation.md)
- [1fps对比与性能 / 1fps evaluation and latency](docs/fps1-validation.md)
- [视频链路审查 / Video pipeline audit](docs/video-audit.md)

## 项目结构 / Repository layout

```text
raya_maas/       # 推理与HTTP服务 / Inference and HTTP service
local_tests/     # 可编辑JSON与独立入口 / Editable JSON requests and local runners
scripts/         # 媒体准备、检查与基准 / Media preparation, checks, benchmarks
tests/           # 自动化测试 / Automated tests
docs/            # 接入规范与验证记录 / Integration guide and verification records
docs/examples/   # 独立客户端与请求模板 / Standalone client and request templates
models/          # 本地下载，不进Git / Downloaded locally, excluded from Git
artifacts/       # 本地产物，不进Git / Local outputs, excluded from Git
```

## 贡献 / Contributing

欢迎通过 [Issues](https://github.com/Aries-ld/raya/issues) 报告可复现的问题，或提交范围明确的Pull Request。请说明设备、依赖版本、输入模态、预期和实际行为；共享请求样例前移除密钥与私人媒体。

Use [Issues](https://github.com/Aries-ld/raya/issues) for reproducible bug reports, or submit a focused pull request. Include the device, dependency versions, input modality, expected behavior, and actual result. Remove credentials and private media from shared examples.

修改前后请运行相关测试与格式检查。协议变更需同步更新API文档、请求示例和OpenAPI定义。

Run the relevant tests and formatting checks before submitting changes. Keep the API guide, request examples, and OpenAPI definition in sync with protocol changes.

## 许可证 / License

本仓库代码采用 [Apache License 2.0](LICENSE)。模型权重、基座和外部媒体不在此代码许可证的授权范围内，其使用条款以各自发布方说明为准。

The source code is licensed under the [Apache License 2.0](LICENSE). This code license does not grant rights to model weights, the base model, or external media; those artifacts are governed by their respective publishers' terms.

## 相关项目 / References

- [Raya model on Hugging Face](https://huggingface.co/yuyu199741/raya-decision-v1)
- [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
- [TypeSafe System One API](https://docs.typesafe.ai/api)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
