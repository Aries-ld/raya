# Raya MaaS

Raya v1 多模态快速决策服务。输入上下文、问题和 2–26 个候选，单次模型前向后，仅在候选标签 token 上做 softmax。没有自回归生成，不调用 `generate()`。

模型：[`yuyu199741/raya-decision-v1`](https://huggingface.co/yuyu199741/raya-decision-v1)，基于 Qwen3.5-2B。视觉 processor 来自原始 Qwen 仓库；分词器来自 Raya。已移除原训练代码、数据集、训练文档；当前工程只负责下载、推理、服务及自测。

## 启动

需要 Python 3.12/3.13 和 [uv](https://docs.astral.sh/uv/)。首次下载约 4.4GB，启动后只读本地模型，服务运行无需访问 Hugging Face。

```bash
uv sync --frozen --extra dev
uv run raya-download
cp .env.example .env
# 将 .env 中 RAYA_API_KEYS 改为随机密钥；不要使用示例值上线。
uv run raya-serve
```

`raya-download` 固定模型和 processor 的 commit，支持 Hugging Face 缓存、断点重试。下载清单位于 `models/manifest.json`。需要镜像时在下载前设置 `HF_ENDPOINT`，私有仓库认证使用 `HF_TOKEN`，均不写入源码。

- 默认模型名：`raya-decision-v1`。
- 配置文件 `.env`，环境变量优先；`.env`、模型和测试媒体均不进 Git。
- `RAYA_DEVICE=auto` 按 **CUDA → MPS → CPU** 选择可用设备；加载或推理遇到后端不支持、显存不足、非有限数值时继续降级，后续请求沿用成功设备。显式指定设备时不降级。
- 自动精度：CUDA 支持 BF16 时使用 BF16，否则 FP16；MPS 使用 FP16；CPU 使用 FP32。
- 默认必须设置 Bearer API key，可用逗号分隔多个 key 进行轮换。只在本地临时测试时设置 `RAYA_ALLOW_ANONYMOUS=true`。
- 监听地址通过 `RAYA_HOST` 设置。默认代码配置是 `127.0.0.1`，部署示例为 `0.0.0.0`。公网部署应在反向代理配置 HTTPS 和按调用方限流。

## OpenAI SDK 调用

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key=os.environ["RAYA_TEST_API_KEY"],  # 与服务端某个 RAYA_API_KEYS 一致
)
response = client.chat.completions.create(
    model="raya-decision-v1",
    messages=[{"role": "user", "content": "用户说：帮我把明天下午3点的会议改到4点。意图是什么？"}],
    extra_body={
        "candidates": ["修改会议时间", "创建新会议", "取消会议", "查询会议详情"],
        "confidence_threshold": 0.6,
    },
)
print(response.choices[0].message.content)  # A
print(response.decision)  # 标签、选项、概率、置信度、门控、实际设备和耗时
```

`candidates` 是 Raya 扩展字段，通过 SDK 的 `extra_body` 传入。候选按顺序映射 A–Z，0 起始的 `index` 对应原数组。默认 `message.content` 返回标签，完整结果在顶层 `decision`。`response_format={"type":"json_object"}` 可让 `message.content` 返回可直接 `json.loads` 的决策对象。

纯标准客户端无需扩展字段：把候选放在最后一条 user 消息末尾，每行 `A. 选项`、`B. 选项`，标签连续，服务自动解析。末尾可附手册中的 `Answer with exactly one label from the options above.`。

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $RAYA_TEST_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"raya-decision-v1","messages":[{"role":"user","content":"Please cancel my reservation. What is the intent?"}],"candidates":["Create a reservation","Cancel a reservation","Check the weather"]}'
```

决策对象中的 `probabilities` 是**当前候选集合内**的归一化概率；`confidence` 是其中最大值。`needs_review` 为 `confidence < confidence_threshold`，不自动调用其他模型。手册中的校准结果是训练方提供的数据，本工程的功能自测不代表重新验证该校准效果。

### 图片和视频

图片采用 OpenAI 的 `image_url` 消息块；视频采用兼容服务常用的 `video_url` 扩展块，**不是 OpenAI Chat Completions 的标准视频字段**。视频仅分析画面，不转录音轨。

```python
import base64
from pathlib import Path

path = Path("tests/fixtures/m3bench_kitchen.mp4")
url = "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode()
response = client.chat.completions.create(
    model="raya-decision-v1",
    messages=[{"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": url}},
        {"type": "text", "text": "Where does this video take place?"},
    ]}],
    extra_body={"candidates": ["A kitchen", "A beach", "An office", "A street"]},
)
print(response.decision)
```

图片将上例的 `video_url` 换成 `image_url`，MIME 换成 `image/jpeg` 或 `image/png`。

默认只接收 base64 data URL，不读取客户端给出的本地路径。需要 HTTPS 媒体时配置 `RAYA_MEDIA_HOSTS=media.example.com`，必须是管理员控制的可信域名。禁用重定向、代理环境和私有/保留 IP；部署层仍应限制出站网络，避免受信域名 DNS 变化访问内部网络。`image_url.detail` 为兼容性接受，实际分辨率统一由服务端像素预算控制。

### 协议范围

| 接口或参数 | 行为 |
| --- | --- |
| `GET /v1/models`、`GET /v1/models/{id}` | OpenAI 模型列表/详情 |
| `POST /v1/chat/completions` | 文本、图片、视频候选决策 |
| `stream=true` | 前向完成后发送 role/content/stop SSE 块及 `[DONE]` |
| `stream_options.include_usage` | 最后单独发送 usage 块 |
| `response_format` | `text` 或 `json_object` |
| `temperature` | 仅接受 0、1 或 null，均不采样、不改变 softmax |
| `top_p`、`n` | 仅接受 1；top_p 可为 null |
| `max_tokens`、`max_completion_tokens` | 接受正整数用于客户端兼容，不影响决策 |
| `usage.completion_tokens` | 0：返回标签/JSON 为服务包装，没有生成 token |
| `/healthz`、`/readyz` | 无需鉴权；活性与就绪检查 |

不实现 Responses、工具调用、任意 JSON Schema、文本生成、logprobs、音频、候选之外的回答；未知参数返回 OpenAI 格式的 400 错误，不静默忽略。流式接口不会提前生成部分决策，也不会缩短模型计算时间。

## 视频优化

1. 从时长均匀取 **8 个时间点**，使用 PyAV/FFmpeg seek 到临近关键帧，只解码附近 GOP。
2. 仅对选中的帧做 RGB 转换，FFmpeg 在转换时按像素预算缩放，不将整个视频转成 RGB 数组。
3. 保留原始 fps 和采样时间位置作为 `VideoMetadata`；processor 使用 `do_sample_frames=False`，并传 `num_frames=8`、`fps=None`、`cap_pixels_per_frame=True`。
4. seek 解码失败时使用有时间/帧数预算的顺序解码，内存只保留 8 帧。
5. 只计算末位 logits，`logits_to_keep=1`、`use_cache=False`，不保留整段词表 logits 或 KV cache。

视频解码在 CPU 上完成，适用于 CUDA/MPS/CPU 推理机器。目前未引入未验证的 NVDEC/torchcodec GPU 解码。CUDA 上可后续验证与 torch 匹配的 `causal-conv1d`/`flash-linear-attention` 加速内核；基础安装使用正确但较慢的 PyTorch 实现。

两段 bench 的采样帧与顺序基线逐像素比较一致。60 秒 kitchen 样例的 3 次解码中位数：**2472.8ms → 304.0ms（约 8.1 倍）**；约 12 秒 living 样例：**783.4ms → 288.4ms（约 2.7 倍）**。这是本机解码结果，不是 CUDA 端到端延迟，也不保证所有编码/GOP 都有相同收益。

## 运行边界

默认限制：输入 4096 token（包含视觉展开）、最多 4 个媒体且其中最多 1 个视频、每个媒体 64MiB、HTTP 请求 96MiB、视频 60 秒、原始帧 2000 万像素。processor 图像预算 262144 像素，视频每帧 131072 像素。按场景在 `.env` 调整。

这些是本工程为延迟与内存设置的输入预算，不是模型本身的理论能力上限。文本还受 schema
限制：每个文本块最多 65536 字符、最多 32 条消息；结构化候选必须为 2–26 个，每项最多
2048 字符。总 token 限制计算完整 prompt，包含问题、上下文、候选及图片/视频展开后的 token；
超出预算会明确报错，不静默截断文字或视频时长。图片会缩小，视频固定采样 8 帧。

常用可调项：`RAYA_MAX_INPUT_TOKENS`、`RAYA_MAX_MEDIA_BYTES`、`RAYA_MAX_IMAGE_PIXELS`、
`RAYA_MAX_VIDEO_SECONDS`、`RAYA_IMAGE_MAX_PIXELS`、`RAYA_VIDEO_MAX_PIXELS`。
当前配置验证允许的 token 上限最多 32768，视频时长最多 600 秒；扩大预算会增加延迟/内存，
且仍需满足其他限制。65536/2048 字符和消息数限制在 `schemas.py` 中定义，不能通过环境变量修改。

单进程、单推理 worker 负责一份模型，等待队列默认 8；队满返回 429，含 `Retry-After`。默认请求超时 120 秒返回 504；已开始的 GPU 任务无法强行取消，会继续占用唯一 worker 直到完成，超时/断开的排队请求会被丢弃。不会因为 HTTP 超时而让两个模型前向重叠。不要为同一 GPU 启动多个 Uvicorn worker；扩容用独立实例/设备配合负载均衡。该版本不做动态批处理。

日志不记录请求正文、图片/视频或密钥。响应携带 `x-request-id`；`decision.timing_ms` 区分预处理与前向，视频另含解码时间和实际解码帧数。模型启动完成后才接收业务请求；启动期间就绪探针连接可能尚未建立。

## 容器

先在宿主机完成模型下载并创建 `.env`：

```bash
# CPU Linux（Docker Desktop 的 Linux 容器无法使用 macOS MPS）
docker compose up -d --build

# NVIDIA Linux，需要宿主机安装驱动和 NVIDIA Container Toolkit
# RAYA_DEVICE 保持 auto，CUDA 可用时自动使用
docker compose -f compose.yaml -f compose.cuda.yaml up -d --build
```

容器以非 root 用户运行，模型目录只读挂载，内置健康检查。不将权重或密钥打包到镜像。Mac 使用 MPS 时请直接运行 Python 服务。当前环境没有 Docker/CUDA，容器配置与 CUDA 实机性能尚未执行验证。

## 自测

### 不启动服务，直接测试模型

三个脚本直接加载本地权重，不发送 HTTP 请求、不要求 API key，默认 CUDA → MPS → CPU。
首次准备环境和下载模型后，可以断网执行。每次运行都重新加载模型，`model_load_ms` 单独记录
加载耗时，`timing_ms` 是该次推理耗时；首次推理还可能有后端初始化开销。

```bash
# 直接运行默认示例
uv run python scripts/test_text.py
uv run python scripts/test_image.py
uv run python scripts/test_video.py

# 换问题、候选与路径；路径带空格时加引号
uv run python scripts/test_text.py \
  --question "请取消明天的会议。这是什么意图？" \
  --candidates "修改会议" "创建会议" "取消会议"

uv run python scripts/test_image.py \
  --image "/path/to/photo.jpg" --question "图中主要是什么动物？" \
  --candidates "猫" "狗" "鸟" "没有动物"

uv run python scripts/test_video.py \
  --video "/path/to/clip.mp4" --question "视频主要发生在哪里？" \
  --candidates "厨房" "客厅" "街道" "海边"
```

也可以直接编辑每个脚本顶部的 `QUESTION`、`CANDIDATES`、`IMAGE_PATH` / `VIDEO_PATH`。
候选必须随问题一起调整，模型只能从候选中选择。输出包括所选标签/文本、各项概率、置信度、
实际设备、token 数、推理耗时；视频另有采样帧数、实际解码帧数和解码耗时。

可选参数：`--device cpu`、`--threshold 0.8`、`--output artifacts/my-test.json`、
`--max-input-tokens 8192`；视频脚本另支持 `--max-video-seconds 120`。
其余限制读取 `.env` / `RAYA_*` 环境变量，与服务推理引擎相同；本地测试没有 HTTP 请求体限制、
鉴权、排队或服务请求超时，仍受媒体大小、像素、时长、解码预算、schema 和 token 上限约束。

### 工程与 HTTP 接口测试

```bash
uv run ruff check raya_maas scripts tests
uv run pytest -q

# 本机已经备好样例；在另一台有原 bench 的机器上可以重建
uv run python scripts/prepare_fixtures.py --bench-root /path/to/mneme
# prepare_fixtures 需要 ffmpeg CLI；服务本身仅需 Python av 包

# 对已经启动的真实服务运行 OpenAI SDK 测试
RAYA_TEST_API_KEY=your-key uv run python scripts/smoke.py
uv run python scripts/benchmark_video.py \
  tests/fixtures/m3bench_kitchen.mp4 tests/fixtures/m3bench_living.mp4 --runs 3
```

文本样例为 mock；图片和视频来自已有 m3bench 派生素材，来源和 SHA256 在 `tests/fixtures/provenance.json`。媒体只用于本地测试，不提交进 Git。自动测试使用隔离的轻量模型桩；`scripts/smoke.py` 通过实际 HTTP/SDK 调用真实 Raya 权重，不使用模型桩。结果保存在 `artifacts/`，验收摘要见 [docs/validation.md](docs/validation.md)。

## 代码

- `raya_maas/app.py`：OpenAI 协议、鉴权、请求边界、SSE。
- `raya_maas/engine.py`：模型加载、设备降级、单次前向和候选概率。
- `raya_maas/media.py`：媒体读取、限制和稀疏视频解码。
- `raya_maas/worker.py`：有界队列、超时、取消和关闭。
- `raya_maas/schemas.py`：请求验证与训练格式 prompt 渲染。
- `uv.lock`：跨平台依赖锁定；模型及 processor revision 由下载模块固定。

协议依据：[OpenAI Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)、[Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)、[Transformers 视频处理](https://github.com/huggingface/transformers/blob/main/docs/source/en/main_classes/video_processor.md)。模型的具体候选标签与 prompt 规则以提供的 Raya v1 手册为准。
