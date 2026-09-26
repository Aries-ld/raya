# Raya 决策 MaaS

Raya v1 是基于 Qwen3.5-2B 的多模态候选决策模型。服务采用 **Jev System One 风格的决策协议**：调用方显式提供待判断材料、问题和候选/等级，返回类型确定的答案与概率。每个问题只做一次前向，不进行自回归生成。

正式接口：`POST /v1/systemone`。请求顶层为 `model / state / questions`，响应顶层为 `model / answers / usage`。本工程不再提供聊天接口或聊天 SDK 兼容层。训练代码、数据集已移除，只保留下载、推理、服务和自测。

## 先直接测试，不启动服务

**反复测试推荐运行 `local_tests/run_session.py`（或 PyCharm 配置 `Raya Session`）。**
首次加载模型并运行图片测试后，进程保持运行：修改 JSON/媒体并保存，在 Run/Debug 控制台按回车
即可重新推理；输入 `text`、`image`、`video` 可切换模态，复用同一模型，输入 `q` 退出。
每轮都重读请求和媒体文件，输入错误可修复后重试，模型不重新加载。
不要点击 IDE 的重新运行按钮：那会新建进程，仍需重新加载。改 Python 代码或模型配置后需要重启。

PyCharm 解释器选本项目 `.venv/bin/python`。在 `local_tests/` 中编辑 JSON，再对相应脚本右键 Run / Debug。无需启动参数或 API key。

| 运行脚本 | 编辑的完整请求 | 默认媒体 |
| --- | --- | --- |
| `local_tests/run_text.py` | `local_tests/text.json` | 文字在 `state` 中；示例含 choice/noul/score 三个问题 |
| `local_tests/run_image.py` | `local_tests/image.json` | `local_tests/image.jpg` |
| `local_tests/run_video.py` | `local_tests/video.json` | `local_tests/video.mp4`，m3bench 的 30 秒片段 |

每份 JSON 中都能直接看到 `state`、`questions.*.instructions` 和 `criteria`。替换媒体可保持文件名不变；换问题就同时调整判断标准。在 `request = load_request_file(...)` 和 `return result` 打断点可分别检查完整请求与响应。

本地 JSON 用 `file:./image.jpg` / `file:./video.mp4` 引用文件，测试脚本先转为 base64 data URL，再交给与 HTTP 相同的 `SystemOneRequest` 和推理路径。**HTTP 接口禁止读取服务端本地文件**。相对路径相对于 JSON 所在目录，模型和 `.env` 相对于项目根目录，不依赖 IDE 工作目录。

```bash
uv run python local_tests/run_text.py
uv run python local_tests/run_image.py
uv run python local_tests/run_video.py
```

三个单次运行脚本仍在退出时释放模型；重复调试用 `run_session.py` 保持模型驻留。
决策响应的字段不变。交互入口还会输出操作提示；本地模型加载时间、设备、各问题前向与视频解码
耗时写入 `artifacts/local-decision-diagnostics.json`，每次覆盖。`model_reused=true` 和
`model_load_ms=0` 表示后续推理复用了本次进程的模型，`session_request_number` 表示会话内请求序号。
也可在调试器查看 `session.engine.last_diagnostics`。

## 启动服务

Python 3.12/3.13；首次权重下载约 4.4GB。已准备好本地环境时直接 `uv run raya-serve`。

```bash
uv sync --frozen --extra dev
uv run raya-download
# 新环境才执行复制，已有 .env 请保留
cp .env.example .env
# 将 RAYA_API_KEYS 改为随机密钥
uv run raya-serve
```

模型为 `yuyu199741/raya-decision-v1`。视觉 processor 来自 Qwen3.5-2B，分词器来自 Raya；两者 commit 固定在下载模块，下载清单在 `models/manifest.json`。启动后仅使用本地模型。

设备自动按 CUDA → MPS → CPU 选择，并在后端不支持、OOM、非有限数值时降级；显式设置 `RAYA_DEVICE` 时只使用指定设备。默认精度 CUDA BF16（不支持则 FP16）、MPS FP16、CPU FP32。

## 决策协议

请求中的材料和问题分开，业务代码自行命名问题 ID 与选项 ID：

```json
{
  "model": "raya-decision-v1",
  "state": {"message": "我忘记了登录密码，无法登录账号。"},
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "用户请求的主要意图是什么？",
      "criteria": {
        "reset_password": "重置登录密码",
        "register": "注册新账号",
        "close_account": "注销账号"
      }
    }
  }
}
```

下面是**示意响应**，数值不是对上例的实测承诺：

```json
{
  "model": "raya-decision-v1",
  "answers": {
    "intent": {
      "type": "choice",
      "choice": "reset_password",
      "probabilities": {"reset_password": 0.98, "register": 0.01, "close_account": 0.01},
      "confidence": 0.98
    }
  },
  "usage": {"input_tokens": 100, "output_tokens": 0}
}
```

返回的是调用方定义的 `reset_password`，内部 A/B/C 标签不暴露为协议答案。缺少 `state`、`questions`、`instructions` 或 choice/score 的 `criteria` 会返回 422；只有媒体、没有问题的请求不会被执行。

三种问题：

| type | 调用方必须给什么 | 返回 |
| --- | --- | --- |
| `choice` | `instructions` + `criteria` 对象（选项 ID → 描述） | `choice`、所有选项的 `probabilities`、`confidence` |
| `score` | `instructions` + 从低到高的 `criteria` 数组 | `score`（Σ索引×概率）、`legend`、`probabilities`、`confidence` |
| `noul` | `instructions`；可选 `criteria.true` / `criteria.false` | `noul`，表示“是”的概率；候选在内部固定为是/否 |

同一请求可混合多个问题，响应 `answers` 使用原问题 ID，问题互相独立。`state` 支持文字、JSON 对象/数组；`instructions` 和描述也支持结构化 JSON。详见 **[完整接口规范](docs/api.md)** 与 [机器可读 OpenAPI](docs/openapi.json)。

### 文字 + 图片 / 视频

顶层字段仍与 Jev 一致；Raya 在 `state` 内扩展 `text` 和 `media`：

```python
import base64
import os
from pathlib import Path
import httpx

image_url = "data:image/jpeg;base64," + base64.b64encode(
    Path("local_tests/image.jpg").read_bytes()
).decode()
request = {
    "model": "raya-decision-v1",
    "state": {
        "text": "观察图片中右侧人物。",
        "media": [{"type": "image", "url": image_url}],
    },
    "questions": {
        "held_object": {
            "type": "choice",
            "instructions": "右侧人物手里拿着什么？",
            "criteria": {"basketball": "篮球", "laptop": "笔记本电脑", "cup": "水杯"},
        },
    },
}
response = httpx.post(
    "http://127.0.0.1:8000/v1/systemone",
    headers={"Authorization": "Bearer " + os.environ["RAYA_API_KEY"]},
    json=request, timeout=120,
)
response.raise_for_status()
print(response.json()["answers"]["held_object"])
```

视频用 `type: "video"` 和 `data:video/mp4;base64,...`；同样必须有明确问题和候选/等级。只分析画面，不识别音轨。Jev 官方当前仅支持文本，所以这个多模态 state 是 Raya 扩展，不能原样发到 Jev。

## 限制与语义差异

| 项目 | Raya 当前约束 |
| --- | --- |
| choice | 2–26 个选项；Jev 文档允许最多 255，Raya v1 标签空间暂不支持 |
| score | 2–10 个等级，分数索引从 0 开始 |
| questions | 每请求 1–16 个 |
| token | 每个问题的完整输入 ≤4096（含 state、问题、候选和视觉展开） |
| 文本 | 完整内部文本 ≤65536 字符；候选渲染后 ≤2048 字符 |
| 媒体 | 默认最多 4 个，最多 1 个视频；每个 ≤64MiB |
| 图片/视频帧 | 原始像素 ≤2000 万；预处理图片预算 262144 像素、视频每帧 131072 |
| 视频 | 默认 ≤60 秒，固定抽取 8 帧 |
| HTTP 请求体 | ≤96MiB（包括 base64 开销） |

这些是工程预算，可按 [配置](raya_maas/config.py) 中的 `RAYA_*` 调整，受字段验证上限约束。超限明确报错，不静默截断文字或视频时长。

Raya 的 `confidence=max(probabilities)`。Jev 官方只说明 confidence 从分布计算，未在所查文档公开精确公式，所以**协议结构对齐不等于置信度数值或校准行为相同**。`noul` / `score` 由 Raya 候选分布映射，未另外宣称经过 Jev 的专项校准。门控阈值由调用方设置。

多问题目前逐个前向，媒体仅下载/解码一次；没有 Jev 的共享 state 并行前向优化。`usage.input_tokens` 是各次实际前向输入 token 之和，重复 state 会重复计数；没有生成步骤，因此 `output_tokens=0`。不把这些模型能力差异伪装成全量 Jev 实现。

## 视频性能与运行

PyAV/FFmpeg 对 8 个均匀时间点 seek，只解码附近 GOP，仅对选中帧缩放和 RGB 转换；失败时回退到有帧数/时间预算的顺序解码。保留时间元数据，processor 不再次采样，并限制每帧像素。只保留末位 logits，不用 KV cache。

此前相同采样帧逐像素一致的对比：60 秒 kitchen 解码中位数 2472.8ms → 304.0ms（约 8.1 倍）；约 12 秒 living 为 783.4ms → 288.4ms。此结果仅是 CPU 解码，不是新 30 秒片段或 CUDA 端到端性能。

单进程、单模型 worker；默认等待队列 8，满时返回 429。请求超时默认 120 秒返回 504；超时不会提前释放仍在执行的 worker，也不会触发重叠前向。禁止为同一 GPU 盲目增加 Uvicorn worker，扩容使用独立实例。

API 默认 Bearer 鉴权，`RAYA_API_KEYS` 以逗号分隔支持轮换。`/healthz` 和 `/readyz` 不鉴权；`GET /v1/models` 返回 Raya 能力信息。API 提供 `x-request-id`。仅本地临时测试可显式打开 `RAYA_ALLOW_ANONYMOUS=true`。

媒体默认仅接受 data URL；HTTPS 必须在 `RAYA_MEDIA_HOSTS` 可信域名名单中，禁止重定向、代理环境和私有/保留 IP。部署层仍需限制出站网络。日志不记录正文、媒体或密钥。公网在网关配置 TLS、调用方配额和限流。

## 容器与验证

```bash
# 模型预下载，.env 配置完毕后
# CPU Linux；Docker Desktop 无法使用 macOS MPS
 docker compose up -d --build
# NVIDIA Linux，宿主机需要驱动和 NVIDIA Container Toolkit
 docker compose -f compose.yaml -f compose.cuda.yaml up -d --build

uv run ruff check raya_maas scripts local_tests tests
uv run pytest -q
# 对已经启动的原生服务进行真实多模态 HTTP 检查（从 .env 读取密钥）
uv run python scripts/smoke.py
# 视频解码对比
uv run python scripts/benchmark_video.py \
  tests/fixtures/m3bench_kitchen.mp4 tests/fixtures/m3bench_living.mp4 --runs 3
```

容器非 root，模型只读挂载。CUDA 实机和容器构建尚未验证；MPS/CPU 与解码测试见 [验收记录](docs/validation.md)。已有 bench 的来源记录在 `local_tests/provenance.json` 和 `tests/fixtures/provenance.json`，不据此假设未参与模型训练。

协议依据：[Jev 官方 API](https://docs.typesafe.ai/api)、[State](https://docs.typesafe.ai/concepts/state)、[Confidence](https://docs.typesafe.ai/confidence)。模型推理格式以提供的 Raya v1 手册为准。
