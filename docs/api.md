# Raya API 接入文档

**版本：v1 · 更新日期：2026-09-27 · 面向 HTTP API 调用方**

Raya 是候选决策服务，支持文字、文字＋图片、文字＋视频。你提供待判断的材料、明确的问题和候选/等级，服务返回可直接用于程序判断的结构化答案。

请求风格采用 `state + questions`，响应采用 `answers`。不提供自由聊天、文本生成、工具调用或流式输出，也不使用 OpenAI Chat Completions 协议。

模型：[Raya decision v1](https://huggingface.co/yuyu199741/raya-decision-v1)；基座：[Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)。

## 1. 接入前准备

请向服务维护者获取：

| 配置 | 说明 |
| --- | --- |
| `RAYA_BASE_URL` | 服务根地址，例如 `https://your-raya-service.example`，**不包含 `/v1`** |
| `RAYA_API_KEY` | 分配给你的 Bearer API key，单独安全传递，不写入代码仓库 |
| 模型 ID | 默认 `raya-decision-v1`，可通过 `/v1/models` 确认 |

文中的 `.example` 地址是占位符，必须替换。`http://127.0.0.1:8000` 仅适用于服务运行在调用方本机的情况，不能作为其他人的远程接入地址。

```bash
export RAYA_BASE_URL='https://your-raya-service.example'
export RAYA_API_KEY='替换为维护者分配的密钥'
```

业务请求统一带上：

```http
Authorization: Bearer <RAYA_API_KEY>
Content-Type: application/json
```

客户端使用单个 `RAYA_API_KEY`。服务端配置名 `RAYA_API_KEYS` 是密钥列表，两者不要混淆。

## 2. 最快跑通一个文字决策

```bash
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

成功返回 HTTP 200。以下是**格式示例，概率和 token 数为说明用数值**：

```json
{
  "model": "raya-decision-v1",
  "answers": {
    "intent": {
      "type": "choice",
      "choice": "reset_password",
      "probabilities": {
        "reset_password": 0.98,
        "register": 0.01,
        "other": 0.01
      },
      "confidence": 0.98
    }
  },
  "usage": {
    "input_tokens": 96,
    "output_tokens": 0
  }
}
```

读取 `answers.intent.choice` 即可得到业务选项 ID。`intent`、`reset_password` 等名称由调用方定义，不要求使用 A/B/C。

## 3. 接口列表

| 方法 | 路径 | 鉴权 | 用途 |
| --- | --- | --- | --- |
| POST | `/v1/systemone` | 必须 | 文字/图片/视频决策 |
| GET | `/v1/models` | 必须 | 模型 ID、支持类型和部分运行限制 |
| GET | `/openapi.json` | 必须 | 下载机器可读接口定义 |
| GET | `/healthz` | 无需 | 进程活性检查 |
| GET | `/readyz` | 无需 | 模型就绪检查 |

查询模型：

```bash
curl --fail-with-body "$RAYA_BASE_URL/v1/models" \
  -H "Authorization: Bearer $RAYA_API_KEY"
```

默认响应结构：

```json
{
  "models": [{
    "id": "raya-decision-v1",
    "question_types": ["choice", "score", "noul"],
    "modalities": ["text", "text+image", "text+video"],
    "limits": {
      "choice_options": 26,
      "score_levels": 10,
      "questions": 16,
      "input_tokens_per_question": 4096,
      "min_video_seconds": 3.0,
      "max_video_seconds": 20.0,
      "video_sampling_fps": 1,
      "video_min_sample_frames": 4
    }
  }]
}
```

`/healthz` 成功返回 `{"status":"ok"}`；`/readyz` 就绪时返回 200 和 `{"status":"ready"}`，未就绪时可返回503。服务尚在启动时也可能暂时无法建立连接。

## 4. 通用请求结构

```json
{
  "model": "raya-decision-v1",
  "state": "要判断的材料或上下文",
  "questions": {
    "your_question_id": {
      "type": "choice",
      "instructions": "需要模型判断什么？",
      "criteria": {
        "option_a": "第一个选项的含义",
        "option_b": "第二个选项的含义"
      }
    }
  }
}
```

| 字段 | 必填 | 类型与要求 |
| --- | --- | --- |
| `model` | 是 | 字符串，与部署的模型 ID 一致 |
| `state` | 是 | 文字、JSON对象、JSON数组，或第6节的多模态对象 |
| `questions` | 是 | 问题ID→问题对象的映射，1–16项 |
| `questions.<id>.type` | 是 | `choice`、`score`、`noul`之一 |
| `questions.<id>.instructions` | 是 | 非空问题/判断说明，可为文字、JSON对象或数组 |
| `questions.<id>.criteria` | 视类型 | choice/score必填，noul可选 |

问题 ID 为1–128字符，响应保留原ID。ID只是索引，**不能代替 `instructions` 中的完整问题**。所有问题独立评估同一份state，一个问题的答案不会作为另一个问题的输入。

普通结构化state示例：

```json
{
  "ticket": {"message": "订单被重复扣费了，请帮我退款。", "order_id": "ORDER-123"},
  "policy": "确认重复扣费后可退回多扣的款项。"
}
```

可在instructions中明确引用字段，如“`ticket.message` 的主要诉求是什么？”。state只放事实和材料；判断逻辑写进instructions，选项含义写进criteria。

未知的顶层字段或问题字段会返回422。不要传入 `messages`、`temperature`、`max_tokens`、`stream` 等聊天接口参数。

## 5. 三种决策类型

### 5.1 `choice`：从候选中选一个

`criteria` 是2–26项的对象，key是选项ID，value是说明；说明支持非空文字、JSON对象/数组或null。null表示直接使用选项ID作为含义。

```json
{
  "type": "choice",
  "instructions": "这个请求应该由哪个团队处理？",
  "criteria": {
    "billing": "账单、付款和退款",
    "account": "密码、登录和账号访问",
    "other": "无法归入上述团队的请求"
  }
}
```

答案字段：

| 字段 | 说明 |
| --- | --- |
| `type` | 固定为 `choice` |
| `choice` | 概率最高的原始选项ID |
| `probabilities` | 每个候选的概率，合计约为1 |
| `confidence` | 当前实现为最大的候选概率 |

候选必须覆盖你真正要区分的情况。漏掉真实答案时，模型仍只能在已给选项中选一个；高置信度不代表选项集合完整。必要时显式加入 `other` 或“信息不足”选项，并写清它的含义。

### 5.2 `score`：按有序等级评分

`criteria` 是2–10项数组，按从低到高排序，等级索引从0开始。

```json
{
  "type": "score",
  "instructions": "用户明确表达的紧急程度如何？",
  "criteria": ["一般请求，无时间要求", "要求今天完成", "必须立即处理"]
}
```

示意答案：

```json
{
  "type": "score",
  "score": 1.6,
  "legend": {"0": "一般请求，无时间要求", "1": "要求今天完成", "2": "必须立即处理"},
  "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
  "confidence": 0.7
}
```

`score = Σ(等级索引 × 该等级概率)`，因此可以是小数，范围为0到等级数减1。它不是“选择了第几个等级”，也不是 `confidence`。`legend` 回显各等级；结构化描述以JSON字符串回显。

### 5.3 `noul`：是/否判断的概率

只需给出问题，候选固定为是/否；可以补充它们各自的含义。

```json
{
  "type": "noul",
  "instructions": "用户是否在请求退款？",
  "criteria": {
    "true": "明确要求退回已支付的款项",
    "false": "未提出退款要求"
  }
}
```

示意答案：

```json
{"type": "noul", "noul": 0.94}
```

`noul` 表示“是”的概率，范围0–1。接近0.5表示不确定，不表示中等等级。没有单独的confidence字段。`criteria`及其中的true/false均可省略。

### 5.4 一次请求多个问题

三种类型可混合在同一个questions中。可直接使用 [text-request.json](examples/text-request.json)；它对同一条账号求助信息判断意图、是否需要账号帮助和紧急度：

```bash
curl --fail-with-body --max-time 150 \
  "$RAYA_BASE_URL/v1/systemone" \
  -H "Authorization: Bearer $RAYA_API_KEY" \
  -H 'Content-Type: application/json' \
  --data-binary @examples/text-request.json
```

上面文件路径按当前目录调整。当前服务逐个执行问题，多个问题会增加RT；同一请求中的媒体下载和解码结果会复用。任意问题失败时整次请求返回错误，不返回部分answers。

## 6. 图片和视频输入

### 6.1 多模态state

```json
{
  "text": "与图片或视频相关的背景信息",
  "media": [{"type": "image", "url": "data:image/jpeg;base64,REPLACE_WITH_BASE64"}]
}
```

- `media`存在时，state只允许 `text`、`media` 两个字段；其他业务数据可放进 `text` 的JSON对象中。
- `text`可省略或为空；**questions和判断说明仍必填**，不能只发媒体让服务自行猜测问题。
- `media[].type` 为 `image` 或 `video`。
- 多媒体按传入顺序提供，问题中可写“第一张图片”“第二张图片”。默认最多4个媒体，其中最多1个视频。
- 示例中的 `REPLACE_WITH_BASE64` 仅为占位符，发送前必须替换；第7节脚本会自动完成。

### 6.2 推荐：Base64 data URL

```text
data:image/jpeg;base64,<图片文件的完整Base64>
data:image/png;base64,<图片文件的完整Base64>
data:image/webp;base64,<图片文件的完整Base64>
data:video/mp4;base64,<视频文件的完整Base64>
```

图片建议JPEG/PNG/WebP，视频建议MP4（H.264）。其他编码能否读取取决于部署环境。Base64必须包含完整data URL前缀，不能只传编码字符串。

### 6.3 可选：HTTPS URL

```json
{"type":"image","url":"https://media.example.com/photo.jpg"}
```

只有维护者配置的可信域名可以使用；默认未开放任意远程URL。媒体地址必须直接返回HTTP 200，服务不跟随重定向。MaaS的Bearer key不会转发给媒体服务器。

不支持 `http://` 媒体地址、`file://`、浏览器 `blob:` URL或调用方本地路径。仓库内本地测试脚本支持的 `file:./xxx` 只是客户端便利语法，**不是HTTP协议**。

### 6.4 视频时长与采样

视频必须为 **3–20秒，包含边界**。服务读取视频流元数据校验实际时长，超限返回422 / `video_duration_out_of_range`，不会自动裁剪。

视频按Qwen `fps=1`规则采样：依据源帧数和帧率确定数量，再均匀取点；通常至少4帧。3/10/20秒视频通常分别采样4/10/20帧，奇数帧在编码时复制末帧补齐。它不是严格在整数秒取一张图片。服务只分析画面，**不转录或理解音轨**。

## 7. 可直接使用的 Python 客户端

附件 [call_raya.py](examples/call_raya.py) 仅使用Python标准库，不需要安装模型、PyTorch或额外SDK。设置第1节的环境变量后，从文档目录执行：

```bash
# 文字，含三种问题类型
python3 examples/call_raya.py --request examples/text-request.json

# 图片：自动将本地文件编码为合法data URL
python3 examples/call_raya.py --request examples/image-request.json --image ./photo.jpg

# 视频：文件必须为3–20秒
python3 examples/call_raya.py --request examples/video-request.json --video ./clip.mp4
```

将 `photo.jpg` / `clip.mp4` 换成自己的文件。脚本会把该模板中的media替换为这个文件，适用于单媒体示例。成功JSON写到标准输出，HTTP状态、request ID和时延写到标准错误，方便分别保存。客户端不跟随重定向，请使用维护者给出的最终服务地址。

也可用HTTP客户端自行调用：

```python
import json
import os
import urllib.request

payload = {
    "model": "raya-decision-v1",
    "state": "我忘记了登录密码。",
    "questions": {
        "needs_account_help": {
            "type": "noul",
            "instructions": "用户是否需要解决账号登录问题？"
        }
    }
}
request = urllib.request.Request(
    os.environ["RAYA_BASE_URL"].rstrip("/") + "/v1/systemone",
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers={
        "Authorization": "Bearer " + os.environ["RAYA_API_KEY"],
        "Content-Type": "application/json"
    },
    method="POST"
)
with urllib.request.urlopen(request, timeout=150) as response:
    result = json.load(response)
print(result["answers"]["needs_account_help"]["noul"])
```

## 8. 响应、置信度与计量

成功JSON顶层固定为：

| 字段 | 说明 |
| --- | --- |
| `model` | 实际处理请求的模型ID |
| `answers` | 按原问题ID返回对应类型的答案 |
| `usage.input_tokens` | 各问题实际输入token数之和，含材料、问题、候选和视觉token |
| `usage.output_tokens` | 恒为0，因为没有自回归文本生成 |

多个问题会重复计算state，因此input_tokens也重复计数。它是当前服务的输入长度统计，不是价格或账单字段。

`choice`和`score`的confidence为最大候选概率。改变候选集合、描述或问题措辞，会改变概率分布，不能把不同候选集合中的置信度直接当成同一指标比较。`score`为等级期望，`noul`为是的概率，两者含义不同。

调用方可自行设置自动处理阈值。阈值需根据本业务的标注集验证，不存在对所有任务通用的“可信分数”。服务不会代替调用方执行选项代表的业务动作。

### 耗时响应头

| 响应头 | 单位与含义 |
| --- | --- |
| `x-request-id` | 请求标识，排障时提供；鉴权/请求体限制阶段的早期错误可能不包含 |
| `X-Raya-Forward-Ms` | 毫秒，各问题前向阶段合计，含设备拷贝和候选概率计算 |
| `X-Raya-Processing-Ms` | 毫秒，worker开始处理到生成结果，包含媒体准备和前向，不含队列等待 |
| `X-Raya-Queue-Ms` | 毫秒，请求排队等待时间 |

客户端RT还包含上传、网络、网关和序列化等耗时，通常更长。响应头大小写不敏感；缺失值按“不可用”处理，不能当0。

## 9. 输入和运行限制

下表为当前默认配置；部分预算可由维护者收紧或调整。视频3–20秒为接口硬边界，部署配置只能收紧。

| 项目 | 默认限制 |
| --- | --- |
| 每请求问题数 | 1–16 |
| choice候选数 | 2–26 |
| score等级数 | 2–10 |
| 问题ID / 选项ID | 1–128字符 |
| 每个criteria描述 | 序列化后最多1900字符 |
| 内部单个候选文本 | 渲染后最多2048字符，包含选项ID |
| 每问题文本 | state与instructions拼接后最多65536字符 |
| 每问题完整输入 | 最多4096 token，**包含视觉展开和候选** |
| 每请求媒体数 | 最多4个，其中最多1个视频 |
| 单个媒体原始文件 | 最多64MiB |
| 原始图片 / 视频单帧 | 最多2000万像素 |
| 图片预处理像素预算 | 262144 |
| 视频单帧预处理像素预算 | 131072 |
| 视频时长 | **3–20秒** |
| 完整HTTP请求体 | 最多96MiB，包含Base64和JSON开销 |
| 等待队列 | 最多8个等待请求，单个worker执行推理 |
| 请求等待/推理超时 | 默认120秒 |

1MiB=1,048,576字节。Base64比原始文件大约增加1/3体积，多个媒体即使单独未超限，合计请求体也可能超过96MiB。

超出token/文件/时长预算会报错，不静默截断文本或视频。请求没有固定的1秒延迟承诺；时长、采样帧数、分辨率、问题数量、硬件和并发都会影响RT。

## 10. 错误处理与重试

服务自身的错误通常为：

```json
{
  "error": {
    "code": "video_duration_out_of_range",
    "message": "Video duration 30.000000s is outside the allowed inclusive range 3–20 seconds"
  }
}
```

| HTTP | 常见code | 调用方处理 |
| --- | --- | --- |
| 401 | `authentication_error` | 检查Bearer key，不重复盲目重试 |
| 404 | `model_not_found` / `invalid_request_error` | 检查模型ID和路径 |
| 413 | `payload_too_large` | 缩小文件或减少媒体数量 |
| 422 | `invalid_request_error` | 检查必填字段、问题类型、候选数量、媒体格式 |
| 422 | `context_length_exceeded` | 缩短材料、问题、候选，或减小视觉输入 |
| 422 | `video_duration_out_of_range` | 使用3–20秒视频 |
| 422 | `media_timeout` | 检查媒体URL可用性或视频解码复杂度 |
| 429 | `rate_limit_exceeded` | 按 `Retry-After` 等待后重试 |
| 500 | `inference_error` | 保留request ID，有限重试或联系维护者 |
| 503 | `service_unavailable` 或连接层错误 | 稍后重试；确认服务就绪 |
| 504 | `request_timeout` | 减小请求、降低并发，必要时有限重试 |

网关、连接层或Uvicorn过载响应可能不是JSON，客户端应先检查HTTP状态和Content-Type。

建议对429和短暂5xx使用带随机抖动的退避，设置有限次数；401/404/413/422先修正请求。客户端HTTP超时建议大于服务端120秒，例如150秒。

客户端取消或超时不保证已开始的模型计算立即停止。服务没有幂等键/结果缓存协议，重试可能重复计算。实时图片场景建议只保留一个在途请求，完成后取最新图片，不排队旧画面。

## 11. 获取接口定义和排障信息

```bash
curl --fail-with-body "$RAYA_BASE_URL/openapi.json" \
  -H "Authorization: Bearer $RAYA_API_KEY" \
  -o openapi.json
```

也可直接使用随文档提供的 [OpenAPI文件](openapi.json)，导入Apifox/Postman等工具后设置服务器地址和Bearer key。

联系维护者时提供：请求ID、请求时间、HTTP状态、error.code、媒体类型/尺寸/时长，以及去除密钥和Base64正文后的请求结构。不要在公共日志或工单里粘贴API key。

## 12. 随附文件

- [text-request.json](examples/text-request.json)：文字和三种问题类型。
- [image-request.json](examples/image-request.json)：图片请求模板。
- [video-request.json](examples/video-request.json)：视频请求模板。
- [call_raya.py](examples/call_raya.py)：无需额外依赖的Python调用脚本。
- [openapi.json](openapi.json)：机器可读协议定义。
