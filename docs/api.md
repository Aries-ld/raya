# Raya System One 决策接口 v1

核对日期：2026-09-27。依据 [Jev 官方 API](https://docs.typesafe.ai/api)、[问题类型](https://docs.typesafe.ai/primitives)、[输入 State](https://docs.typesafe.ai/concepts/state)、[置信度](https://docs.typesafe.ai/confidence)。第三方仿站不作为规范来源。

## HTTP

`POST /v1/systemone`

- `Authorization: Bearer <RAYA_API_KEY>`
- `Content-Type: application/json`
- 成功：200；响应带 `x-request-id`。
- 成功 JSON 顶层**仅** `model`、`answers`、`usage`。
- 不提供聊天 completion、流式生成、自由文本答案。
- 机器可读定义：[openapi.json](openapi.json)，运行时 `/openapi.json` 需要 Bearer key。

## 输入

| 字段 | 类型 | 规则 |
| --- | --- | --- |
| model | string | 必填，`raya-decision-v1` |
| state | string / object / array | 必填，包含事实和材料 |
| questions | object | 必填，1–16 个命名问题，每个问题独立评估同一 state |

问题 ID 不进入模型 prompt，只用来索引响应。每个问题都必须明确设置 `type`、`instructions`，不能靠 ID 猜问题。`instructions` 可以是非空文字、JSON 对象或数组。额外未知协议字段会报错。

### Choice

```json
{
  "type": "choice",
  "instructions": "这个请求应该交给哪个团队？",
  "criteria": {
    "billing": "账单、付款和退款问题",
    "account": "密码和登录问题",
    "other": null
  }
}
```

2–26 个选项。key 是调用方定义的 ID（1–128 字符），value 是描述（string/object/array），或 null，此时直接使用 ID 作为候选含义。内部按序映射 A–Z，输出重新映射到原 ID。描述允许结构化 JSON，渲染长度有限制。

答案：`{"type":"choice","choice":"account","probabilities":{"billing":0.05,"account":0.9,"other":0.05},"confidence":0.9}`。

### Score

```json
{
  "type": "score",
  "instructions": "画面清晰程度如何？",
  "criteria": ["严重模糊，主体难辨", "基本可辨认", "主体和细节清晰"]
}
```

2–10 个从低到高的等级。输出 `score = Σ(i × p_i)`，索引从 0 开始，可能为小数。`legend` 将字符串索引映射到等级描述；结构化描述以 JSON 字符串回显。

答案：`{"type":"score","score":1.6,"legend":{"0":"严重模糊，主体难辨","1":"基本可辨认","2":"主体和细节清晰"},"probabilities":{"0":0.1,"1":0.2,"2":0.7},"confidence":0.7}`。

### Noul

```json
{
  "type": "noul",
  "instructions": "用户是否需要找回账号？",
  "criteria": {"true": "明确要求解决账号访问问题", "false": "与账号访问无关"}
}
```

`criteria` 可省略，true/false 也可分别省略，使用默认是/否含义。内部仍有两个明确候选。答案：`{"type":"noul","noul":0.94}`，只表示是的概率，不另返回 confidence。以上答案数值均为说明示例。

## 多模态 state：Raya 扩展

Jev 当前只支持文本。Raya 顶层保持相同字段，在 `state` 中保留 `media` 作为多模态标记：

```json
{
  "model": "raya-decision-v1",
  "state": {
    "text": "观察图片中右侧的人物。",
    "media": [{"type": "image", "url": "https://your-trusted-media-host.example/image.jpg"}]
  },
  "questions": {
    "held_object": {
      "type": "choice",
      "instructions": "右侧人物手里拿着什么？",
      "criteria": {"basketball": "篮球", "laptop": "笔记本电脑", "cup": "水杯"}
    }
  }
}
```

示例 HTTPS 域名是占位符，须替换为服务端 `RAYA_MEDIA_HOSTS` 允许的真实域名。推荐直接用 `data:image/jpeg;base64,...` / `data:video/mp4;base64,...`。视频对应 `type:"video"`，问题定义完全相同。

包含 `media` 的 state 按多模态对象校验，只允许 `text` 与 `media` 两个字段；额外业务数据可嵌套到 `text` 中。不包含 `media` 的 object/array 是普通结构化文字材料。多模态 `text` 可为空，但 `questions` 仍必填；媒体本身不替代问题。

默认最多 4 个媒体、最多 1 个视频。URL 必须为匹配模态的 data URL 或可信 HTTPS URL。禁止 HTTP/file URL、服务端本地路径。图像按内容解码校验，视频按时间/帧数/像素预算解码，仅处理画面。

本地测试文件允许 `file:./video.mp4`；加载器读取后转换为 data URL，再进行正式 schema 校验。这个语法不是 HTTP 协议的一部分。

## 输出、计量、阈值

```json
{
  "model": "raya-decision-v1",
  "answers": {
    "held_object": {
      "type": "choice",
      "choice": "basketball",
      "probabilities": {"basketball":0.98,"laptop":0.01,"cup":0.01},
      "confidence": 0.98
    }
  },
  "usage": {"input_tokens":500,"output_tokens":0}
}
```

`answers` 和请求中的问题 ID 一一对应。概率是当前候选集合内的局部 softmax。Raya 的 choice/score confidence 明确定义为最大候选概率；Jev 的公开文档未给出精确 confidence 算式，不宣称数值定义相同。

`score`、`noul` 是候选分布上的映射，不生成文本，也不意味着已经重新验证这两种任务的校准性能。调用方可以对 choice/score.confidence 或 noul 的概率自行设阈值；不在协议里添加替你执行动作的字段。

多问题复用媒体下载/解码结果，但逐个执行模型前向，问题之间不传入其他问题的答案。`usage.input_tokens` 是各问题实际输入长度的总和（每个都会计算 state），不是 Jev 的共享前缀计费语义；`output_tokens` 恒为 0。

## 限制和错误

当前默认：每个问题完整输入 4096 token，state 与 instructions 合成后的内部文本最多 65536 字符，候选渲染后最多 2048 字符；每个 criteria 描述序列化后最多 1900 字符。媒体 64MiB/个，原始帧 2000 万像素，视频 60 秒、8 帧，HTTP 请求体 96MiB。设置项见 `raya_maas/config.py`。

验证在执行前检查全部问题结构；视觉展开后的 token 上限在每个问题预处理后检查。失败时整次 HTTP 请求返回错误，不返回部分 answers。

错误 JSON 为 `{"error":{"code":"invalid_request_error","message":"..."}}`，不返回请求原文或 base64：

| HTTP | 场景 |
| --- | --- |
| 401 | 缺少/错误 API key |
| 404 | 模型或路由不存在 |
| 413 | 请求体/媒体字节超限 |
| 422 | 协议格式错误、缺少问题/候选、非法媒体、token/时长/像素超限 |
| 429 | 等待队列已满，携带 Retry-After |
| 500 | 推理内部失败 |
| 503 | 尚未就绪或正在关闭 |
| 504 | 请求等待/推理超时 |

Jev 官方还使用 529；Raya 此版本未提供相同的过载服务实现。未知参数不静默忽略。API key、媒体 URL、模型名需按 Raya 配置替换，不能声称直接复用 Jev 的密钥和模型。
