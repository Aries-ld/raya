# 三种模态，同一个决策协议

**编辑 JSON 完整请求，再右键 Run / Debug。每次都显式提供材料、问题和候选。**

| 运行入口 | 编辑请求 | 默认媒体 |
| --- | --- | --- |
| `test_text.py` | `text.json` | state 中的文字 |
| `test_image.py` | `image.json` | `image.jpg` |
| `test_video.py` | `video.json` | `video.mp4`，30 秒 m3bench 片段 |

JSON 分成三个部分：

- `model`：`raya-decision-v1`。
- `state`：待判断的文字或 `{"text":"背景", "media":[...]}`。
- `questions`：自己给问题命名；每个问题有 `type`、`instructions`，choice/score 还必须有 `criteria`。

比如图片里的 `held_object` 问题明确问“右侧人物手里拿着什么？”，候选是 `basketball / laptop / cup / book`，描述分别是篮球、笔记本电脑、水杯和书。修改问题时要同步修改候选。输出 `answers.held_object.choice` 直接是你的选项 ID，不是 A/B/C。

`choice` 从选项中选一个；`score` 从有序等级的概率计算分数；`noul` 内部固定是/否选项，返回“是”的概率。`text.json` 演示一份 state 同时问三种问题。

PyCharm 解释器选择项目 `.venv/bin/python`，无需参数、API key 或 HTTP 服务。`request` 变量是已经校验的正式请求；`result` 只有 `model / answers / usage`，与 HTTP 输出相同。可在这两行打断点。诊断耗时在 `artifacts/local-decision-diagnostics.json`。

文件中的 `file:./image.jpg` / `file:./video.mp4` 是本地测试加载器的便利语法，运行时转换成 data URL；HTTP 服务不支持 file URL。可以直接替换同名媒体，也可直接修改 JSON 中的相对文件路径，路径相对于 JSON 所在目录。

默认视频：m3bench `bedroom_01` 派生 `clip002.mp4` 前 30 秒，H.264、30fps、900 帧、无音轨。视频问题问“红色纸盒里是什么食物”，并提供四个明确候选。模型仍固定采样 8 帧。是否参与过训练未知，不能把这条样例当作严格的未见数据评测。

媒体已准备在本机，不进 Git；来源、转换参数、SHA256 在 `provenance.json`。完整协议及差异见 [docs/api.md](../docs/api.md)。
