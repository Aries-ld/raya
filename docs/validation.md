# 原生决策协议验收

日期：2026-09-27。Apple Silicon arm64，24GiB 内存，Python 3.12、torch 2.10.0、Transformers 5.17.0。模型与 processor revision 固定在 `raya_maas/download.py`，本地下载记录在 `models/manifest.json`。

本记录验证 Raya 实现的决策协议与真实模型调用，不是 Jev 性能对比或 Raya 训练校准复测。

## 协议和自动测试

正式接口为 `POST /v1/systemone`，请求顶层仅 `model / state / questions`，成功响应仅 `model / answers / usage`。图片、视频在 `state.media` 中显式提供，问题在 `questions.*.instructions`，选择项在 `criteria`。

44 项自动测试通过。覆盖：

- choice 原选项 ID 映射、score 概率加权计算、noul 是的概率，以及混合问题响应。
- 缺少 state/questions/instructions/criteria、非法问题类型、候选/等级/问题数量越界的 422。
- 图片不能替代问题；不同问题 ID 不改变模型输入；结构化 state/instructions/criteria。
- 单次前向、末位 logits、候选掩码 softmax、CUDA→MPS→CPU 顺序及故障回退。
- 鉴权、请求体限制、未知路由/模型、队满、超时后 worker 仍独占模型、丢弃失效排队任务。
- 媒体来源/大小/时长/像素限制、稀疏和顺序解码采样帧逐像素一致。
- OpenAPI 请求/响应字段和路由；旧聊天端点不存在。
- 本地常驻会话只加载一次，重新读取修改后的 JSON/媒体，输入错误可修复后重试，关闭/加载失败释放资源。

测试使用模型桩验证协议/调度；真实模型结果另外记录。Starlette TestClient 对 httpx 存在一条上游弃用提示，不影响当前测试通过。

## 真实模型本地测试

通过直接运行 `local_tests/run_text.py`、`run_image.py`、`run_video.py`，不启动客户端 HTTP 请求、不传启动参数，并将工作目录设为 `/tmp`，确认与 IDE 工作目录无关。读取同目录完整 JSON 请求，加载真实模型到 MPS FP16：

| 输入 | 问题 | 输出 | 概率/置信度 |
| --- | --- | --- | --- |
| 找回密码文字 | intent / choice | reset_password | 0.999118 |
| 同一文字 | needs_account_help / noul | 是的概率 | 0.983940 |
| 同一文字 | urgency / score | 0.166383，等级范围 0–2 | 最大等级概率 0.909444 |
| m3bench 图片 | held_object / choice | basketball | 0.999689 |
| m3bench 30 秒视频 | food_in_box / choice | fries | 0.981071 |

choice 结果与人工预期相符；概率和约为 1。noul/score 用来验证计算与输出类型，不把几条样例当作专项校准结果。文本三问、图片一问、视频一问，单次本地推理总耗时分别约 1759ms、960ms、2714ms；不含模型加载，存在首次初始化影响，非稳态基准。

原始响应：`artifacts/systemone-local-{text,image,video}.json`。诊断：同名前缀加 `-diagnostics.json`，记录设备、每问 token/时间及媒体解码时间。公开响应不包含额外调试字段。

新视频由 m3bench `bedroom_01` 派生 `clip002.mp4` 裁剪转码，ffprobe 实测 **30.000 秒、30fps、900 帧**，模型采样 8 帧。图片、视频来源及校验和在 `local_tests/provenance.json`。训练重叠未知。

## 真实 HTTP 服务

服务已切换到原生协议，通过 `scripts/smoke.py` 读取相同的三份 JSON、展开本地媒体为 data URL，
携带实际 Bearer key 调用 `/v1/systemone`。文本三种问题和图片/视频 choice 均通过，响应结构
与本地一致。缺少 questions 返回 422，旧聊天路径返回 404。原始输出在
`artifacts/systemone-http.json`。`/readyz` 探针正常，服务仍监听 `0.0.0.0:8000`。

## 本地常驻会话

真实运行 `local_tests/run_session.py`，在同一进程依次执行图片、回车重复图片、切换文字、
切换 30 秒视频，再输入 q 正常退出。模型初始化仅一次，约 4855ms；重复图片这次约 584ms，
第 4 次视频约 2315ms，后续诊断均为 `model_reused=true`、`model_load_ms=0`。
分别保存为 `artifacts/session-repeat-image.json` 和 `artifacts/session-video.json`。
这两个耗时仅描述本次运行，不是稳态性能承诺；视频解码和每次前向仍会执行。

## 视频解码对比

此前使用同样的解码模块执行 `scripts/benchmark_video.py`，每种方式 3 次取中位数：

| 文件 | 顺序解码 | seek 解码 | 加速 | 实际解码帧数 |
| --- | --- | --- | --- | --- |
| kitchen，60 秒 | 2472.811ms | 303.977ms | 8.14× | 1800 → 204 |
| living，约 12 秒 | 783.431ms | 288.441ms | 2.72× | 362 → 108 |

时间点与选中 RGB 帧逐像素一致。这里是纯 CPU 解码，不是原生多问题协议吞吐或 CUDA 性能。新 30 秒视频编码/GOP 不同，本次解码约 1257ms，不沿用 8.14× 的收益。

## 尚未验证的范围

当前无 CUDA、无 Docker，CUDA 实机性能和容器构建未验证。此前 CPU FP32 的基础候选推理验证通过，但本轮原生协议的多模态真实测试在 MPS 上完成。NVDEC/GPU 解码、共享前缀并行问题、动态批处理未实现。

choice/score confidence 使用最大概率，Jev 官方未公开精确计算式，不能声称两者数值相同。Raya 上限 26 个候选、默认每问 4096 token，也不是 Jev 全部容量/性能指标的复现。公网域名、TLS 和网关配额不属于本次本地验收。
