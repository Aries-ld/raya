# 验收记录

日期：2026-09-27。环境：Apple Silicon arm64，24GiB 统一内存，Python 3.12；torch 2.10.0、Transformers 5.17.0。模型与 processor commit 见 `models/manifest.json`。本记录区分协议/工程测试、真实模型冒烟测试和性能对比；不是模型准确率或校准评测。

## 自动测试

`uv run pytest -q`：32 项通过。

覆盖：OpenAI Python SDK、模型列表、JSON 和 SSE/usage、鉴权、请求大小和参数限制、候选模板、候选 logits 局部 softmax、单次无梯度前向、CUDA→MPS→CPU 设备顺序及后端异常回退、队满 429、超时不释放仍运行的 worker、取消排队任务、媒体限制和非法来源、视频采样一致性。

`ruff check`、`ruff format --check`、`git diff --check`、依赖锁检查均通过。测试中的模型桩不冒充真实模型；以下结果另外通过实际加载权重取得。

## 真实模型

`scripts/smoke.py` 通过 TCP HTTP 服务与官方 OpenAI SDK 实测。MPS FP16，单次样例耗时，包含首次视觉形状编译/初始化的影响，不是稳态压测数据。

| 样例 | 结果 | 置信度 | 模型前向 | HTTP 总耗时 |
| --- | --- | --- | --- | --- |
| 中文会议改期（mock 文本） | 修改会议时间 | 0.99451 | 439.5ms | 486.6ms |
| 英文取消预约（mock 文本） | Cancel a reservation | 0.99936 | 274.2ms | 276.8ms |
| m3bench 图片：右侧人物手持物品 | A basketball | 0.99988 | 3878.8ms | 3921.0ms |
| m3bench kitchen，60 秒 | A kitchen | 0.99019 | 1773.3ms | 2183.1ms |
| m3bench living，约 12 秒 | An indoor room | 0.99930 | 1134.9ms | 1485.2ms |

文本和图片案例另有人工设定预期，均相符；视频案例仅用于验证真实解码、输入、决策和概率结构，不作为带标注准确率统计。所有分布概率和约为 1，选项均在候选集内。另通过标准内联选项 + `json_object` + SSE 流式组合调用。

CPU FP32 单独加载真实权重，英文取消预约结果同为 B，置信度 0.99936，前向约 3269.3ms。CUDA 不可用，未做 CUDA 实机测试；自动降级路径由故障注入测试验证，本机正常路径直接选择 MPS。

原始输出：`artifacts/smoke.json`、`artifacts/cpu-smoke.json`。这些文件为本地产物，不入版本控制。

最终鉴权实例监听 `0.0.0.0:8000`，本机 `/readyz` 为 200，无密钥访问 `/v1/models` 为 401，正确密钥访问为 200；携带密钥的真实模型请求返回正确选项 B，设备为 MPS。密钥仅存于权限 0600 的 `.env`，鉴权复测输出在 `artifacts/authenticated-smoke.json`。本地运行实例的 PID 和日志分别在 `artifacts/server.pid` 与 `artifacts/server.log`。

## 无服务的三个独立脚本

`scripts/test_text.py`、`scripts/test_image.py`、`scripts/test_video.py` 均直接加载本地模型，
无需 HTTP 服务或 API key。分别使用命令行覆盖问题/候选、图片路径、视频路径后完成真实 MPS 测试：

- 文本：`请取消明天的会议。这是什么意图？`，候选为修改/创建/取消会议，返回 C「取消会议」。
- 图片：使用 `m3bench_living.jpg`，返回 A「A basketball」。
- 视频：使用 `m3bench_kitchen.mp4`，返回 A「A kitchen」，固定采样 8 帧。

三个样例均检查预期标签和概率归一化；输出位于 `artifacts/local-text.json`、
`artifacts/local-image.json`、`artifacts/local-video.json`。另检查三个脚本的 `--help`，
以及不存在的图片路径会在加载模型前报告输入错误。新增脚本后原有 32 项自动测试、格式检查仍通过。

后续将三个入口统一迁移至 `local_tests/`，默认读取同目录的 `text.txt`、`image.txt`、
`video.txt` 和 `image.jpg`、`video.mp4`，不再要求编辑脚本变量。
新视频来自 m3bench `bedroom_01` 派生片段 `clip002.mp4`，裁剪并转码后 ffprobe
验证时长为 30.000 秒、30fps、900 帧。使用 `/tmp` 作为工作目录，无启动参数运行三个脚本，
真实 MPS 推理分别返回「重置登录密码」「篮球」「薯条」，均匹配人工检查的默认预期。
结果在 `artifacts/default-files-{text,image,video}.json`。原有 32 项自动测试仍通过。
新视频仅可确认不同于此前的 kitchen/living 自测素材，训练数据重叠情况未知。

## 视频解码优化

`scripts/benchmark_video.py`，相同数据、相同 8 个采样时间点、相同 FFmpeg 缩放和 RGB 格式，分别运行 3 次，取中位数。计时仅包含视频解码/采样/缩放，不含读磁盘、HTTP 上传或模型前向。

| 文件 | 顺序解码 | seek 解码 | 加速 | 实际解码帧数 |
| --- | --- | --- | --- | --- |
| kitchen，60 秒 | 2472.811ms | 303.977ms | 8.14× | 1800 → 204 |
| living，约 12 秒 | 783.431ms | 288.441ms | 2.72× | 362 → 108 |

两种方法返回的 8 帧时间索引与 RGB 像素逐项完全一致，已用数组断言检查。基线也是只保留 8 帧的有界顺序实现，并非将整个视频展开成 RGB 后再采样的高内存实现。长 GOP、不同编码、磁盘/网络、分辨率会影响收益。

原始记录：`artifacts/video-benchmark.json`。bench 来源、裁剪方式和文件 SHA256：`tests/fixtures/provenance.json`。

## 部署边界

已提供 Dockerfile、CPU Compose 与 NVIDIA GPU override、非 root 运行、只读模型挂载、就绪探针。当前机器未安装 Docker，无 CUDA，容器构建和 GPU 吞吐未验证。公网域名、TLS、网关配额和服务器发布不在本次 MaaS 工程交付的实测范围内。

工程采用单模型 worker、有界队列，不提供动态批处理。视频解码使用 CPU PyAV/FFmpeg；没有宣称已完成 NVDEC/GPU 解码。手册给出的服务器毫秒级延迟与 ECE 结果未在此环境重新验证。
