# Raya Gesture Lab

交互参考 [jev-visual 的 Camera gestures 示例](https://github.com/hr98w/jev-visual/tree/4382bba455647400951429134ceb012ca155e3fe/demo/gestures)：摄像头图片驱动粒子效果，右侧展示实际输入与决策。这里接入的是当前 Raya v1 和 `/v1/systemone`，没有换用参考项目的 Qwen0.8B/MLX 推理，也没有引入手势关键点检测器。

## 启动与操作

```bash
# 终端一；已启动则无需重复运行
uv run raya-serve

# 终端二；不加载另一份模型，只启动本地网页与代理
uv run raya-demo
```

在 Chrome 等支持摄像头的浏览器打开 **http://127.0.0.1:8765**。PyCharm 也可运行
`local_tests/run_gesture_demo.py` 或共享配置 `Raya Camera Demo`。

1. 点“开启摄像头”，允许浏览器访问。此时仅预览，不发送图片。
2. 将一只手完整放在中央方框内，点“开始识别”。拇指伸直也计入手指数。
3. 保持手势一小段时间，观察效果与右侧返回的概率是否符合预期。
4. “暂停识别”停止新请求但保留预览与结果；“关闭摄像头”停止摄像头轨道并清空页面历史。

| 手势 | API 选项 ID | 效果 |
| --- | --- | --- |
| 拳头，0根手指 | fist | 引力核 |
| 1根手指 | one | 光柱 |
| 2根手指 | two | 双环 |
| 3根手指 | three | 三叶旋 |
| 4根手指 | four | 四象轨道 |
| 5根手指，张开手掌 | five | 星芒绽放 |
| 无清晰单手、遮挡、多个手 | none | 待机星尘 |

## 模型看到什么

浏览器每秒采样一次，取摄像头中央正方形，缩放到384×384，以JPEG图片输入：

```json
{
  "model": "raya-decision-v1",
  "state": {
    "text": "A current unmirrored camera crop containing the hand to classify.",
    "media": [{"type": "image", "url": "data:image/jpeg;base64,..."}]
  },
  "questions": {
    "gesture": {
      "type": "choice",
      "instructions": "Count the fully extended fingers of the single visible hand…",
      "criteria": {"fist":"zero…","one":"exactly one…","two":"exactly two…","three":"exactly three…","four":"exactly four…","five":"all five…","none":"no clear hand…"}
    }
  }
}
```

上面的长描述为节选；准确提示词与候选在 `raya_maas/demo_server.py`。模型只收到真实图片和固定问题/候选，没有坐标、手指关键点、人工标签或期望效果。

预览为镜像方便摆手，实际模型输入不镜像。只请求视频，不请求麦克风。它做的是图片推理，
与视频接口的 fps=1 解码策略无关。每个摄像头请求只有一个 choice 问题。

## 采样和结果应用

- 以1000ms间隔调度；上一帧仍在请求中则跳过，下一次取最新帧，不积压旧帧。
- 浏览器延迟的计时器也不允许补发采样突发流量。1Hz是采样目标，不承诺模型每秒必定返回一次。
- 动画用独立 `requestAnimationFrame` 持续运行。粒子位置有视觉过渡，分类结果不做投票或概率平滑。
- 置信度低于0.55，或图像到结果已超过2.5秒，展示原始结果，但恢复待机效果。
- 暂停、关闭、页面隐藏后，旧请求结果不能再控制粒子；关闭时清除页面内的图片和历史。
- 右侧保留最近12条，每条包含采样图、7个候选概率、耗时和原始JSON。默认展开最新JSON，
  用户主动展开的旧JSON不会被新结果强行折叠。导出按钮下载最近12条响应与耗时，不含图片。

## 耗时定义

每条记录显示三个不同的指标，不改变 Jev 风格的 JSON 响应：

| 指标 | 来源与含义 |
| --- | --- |
| 模型前向 | `X-Raya-Forward-Ms`，该请求各问题前向阶段合计，含设备拷贝和候选分数计算，不含排队/媒体准备 |
| 服务端处理 | `X-Raya-Processing-Ms`，worker开始处理到生成结果，包含媒体准备和全部前向，不含队列等待 |
| 浏览器往返 RT | fetch开始到收到结果，包含代理、网络传输和队列等待，不含模型加载 |

队列等待另由 `X-Raya-Queue-Ms` 记录，导出时可查看。缺失的计时显示 `—`，不会伪装成0。
worker在模型所属线程里快照每个请求的计时，避免并发请求读到其他请求的时间。

## 本地代理与配置

演示代理仅绑定 `127.0.0.1`，默认8765端口；不会把 `.env` 中的 API key 放进 HTML、JavaScript、
URL或浏览器存储。它使用相同的标准决策请求调用现有 MaaS。启用Host/Origin检查、POST专用请求头、
2MiB实际请求体限制、禁止缓存和页面嵌入。页面资源都在本地，无远程字体或统计脚本。

- `RAYA_DEMO_PORT`：本地网页端口。
- `RAYA_DEMO_MAAS_URL`：MaaS地址，默认 `http://127.0.0.1:8000`。
- `RAYA_API_KEYS`：代理使用其中第一个key，必须与MaaS匹配。

仅用于本机调试，不将这个无登录页的代理公开到公网。当前默认帧只发送到本机MaaS；
若手动配置远程MaaS，图片会发送到那个地址。页面历史在内存中，不自动落盘。

## 验证和边界

```bash
uv run pytest -q
npm ci
npm run test:demo
# 使用已安装的 Chrome；要求 demo 已启动。只使用合成摄像头，不调用真实摄像头。
npm run test:browser
```

- Python验证：代理不泄露key、拒绝跨源/Host伪造/超大请求、透传原始决策与计时、上游不可用状态，
  以及请求计时不会串到下一次推理。
- 浏览器验证：摄像头需主动开启、预览不发请求、六种粒子映射、采样节奏、慢请求跳过、过期结果不应用、
  JSON/耗时展示、导出、暂停、轨道关闭、权限拒绝、手机布局。
- 浏览器使用合成摄像头和模拟模型响应验证UI；截图文件名中的 mocked 表示这一点，不是模型识别准确率。
- 真实Raya接口经本地代理使用已有bench图片跑通，收到实际候选分布与计时；记录在
  `artifacts/gesture-api-smoke.json`。该图片不是六种手势的标注数据集。
- **真实手势准确率未做完整标注评测**，需在实际摄像头、光照、背景和手型下验证。示例会如实显示误识别，
  不通过外部手势模型或手工规则替换Raya的结果。

参考仓库：`hr98w/jev-visual`，查看时commit `4382bba455647400951429134ceb012ca155e3fe`，MIT许可。
本演示采用独立的Raya协议接入和粒子实现，没有复制其模型推理代码。
