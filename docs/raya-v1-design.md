# Raya V1 完整技术方案

> 版本：v1.0（2026-09-23 定稿）。本文档是 raya 项目的**唯一权威方案**，吸收并取代 `v1-training-plan.md` / `datasets-v1.md` 的内容。
> 协作纪律：改前对齐 → 实现+自测 → 用户 review 验收；向前兼容；每层必测。

## 1. 项目概述

**Raya** = 多模态 System-1 快速决策模型，复刻 TypeSafe Jev 的决策范式并扩展到多模态。

- **V1 范围**：文本 + 图片 + 视频（音频 V2 再做）
- **输入**：多模态 state + 结构化问题（候选选项运行时传入）
- **输出**：校准概率分布，三种决策原语：
  - `choice`：枚举选择（每选项概率）
  - `noul`：是非/真假概率 P(true) ∈ [0,1]
  - `score`：序数评分期望
- **关键特性**：单次前向（零生成步）、输出结构性可验证（恒在候选空间内）、置信度经校准可用于门控、共享前缀扇出（同 state 多问题并发，边际延迟近零）

**目标场景**：智能硬件 / 机器人 / Agent 决策层（意图路由、动作选择、内容审核、权限检查、状态机流转），低置信时升级给 System-2 大模型。

## 2. 背景：Jev 范式与竞对分析

### 2.1 Jev 是什么（复刻对象，2026-09 TypeSafe AI）

Jev 是 decoder-only Transformer（与 LLM 同族），**拿掉自回归生成循环**。决策四步：

1. 单次 prefill 前向（state + 问题 + 候选列表都在 prompt 内）
2. 末位 logits **掩码投影**到候选标签 token（A/B/C、yes/no 等）
3. 局部 softmax 出概率（只归一化候选位置）
4. 生成步数 = 0

**相对 LLM 的四项优势**（注意：优势不来自「换架构」）：

| 优势 | 来源 |
|---|---|
| 延迟 20-200ms（官方称 ~200×） | 无生成循环 |
| 结构性可验证 | 选项空间由服务端掩码强制，模型不可能越界 |
| 校准置信度 | RLCD 后训练（目标 ECE 而非困惑度） |
| 推测性扇出 | 长 state 只 prefill 一次，N 问题挂同一 KV cache |

**选项空间动态化原理**：候选列表是请求期以结构化 JSON 传入的；服务端用固定模板渲染 prompt 并为每个选项分配标签 token、建立「选项 id ↔ token id」映射；掩码按映射表从 ~15 万维 logits 里抽出对应位置。模型对选项数量/内容无感，3 个还是 4 个选项只是抽几行的区别。**约束在模型外**：模型负责语义打分（RLCD 解决），服务端负责空间合规（掩码解决）。

### 2.2 竞对与 raya 差异化

| 项目 | 路线 | 缺口 |
|---|---|---|
| Jev（官方） | decoder-only + RLCD，闭源 | 纯文本 |
| Laya（开源） | ModernBERT encoder + RLCD，权重+训练 notebook 全开源 | 纯文本 |
| OpenJev | 冻结基座（Qwen3.5-4B）+ 掩码投影 + temperature scaling | 纯文本、无校准训练 |
| jev-visual | Qwen3.5-0.8B + MLX 工程（共享 prefill + 候选 logits） | 无校准、无视频/音频 |
| **raya** | **多模态原生（文/图/视频）+ RLCD 校准 + 软标签训练** | — |

raya 的两个真护城河：**多模态原生输入**（三家都没有）+ **校准训练**（jev-visual/OpenJev 没有）。

## 3. 架构设计

### 3.1 基座

**Qwen3.5-2B**（2026-02 最新代，Apache 2.0；架构决策见 [ADR 0001](decisions/0001-base-architecture-decoder-vs-encoder.md)）：原生早融合多模态（文/图/视频），混合架构 Gated DeltaNet+Gated Attention，262K 上下文，VideoMME 69.0（超上代 3B）；4bit ~1.5GB 端侧可部署。备选基线 InternVL3-2B（M2 A/B）；降级保底 Qwen2.5-VL-3B / Qwen3-VL-2B。V2 音频再议。

### 3.2 API 协议（结构化，调用方不拼 prompt）

```json
{
  "id": "req-001",
  "state": {"text": "...", "images": ["..."], "video": "..."},
  "question": {"type": "choice | noul | score", "text": "...", "criteria": "..."},
  "options": [{"id": "opt1", "description": "..."}, ...]
}
```

### 3.3 服务端渲染与读出（训练/推理严格一致）

1. 固定模板渲染：`[多模态 state 前置] → [问题] → [候选列表后置]`（顺序锁死，防注意力竞争）；
2. 每选项分配标签 token（A/B/C…，标签集协议预定义，服务端维护 label→token id 表）；
3. 渲染时随机化：选项洗牌（正确答案位置均匀）+ 标签风格轮换（训练期）；
4. 单次前向 → 末位 logits 按映射表掩码抽取 → 局部 softmax → 按选项 id 回填概率；
5. 响应带 prompt hash + 模型版本（审计可复盘）；
6. 扇出：state KV cache 共享，N 个问题后缀批量前向。

## 4. 训练流水线（四步，顺序执行，每步过门禁才进下一步）

### Step 0 — 教师蒸馏数据生产
- 教师：**MiniMax-M3**（已定，2026-09-23 探针三通：文/图/视频 ✅）。OpenAI 兼容端点 `https://api.minimax.cn/v1`，原生 `image_url`/`video_url` 输入（fps 0.2-5 可调），1M 上下文
- **软分布取法**：API 不给 logits → 每题 **温度采样 ~8 次投票**估计候选分布；投票调用一律 `thinking: disabled` 控成本，并发上限 5（账号配额）
- 对象：**视频训练集 + 决策场景集**（无人类软标签的类别）
- 产出：每题候选概率分布（软标签），供 SFT 软目标 + RLCD 分布奖励
- 成本：~100K 题 × 8 票 ≈ 80 万调用，走 coding plan 配额
- 凭证：`{workspace}/.env`（MINIMAX_API_KEY，只做存在性检查，不进 git）

### Step 1 — SFT 格式对齐（LoRA，配方迭代）
- **损失：option-local listwise 交叉熵**——只在本次候选标签 token 上归一化做 CE；⚠️ 不是全词表 LM loss（训练读出方式必须与推理的掩码投影一致）
- 冻结视觉编码器，LoRA 作用于 projector + LLM
- 有 `answer_distribution` 的样本用分布软目标（KL/软 CE）

### Step 2 — 全量微调（配方定稿后，交付版）
- 解冻视觉塔顶层 + LLM 全参数，沿用 option-local 损失
- 拉能力上限（尤其视频时序推理）

### Step 3 — RLCD 校准（GRPO）
- 奖励：严格恰当评分规则（log score / Brier）对照硬标签或**软分布**
- GRPO：对掩码后 softmax 加温度采样，同题组内归一化，无 value model
- 全量策略版需策略+参考双模型 ~96GB：H20(96G) 单卡勉强 / 多卡 ZeRO；本地 PRO 5000(72G) 走 LoRA 策略降级版
- 参考：Laya 开源 notebook（30K 题、2×T4、4-5h 全量 421M）

## 5. 数据方案

### 5.1 理解题基底（学术集）

| 模态 | 数据集 | 取用 | 原语 |
|---|---|---|---|
| 文本 | RACE / MNLI / CommonsenseQA | 40K | choice |
| 文本 | BoolQ / SST-2 / SST-5 | 15K | noul / score |
| 文本 | Banking77 / CLINC150 / AG News | 15K | choice(高基数) |
| 图片 | **VQA v2 train**（10 人软标签 ⭐） | 45K | choice/noul |
| 图片 | GQA train 子集 | 25K | choice/noul |
| 图片 | KonIQ-10k / AVA 子集 | 10K | score |
| 视频 | LLaVA-Video-178K MC（合成 ≤60%） | 35K | choice |
| 视频 | NExT-QA / STAR / TGIF-QA train | 30K | choice |

### 5.2 决策场景题（生产题型）
内容审核 ToxicChat/Aegis 10K + Agent 动作选择 Mind2Web/AndroidWorld 10K。

### 5.3 拒答/弃权类（~8%）⭐
CLINC150 OOS + 自构造三类：选项全不对 / 输入模态缺失 / 证据不足 → 「无法判断」。生产必备。

### 5.4 中文决策数据（~10-15%）
教师蒸馏 + 高质量翻译，三模态覆盖，中文子集单独报 ECE。

### 5.5 软标签资产（RLCD 燃料）⭐
VQA v2（10 人分布，图）、ChaosNLI（100 人分布，文，4.6K）、Step 0 教师分布（视频/场景）。

### 5.6 配比与总量
**SFT ~200K**：文 25% / 图 40% / 视频 35%（场景+拒答 ~15%，中文 ~10-15% 跨模态）。
**RLCD ~25-50K**：人工标注 MC + 全部软标签子集。

### 5.7 质量纪律
1. 干扰项用难负例（同大类 / embedding 近邻），禁随机负例
2. 合成:人工 ≈ 6:4，人工集为质量锚
3. 选项洗牌 + 标签风格渲染期随机化
4. 训练/评测严格分离；每条样本带血缘（source/split/license）
5. 许可证红线：ScienceQA CC BY-NC-SA、A-OKVQA 不明 → 开源发布权重前替换/确权

### 5.8 评测集（held-out，一条不进训练）
SEED-Bench、Video-MME、MVP（防捷径，替代有捷径问题的 MVBench）、POPE、MMLU、Banking77 test、ChaosNLI + 自建决策场景集 + 上线前真实日志回放集。

### 5.9 统一 Schema（adapter 转换目标）

```json
{
  "id": "vqav2:train:0001",
  "modality": "image | video | text",
  "media_path": "data/media/...",
  "state": "可选补充上下文",
  "question": {"type": "choice | noul | score", "text": "..."},
  "options": [{"id": "opt1", "text": "..."}],
  "answer": "opt1",
  "answer_distribution": {"opt1": 0.7, "opt2": 0.3},
  "source": "vqa_v2", "split": "train", "license": "..."
}
```

## 6. 资源与工作流

### 6.1 实际到位资源（2026-09-23 定稿）

| 资源 | 用途 |
|---|---|
| **seetacloud 服务器：RTX PRO 6000 Blackwell 96G**（`ssh raya-gpu` 免密） | **训练主力 + 数据生产**。208 核 / 1TB RAM / 数据盘 `/root/autodl-tmp` 815G；工作区 `/root/autodl-tmp/raya/{code,data,checkpoints,logs}`；conda env `raya`（torch 2.12.1+cu130，支持 Blackwell sm_120）。2B 全流程峰值 ~44GB，余量一倍 |
| Mac 本机 | 本地开发（`~/PycharmProjects/raya`），代码经 git 同步服务器 |
| MiniMax-M3 API（coding plan，并发 5） | Step 0 教师蒸馏，key 在服务器 `.env` |
| `/autodl-pub` 14T 公共数据集盘 | 淘现成数据集，省下载 |

注：QS H20 容器方案已废弃（共享授权不通）；系统盘仅 30G，一切落 autodl-tmp（租借机重建会丢系统盘）。

### 6.2 时间预估（PRO 5000 单卡，2B 模型）
SFT-LoRA ~半天/轮；全量 FT ~1 天/轮；RLCD ~1 天/轮；**一轮完整迭代 ~1.5-2 天**。教师蒸馏（数据生产）走 API 或独立环境，不抢训练卡。

### 6.3 开发-训练工作流（已定）
本地开发 + PRO 5000 冒烟（1K 样本全绿）→ git 同步（数据走下载脚本不进 git）→ 服务器重跑同一 1K 冒烟（排环境差异）→ 正式训练 → Claude SSH 上机监控（loguru 结构化日志落盘：loss/ECE/lr/吞吐/显存；nvidia-smi）→ checkpoint 双轨保存（step + 指标）→ 每轮跑 held-out 门禁。

**硬件无关纪律**：单卡/多卡同一套代码（torchrun + ZeRO 配置化），不允许服务器特供路径。

## 7. 里程碑与验收门禁

| 指标 | 目标 |
|---|---|
| 决策准确率 | ≥ 基座 zero-shot + 15pt（各 held-out） |
| ECE | ≤ 0.08（对标 Laya 0.081） |
| 延迟 | 单决策 p50 < 50ms；扇出边际 < 10ms（H20 实测） |
| 拒答准确率 | 弃权类 ≥ 85% |
| 结构合规率 | 100%（掩码保证，回归测试守住） |

- [ ] **M1 数据层**（~1 周）：schema + 下载脚本 + adapter（RACE/VQA v2/KonIQ/NExT-QA/LLaVA-Video）+ 血缘 + pytest 全绿
- [ ] **M2 渲染器 + 推理 harness**（~1 周）：模板/标签随机化/洗牌 + 掩码投影 + 共享前缀扇出 + 基座 zero-shot 基线数值
- [ ] **M3 Step 0+1**（~1.5 周）：教师蒸馏 + SFT-LoRA 配方迭代 → temperature scaling 基线 → 首份 accuracy/ECE/延迟报告
- [ ] **M4 Step 2+3**（~1.5 周）：全量 FT + RLCD，过全部门禁
- [ ] **M5 评测报告 + 场景验收**（~1 周）：held-out 全量，对标 Jev/Laya 公开数据，真实日志回放验收

## 8. 风险登记

| 风险 | 缓解 |
|---|---|
| MVBench 捷径问题污染评测 | 评测主力 Video-MME + MVP |
| 合成数据偏置 | 6:4 配比 + 人工锚 + RLCD 只用人工/软标签 |
| 候选项注意力竞争（多模态 token 长放大） | 模板锁死 state 前置候选后置；M2 实测 |
| 教师蒸馏质量不及人类分布 | 软标签训练中人类分布权重 > 教师分布 |
| 中文校准差 | 中文子集单独报 ECE，不达标加量 |
| 3B 视频能力天花板 | Step 2 解冻视觉塔；兜底升级 Qwen2.5-VL-7B（96G 单卡仍可全量 FT） |
| M3 教师配额/限流（并发 5） | 蒸馏管线断点续跑 + 指数退避；投票数 8→5 降级预案 |
| 租借机系统盘重建丢数据 | 一切落 `/root/autodl-tmp`；checkpoint 定期 rsync 回 Mac |
| transformers 对 Qwen3.5 支持滞后 | M2 首验 `AutoModel` 加载 + 混合架构 LoRA target；不兼容则临时用官方推荐分支 |

## 9. 主要参考

- [TypeSafe AI: Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- [Jev 架构深度拆解 (kuhung.me)](https://jev.kuhung.me/)
- [Laya（开源竞对）](https://github.com/Aries-ld/laya) · [OpenJev](https://github.com/TheoLeeCJ/openjev) · [jev-visual](https://github.com/Aries-ld/jev-visual)
- 数据集：[VQA v2](https://visualqa.org/) · [GQA](https://arxiv.org/pdf/1902.09506) · [LLaVA-Video-178K](https://llava-vl.github.io/blog/2024-09-30-llava-video/) · [NExT-QA](https://arxiv.org/pdf/2105.08276) · [STAR](https://bobbywu.com/STAR/) · [TGIF-QA](https://github.com/YunseokJANG/tgif-qa) · [RACE](https://learn-en.org/pdf/reading-comprehension-dataset-race.pdf) · [ChaosNLI](https://huggingface.co/datasets/metaeval/chaos-mnli-ambiguity) · [ScienceQA](https://scienceqa.github.io/) · [A-OKVQA](https://arxiv.org/pdf/2206.01718v1.pdf)
- 评测：[SEED-Bench](https://arxiv.org/pdf/2307.16125.pdf) · [Video-MME](https://video-mme.github.io/) · [MVP](https://arxiv.org/html/2506.09987v1) · [MMMU](https://mmmu-benchmark.github.io/)
- [Qwen3-Omni Technical Report](https://arxiv.org/html/2509.17765v1)
