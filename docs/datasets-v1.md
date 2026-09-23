> ⚠️ **已归档**：训练方式与配比以定稿 [`v1-training-plan.md`](v1-training-plan.md) 为准；本文档保留数据集调研细节作参考。

# Raya V1 数据集方案（文本 + 图片 + 视频）

> 目标：为 Qwen2.5-VL-3B 的决策后训练（SFT 格式对齐 + RLCD 校准）提供数据。
> 决策三原语：choice（枚举选择）/ noul（是非概率）/ score（序数打分）。
> 核心原则：**训练集与评测集严格分离**（评测集一条都不许进训练，防污染）。

## 一、数据资产总览

### 文本（Text）

| 数据集 | 规模(训练) | 格式 | 原语 | 质量/备注 |
|---|---|---|---|---|
| RACE | 87.9K | 4 选 1 阅读理解 | choice | 人类考试题，质量极高 ⭐ |
| MNLI | 393K | 3 选(蕴含/中立/矛盾) | choice | 大规模，标注一致性一般 |
| BoolQ | 9.4K | yes/no | noul | 自然是非问题 |
| CommonsenseQA | 9.7K | 5 选 1 | choice | 常识推理 |
| Banking77 | 10K(train) | 77 类意图 | choice(高基数) | Jev 官方评测用过的集 |
| CLINC150 | 18K(train) | 150 类意图 | choice(高基数) | 含 OOS 拒识样本 |
| AG News | 120K | 4 类主题 | choice | Laya 训练/评测用过 |
| SST-2 / SST-5 | 67K / 8.5K | 2 类 / 5 级 | noul / **score** | SST-5 星级天然序数 |
| **ChaosNLI** ⭐ | 4.6K | 3 选 + **100 人标注分布** | choice(软标签) | 校准训练的金标准软标签 |
| GoEmotions | 58K | 多标签情感 | noul(可拆) | 多标注员，可构造分布 |

评测（held-out）：MMLU、ARC-Challenge、Banking77 test、MNLI dev、ChaosNLI（兼作校准评测）

### 图片（Image）

| 数据集 | 规模(训练) | 格式 | 原语 | 质量/备注 |
|---|---|---|---|---|
| **VQA v2 train** ⭐⭐ | ~1.1M 问题 | 开放题 + **10 人标注分布** | choice(改造)/noul | yes/no 子集天然 noul；软标签直接服务 RLCD |
| GQA train | ~14M(22M×70%) | 开放/是非，场景图生成 | choice(改造)/noul | 程序化生成、答案均衡、组合推理强 |
| ScienceQA train | 12.7K | MC(带讲解) | choice | ⚠️ CC BY-NC-SA（非商用） |
| A-OKVQA train | 17.1K | MC(4选) + 理由 | choice | ⚠️ 许可证不明确，需确认 |
| KonIQ-10k | 8K(train) | 图像质量 MOS | **score** | 1-5 有序等级 |
| AVA | ~230K | 美学评分 1-10 | **score** | 分布可做软标签 |

评测（held-out）：SEED-Bench(19K MC)、MMBench、MMMU、POPE(noul)、ScienceQA test

### 视频（Video）

| 数据集 | 规模(训练) | 格式 | 原语 | 质量/备注 |
|---|---|---|---|---|
| **LLaVA-Video-178K** ⭐ | 196K MC | MC(GPT-4o 生成) | choice | 最大规模现成视频 MC 训练集；合成数据需与人工集混用 |
| NExT-QA train | 34.1K | MC(5 选) | choice | 人工标注，因果/时序/描述 |
| STAR train | 45.7K | MC(4 选) | choice | 情境推理(交互/序列/预测/可行性)——最贴近机器人决策 |
| TGIF-QA train | 125K | MC(5 选) + 计数 | choice/score | GIF 短片段，贴近端侧输入形态 |
| Something-Something V2 | 220K 视频 | 174 类动作模板 | choice(改造) | 干扰项从动作模板采样 |
| Kinetics-700 | 650K 片段 | 700 类动作 | choice(改造) | 量大，作为规模兜底 |

评测（held-out）：Video-MME、MVP(55K,防捷径)、MVBench(⚠️ 有捷径问题,仅参考)、NExT-QA test、Ego4D 子集(第一视角,机器人场景)

## 二、质量策略（比数量更重要）

1. **软标签是 RLCD 的稀缺燃料**：VQA v2（10 人/题）和 ChaosNLI（100 人/题）提供人类标注分布，恰当评分规则可以直接以分布为目标——这是校准训练和别家拉开差距的关键，优先接入。
2. **原生 MC 优先，改造集兜底**：原生 MC（RACE/NExT-QA/STAR/LLaVA-Video）的干扰项是人或程序精心设计的；分类集改造（Sth-Sth/Kinetics/AG News）需自己构造干扰项——**用难负例**（同大类内 / 标签 embedding 近邻采样），不用随机负例，否则模型学会走捷径。
3. **合成数据与人工数据混配**：LLaVA-Video 量大但是 GPT-4o 合成，单独训会继承其偏置；以人工标注集（NExT-QA/STAR/VQA v2）为质量锚，配比上合成:人工 ≈ 6:4。
4. **位置偏置消毒**：渲染时选项顺序随机洗牌，正确答案在 A/B/C/D 上均匀分布。
5. **标签风格随机化**：渲染层随机轮换标签体系（A/B/C、1/2/3、yes/no、自定义枚举词），让模型学「读当前 prompt 的标签」而非背标签。
6. **许可证红线**：ScienceQA 是 CC BY-NC-SA（非商用）、A-OKVQA 许可证不明——研究可用，但若 raya 要开源发布权重，需在发布前替换或确认。

## 三、配比与规模估算

**SFT 阶段（格式对齐）**，总量 ~200K：

| 模态 | 占比 | 来源 |
|---|---|---|
| 文本 | 25% (~50K) | RACE 20K + MNLI 10K + Banking77/CLINC 10K + BoolQ/SST 10K |
| 图片 | 40% (~80K) | VQA v2 40K + GQA 25K + ScienceQA/A-OKVQA 10K + KonIQ 5K |
| 视频 | 35% (~70K) | LLaVA-Video 35K + NExT-QA 15K + STAR 15K + TGIF 5K |

**RLCD 阶段（校准）**，总量 ~25K：以人工标注 MC 为主 + 全部软标签子集（VQA v2 软标签 10K + ChaosNLI 4.6K），确保每题有明确（或分布化）ground truth。

3B + LoRA（冻结视觉编码器）：SFT 单卡 A100 约 1 天内；RLCD 参考 Laya 规模（30K 题 2×T4 4-5h 全量 421M），3B LoRA 单卡 A100 可控。

## 四、统一 Schema（adapter 转换目标）

```json
{
  "id": "vqav2:train:0001",
  "modality": "image | video | text",
  "media_path": "data/media/...",
  "state": "可选的补充上下文文本",
  "question": {"type": "choice | noul | score", "text": "..."},
  "options": [{"id": "opt1", "text": "..."}, ...],
  "answer": "opt1",
  "answer_distribution": {"opt1": 0.7, "opt2": 0.2, "opt3": 0.1},
  "source": "vqa_v2", "split": "train", "license": "..."
}
```

`answer_distribution` 可空；有（VQA v2/ChaosNLI）时 RLCD 用分布目标。
渲染层（模板拼装 + 标签风格随机 + 选项洗牌）在训练时在线进行，不落盘。

## 五、V1 里程碑

- [ ] M1 数据层：统一 schema + 下载脚本 + 前 5 个 adapter（RACE/VQA v2/KonIQ/NExT-QA/LLaVA-Video）
- [ ] M2 渲染器 + 推理 harness（掩码投影 + 共享前缀扇出）
- [ ] M3 SFT → temperature scaling 基线（accuracy/ECE/延迟首报）
- [ ] M4 RLCD (GRPO + 恰当评分规则 + 软标签)
- [ ] M5 评测报告：SEED-Bench/Video-MME/MVP/POPE + 文本基准，对标 Jev/Laya 公开数据
