# ADR 0001: 基座架构选型——小 VLM decoder + Jev 式读出，而非多模态融合 encoder

- 状态：已定（2026-09-23）
- 背景：H20 资源未到位，训练目标硬件降为单卡工作站级（RTX 6000D / PRO 5000-72G）。重新审视「小模型」选型时出现两条路线分叉。

## 候选路线

- **Path A**：小 VLM decoder（Qwen3-VL-2B）+ 掩码投影读标签 logits（Jev 官方同构）
- **Path B**：多模态融合 encoder（SigLIP2 级视觉 encoder + ModernBERT 级文本 encoder + 融合层 + 决策头，~0.5-1B），即 Laya 路线的多模态版

## 决策

**选 Path A。**

## 理由

1. **视频是 Path B 的死穴**：不存在现成的「小视频-文本融合 encoder」预训练 checkpoint，自研时序-文本融合是数月级研究项目；Qwen3-VL-2B 原生支持视频（时间戳 token、20min、256K 上下文）。
2. **常识决策能力**：raya 的目标场景（机器人/硬件决策）需要场景理解 + 物理/社会常识，CLIP/ALBEF 系融合 encoder 在知识型 VQA 上有历史天花板；decoder 基座继承 LLM 预训练常识。
3. **动态候选空间**：Path A 候选经 prompt 传入、掩码投影读出，选项数量/内容运行时任意变；Path B 受 encoder 上下文（512-1024 token）限制，视觉 token 挤压下候选空间窘迫。
4. **「Response Wide Shut」(arXiv 2025) 证据**：生成式 VLM 的「知道但说不出」退化只发生在文本生成空间；Jev 式 logits 读出工作在内部表示空间，**在 decoder 骨干上获得了分类头的结构可靠性**——Path B 的核心优势被零成本吸收。
5. **延迟不构成差异**：两者均在 Jev 的 20-200ms 决策窗口内（Path A 单前向 ~20-50ms）。
6. **工程风险与生态**：Qwen 全家桶（transformers/vLLM/GGUF/MLX/LLaMA-Factory）vs 全手搓融合架构。

## 后果

- 基座锁定 **Qwen3.5-2B**（2026-02 发布，Apache 2.0；Qwen3.5 代取消独立 -VL 线，原生早融合多模态；混合架构 Gated DeltaNet+Gated Attention，262K 上下文；VideoMME 69.0 超上代 3B）。备选基线 InternVL3-2B（M2 做 zero-shot A/B）。
- ⚠️ M2 验证项：混合架构的 LoRA target module 配置（DeltaNet 层与标准 transformer 不同）；视觉 tower 冻结/解冻策略实测。
- 训练全流程（含 RLCD 全量版，~44GB）单卡 72G 可完成，多卡/服务器依赖解除。
- Path B 保留为 V1 后的衍生品选项（raya-nano：固定垂直域、图文 only、极致小），届时复用本项目数据管线。
- 参考：MiniCPM-V 4.5（8B，面壁）超出 tiny 目标，但其 3D-Resampler 视频 token 压缩（96×）值得在长视频场景借鉴。

## 备选基线记录（2025 统一抽帧评测，VideoMME / MVBench）

| 模型 | VideoMME | MVBench |
|---|---|---|
| Qwen2.5-VL-3B | 60.9 | 65.0 |
| InternVL3-2B | 54.6 | 70.4-79.2 |
| SmolVLM2-2.2B | 50.7 | 53.5 |

注：跨论文对比受抽帧协议影响大，M2 以自测 zero-shot 基线为准。
