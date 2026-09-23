# raya

多模态 System-1 快速决策模型（复刻 TypeSafe Jev 决策范式，扩展到文/图/视频）。

- **权威方案**：[`docs/raya-v1-design.md`](docs/raya-v1-design.md)
- **场景数据矩阵**：[`docs/scenario-coverage.md`](docs/scenario-coverage.md)
- **架构决策**：[`docs/decisions/`](docs/decisions/)

## 环境

```bash
# 本地（Mac）
~/miniconda3/envs/raya/bin/python -m pytest tests/     # PYTHONPATH=. 或先 cd 仓库根
# 训练机（seetacloud, RTX PRO 6000 96G）
ssh raya-gpu                                            # 免密
# 工作区 /root/autodl-tmp/raya/{code,data,checkpoints,logs}，conda env raya（torch 2.12）
```

约定：flat layout 不做 `pip install -e .`，一律仓库根 `PYTHONPATH=.` 运行；凭证放 `.env`（不进 git，只做存在性检查）。

## 结构

```
raya/
  raya/data/        # M1 数据层：schema（统一决策样本）+ registry + adapters
  scripts/m1/       # 数据导出/目检脚本
  tests/            # pytest；integration marker = 需网络下载，默认跳过
  docs/             # 设计文档与 ADR
```

## 里程碑

M1 数据层 → M2 渲染器+推理 harness → M3 教师蒸馏+SFT → M4 全量 FT+RLCD → M5 评测报告。验收门禁见设计文档 §7。
