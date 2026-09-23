"""统一决策样本 schema（设计文档 §5.9 的代码化身）。

所有 adapter 的唯一产出类型。训练/渲染/评测只认这个结构，
校验失败 = adapter bug，宁可在数据层炸也不带进训练。
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

Modality = Literal["text", "image", "video"]
QuestionType = Literal["choice", "noul", "score"]

# answer_distribution 求和容差：人类标注归一化后有浮点毛刺，1e-3 内放过
_DIST_SUM_TOL = 1e-3


class Option(BaseModel):
    """一个候选选项。id 是样本内唯一标识（如 opt1），渲染期才映射到标签 token。"""

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class Question(BaseModel):
    type: QuestionType
    text: str = Field(min_length=1)


class DecisionSample(BaseModel):
    """一条决策样本。三种原语统一形态：答案永远是 option id，分布永远覆盖全部选项。"""

    id: str = Field(min_length=1, description="全局唯一，建议 '{source}:{split}:{seq}'")
    modality: Modality
    media_path: Optional[str] = Field(default=None, description="文本样本为 None；图/视频必填")
    state: Optional[str] = Field(default=None, description="可选补充上下文（如第一视角自述）")
    question: Question
    options: list[Option] = Field(min_length=2)
    answer: str = Field(description="正确选项的 id")
    answer_distribution: Optional[dict[str, float]] = Field(
        default=None,
        description="软标签（VQA v2 10人 / ChaosNLI 100人 / 教师投票分布）；None=只有硬标签",
    )
    source: str = Field(min_length=1, description="血缘：数据集名")
    split: str = Field(min_length=1, description="血缘：原始 split")
    license: str = Field(min_length=1, description="血缘：许可证；不明就写 'unknown' 显式标记")

    @model_validator(mode="after")
    def _check_consistency(self) -> "DecisionSample":
        option_ids = [o.id for o in self.options]
        if len(set(option_ids)) != len(option_ids):
            raise ValueError(f"duplicate option ids: {option_ids}")

        if self.answer not in option_ids:
            raise ValueError(f"answer '{self.answer}' not in option ids {option_ids}")

        if self.modality in ("image", "video") and not self.media_path:
            raise ValueError(f"modality={self.modality} requires media_path (sample {self.id})")

        if self.answer_distribution is not None:
            dist_keys = set(self.answer_distribution)
            if dist_keys != set(option_ids):
                raise ValueError(
                    f"distribution keys {sorted(dist_keys)} != option ids {sorted(option_ids)}"
                )
            if any(v < 0 for v in self.answer_distribution.values()):
                raise ValueError("distribution has negative probability")
            total = sum(self.answer_distribution.values())
            if abs(total - 1.0) > _DIST_SUM_TOL:
                raise ValueError(f"distribution sums to {total}, expected 1.0±{_DIST_SUM_TOL}")

        return self
