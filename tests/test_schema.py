"""schema 校验测试：合法样本必须过，各类脏数据必须在数据层炸出来。"""

import pytest
from pydantic import ValidationError

from raya.data.schema import DecisionSample


def _base_kwargs(**overrides):
    kw = dict(
        id="race:train:0001",
        modality="text",
        question={"type": "choice", "text": "What is the main idea?"},
        options=[
            {"id": "opt1", "text": "A cat"},
            {"id": "opt2", "text": "A dog"},
            {"id": "opt3", "text": "A bird"},
        ],
        answer="opt2",
        source="race",
        split="train",
        license="unknown",
    )
    kw.update(overrides)
    return kw


class TestValidSamples:
    def test_text_sample_without_media(self):
        s = DecisionSample(**_base_kwargs())
        assert s.media_path is None
        assert s.answer_distribution is None

    def test_image_sample_with_soft_labels(self):
        s = DecisionSample(
            **_base_kwargs(
                modality="image",
                media_path="data/media/vqa/0001.jpg",
                answer_distribution={"opt1": 0.3, "opt2": 0.6, "opt3": 0.1},
            )
        )
        assert sum(s.answer_distribution.values()) == pytest.approx(1.0)

    def test_distribution_sum_within_float_tolerance(self):
        # 10 人标注归一化后的浮点毛刺（0.1×10 = 0.9999...）必须放过
        DecisionSample(
            **_base_kwargs(
                answer_distribution={"opt1": 0.1, "opt2": 0.1, "opt3": 0.8},
            )
        )


class TestRejectedSamples:
    def test_answer_not_in_options(self):
        with pytest.raises(ValidationError, match="not in option ids"):
            DecisionSample(**_base_kwargs(answer="opt9"))

    def test_duplicate_option_ids(self):
        with pytest.raises(ValidationError, match="duplicate option ids"):
            DecisionSample(
                **_base_kwargs(
                    options=[
                        {"id": "opt1", "text": "A"},
                        {"id": "opt1", "text": "B"},
                    ],
                )
            )

    def test_image_modality_requires_media_path(self):
        with pytest.raises(ValidationError, match="requires media_path"):
            DecisionSample(**_base_kwargs(modality="image"))

    def test_video_modality_requires_media_path(self):
        with pytest.raises(ValidationError, match="requires media_path"):
            DecisionSample(**_base_kwargs(modality="video"))

    def test_distribution_keys_must_cover_all_options(self):
        with pytest.raises(ValidationError, match="!= option ids"):
            DecisionSample(
                **_base_kwargs(answer_distribution={"opt1": 0.5, "opt2": 0.5})
            )

    def test_distribution_must_sum_to_one(self):
        with pytest.raises(ValidationError, match="sums to"):
            DecisionSample(
                **_base_kwargs(
                    answer_distribution={"opt1": 0.5, "opt2": 0.4, "opt3": 0.4}
                )
            )

    def test_distribution_rejects_negative(self):
        with pytest.raises(ValidationError, match="negative"):
            DecisionSample(
                **_base_kwargs(
                    answer_distribution={"opt1": 1.5, "opt2": -0.5, "opt3": 0.0}
                )
            )

    def test_fewer_than_two_options(self):
        with pytest.raises(ValidationError):
            DecisionSample(
                **_base_kwargs(options=[{"id": "opt1", "text": "only one"}])
            )
