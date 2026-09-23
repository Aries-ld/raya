"""VQA v2 adapter 测试：软标签构建是核心资产，重点测分布正确性与跳过规则。"""

import pytest

from raya.data.adapters.vqav2 import VQAv2Adapter, _convert, media_path_for
from raya.data.registry import get


def _record(answer_type, annotator_answers, mca, question="What is this?"):
    return {
        "question_id": 262148000,
        "image_id": 262148,
        "question": question,
        "question_type": "what",
        "answer_type": answer_type,
        "multiple_choice_answer": mca,
        "answers": [
            {"answer": a, "answer_confidence": "yes", "answer_id": i + 1}
            for i, a in enumerate(annotator_answers)
        ],
    }


class TestMediaPath:
    def test_canonical_coco_path(self):
        assert media_path_for("train", 262148) == (
            "data/media/vqav2/train/COCO_train2014_000000262148.jpg"
        )
        assert media_path_for("validation", 9) == (
            "data/media/vqav2/validation/COCO_val2014_000000000009.jpg"
        )


class TestYesNoToNoul:
    def test_full_agreement(self):
        s = _convert(_record("yes/no", ["no"] * 10, "no", "Is this soup?"), "validation")
        assert s.question.type == "noul"
        assert [o.text for o in s.options] == ["yes", "no"]
        assert s.answer == "opt2"
        assert s.answer_distribution == {"opt1": 0.0, "opt2": 1.0}

    def test_split_vote_distribution(self):
        s = _convert(
            _record("yes/no", ["yes"] * 7 + ["no"] * 3, "yes"), "validation"
        )
        assert s.answer == "opt1"
        assert s.answer_distribution["opt1"] == pytest.approx(0.7)
        assert s.answer_distribution["opt2"] == pytest.approx(0.3)

    def test_non_yesno_annotator_answer_renormalized(self):
        # 标注者写了 "maybe" → 不计入，yes/no 内部再归一
        s = _convert(
            _record("yes/no", ["yes"] * 6 + ["no"] * 2 + ["maybe"] * 2, "yes"),
            "validation",
        )
        assert s.answer_distribution["opt1"] == pytest.approx(0.75)
        assert s.answer_distribution["opt2"] == pytest.approx(0.25)

    def test_no_yesno_votes_skipped(self):
        assert _convert(_record("yes/no", ["maybe"] * 10, "maybe"), "validation") is None


class TestOtherToChoice:
    def test_annotator_pool_as_options(self):
        s = _convert(
            _record("other", ["down"] * 7 + ["at table"] * 3, "down"), "validation"
        )
        assert s.question.type == "choice"
        assert [o.text for o in s.options] == ["down", "at table"]
        assert s.answer == "opt1"
        assert s.answer_distribution == {"opt1": 0.7, "opt2": 0.3}
        assert s.media_path == "data/media/vqav2/validation/COCO_val2014_000000262148.jpg"
        assert s.id == "vqav2:validation:262148000"

    def test_unanimous_answer_skipped(self):
        # 全员一致 → 选项空间为 1 不构成决策，留给难负例挖掘阶段
        assert _convert(_record("other", ["cat"] * 10, "cat"), "validation") is None

    def test_empty_annotations_filtered(self):
        s = _convert(
            _record("other", ["cat"] * 5 + [""] * 3 + ["dog"] * 2, "cat"), "validation"
        )
        assert s is not None
        assert sum(s.answer_distribution.values()) == pytest.approx(1.0)

    def test_mca_outside_pool_skipped(self):
        assert (
            _convert(_record("other", ["cat"] * 5 + ["dog"] * 5, "bird"), "validation")
            is None
        )

    def test_distribution_sums_to_one(self):
        s = _convert(
            _record("number", ["2"] * 5 + ["3"] * 3 + ["two"] * 2, "2"), "train"
        )
        assert sum(s.answer_distribution.values()) == pytest.approx(1.0)


class TestRegistry:
    def test_registered_with_soft_label_flag(self):
        adapter = get("vqav2")
        assert isinstance(adapter, VQAv2Adapter)
        assert adapter.meta.has_soft_labels is True


@pytest.mark.integration
class TestLoadIntegration:
    """真实下载官方 validation 标注 JSON（~80MB，走 cache）。pytest -m integration（服务器上跑）"""

    def test_load_small_slice(self, tmp_path):
        adapter = VQAv2Adapter(cache_dir=str(tmp_path))
        samples = list(adapter.load(split="validation", limit=10))
        assert len(samples) == 10
        assert all(s.modality == "image" and s.media_path for s in samples)
        assert len({s.id for s in samples}) == 10
        # 软标签资产必须真实存在（VQA v2 的核心价值）
        assert all(s.answer_distribution is not None for s in samples)

    def test_train_split_available(self, tmp_path):
        # lmms-lab 版无 train 的坑已踩过：官方源必须能出 train
        adapter = VQAv2Adapter(cache_dir=str(tmp_path))
        samples = list(adapter.load(split="train", limit=5))
        assert len(samples) == 5
        assert all(s.split == "train" for s in samples)
