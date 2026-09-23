"""NExT-QA adapter 测试。"""

import pytest

from raya.data.adapters.nextqa import NextQAAdapter, _convert
from raya.data.registry import get

_ROW = {
    "video": "3238737531",
    "frame_count": "2303",
    "width": "640",
    "height": "480",
    "question": "how many children are in the video",
    "answer": "3",
    "qid": "2",
    "type": "DC",
    "a0": "one",
    "a1": "three",
    "a2": "seven",
    "a3": "two",
    "a4": "five",
}


class TestConvert:
    def test_happy_path(self):
        s = _convert(_ROW, "train")
        assert s.id == "nextqa:train:3238737531:2"
        assert s.modality == "video"
        assert s.media_path == "data/media/nextqa/3238737531.mp4"
        assert s.question.type == "choice"
        assert len(s.options) == 5
        assert s.answer == "opt4"  # answer=3 → a3="two" → 第 4 个选项
        assert {o.id: o.text for o in s.options}[s.answer] == "two"
        assert s.answer_distribution is None

    def test_split_validation_accepted(self):
        assert _convert(_ROW, "val").split == "val"


class TestLoadGuards:
    def test_invalid_split_rejected(self, tmp_path):
        adapter = NextQAAdapter(cache_dir=str(tmp_path))
        with pytest.raises(ValueError, match="split must be one of"):
            list(adapter.load(split="test"))  # test 无答案，显式拒绝


class TestRegistry:
    def test_registered(self):
        adapter = get("nextqa")
        assert isinstance(adapter, NextQAAdapter)
        assert adapter.meta.has_soft_labels is False


@pytest.mark.integration
class TestLoadIntegration:
    """真实下载 train.csv（~4MB）。pytest -m integration（服务器上跑）"""

    def test_load_train_slice(self, tmp_path):
        adapter = NextQAAdapter(cache_dir=str(tmp_path))
        samples = list(adapter.load(split="train", limit=20))
        assert len(samples) == 20
        assert all(len(s.options) == 5 for s in samples)
        assert len({s.id for s in samples}) == 20
