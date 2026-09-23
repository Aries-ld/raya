"""KonIQ adapter 测试：score 原语 + 评分分布软标签。"""

import pytest

from raya.data.adapters.koniq import QUALITY_LEVELS, KoniqAdapter, _convert
from raya.data.registry import get


def _row(c, img="10004473376.jpg", split="train"):
    return {
        "img_name": img,
        "mos": "77.38",
        "official_split": split,
        "c_total": "105",
        **{f"c{i + 1}": str(v) for i, v in enumerate(c)},
    }


class TestConvert:
    def test_happy_path(self):
        s = _convert(_row([0.0, 0.0, 0.238, 0.695, 0.067]))
        assert s.question.type == "score"
        assert [o.text for o in s.options] == QUALITY_LEVELS
        assert s.answer == "opt4"  # c4 最大
        assert s.answer_distribution["opt4"] == pytest.approx(0.695, abs=1e-3)
        assert s.modality == "image"
        assert s.media_path == "data/media/koniq/10004473376.jpg"
        assert s.split == "train"

    def test_distribution_renormalized_after_truncation(self):
        # csv 6 位截断 → 和可能略偏 1，再归一后必须严格归一
        s = _convert(_row([0.010417, 0.0, 0.208333, 0.760417, 0.020833]))
        assert sum(s.answer_distribution.values()) == pytest.approx(1.0)

    def test_all_zero_distribution_rejected(self):
        with pytest.raises(ValueError, match="empty rating distribution"):
            _convert(_row([0.0] * 5))

    def test_plurality_breaks_to_level1(self):
        s = _convert(_row([0.9, 0.05, 0.05, 0.0, 0.0]))
        assert s.answer == "opt1"


class TestRegistry:
    def test_registered(self):
        adapter = get("koniq")
        assert isinstance(adapter, KoniqAdapter)
        assert adapter.meta.has_soft_labels is True


@pytest.mark.integration
class TestLoadIntegration:
    """真实拉取 metainfo csv（~2MB）。pytest -m integration（服务器上跑）"""

    def test_load_train_slice(self, tmp_path):
        adapter = KoniqAdapter(cache_dir=str(tmp_path))
        samples = list(adapter.load(split="train", limit=20))
        assert len(samples) == 20
        assert all(s.split == "train" for s in samples)
        assert all(s.answer_distribution is not None for s in samples)
