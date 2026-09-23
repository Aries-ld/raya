"""RACE adapter 测试：纯函数层单测（无网络）+ 集成层小样本实测（需网络，手动跑）。"""

import pytest

from raya.data.adapters.race import RaceAdapter, _convert
from raya.data.registry import get, list_adapters
from raya.data.schema import DecisionSample

_RECORD = {
    "example_id": "high100.txt:1",
    "article": "Tom went to the park yesterday. He played football with friends.",
    "question": "What did Tom do yesterday?",
    "options": ["Went swimming", "Played football", "Watched TV", "Slept all day"],
    "answer": "B",
}


class TestConvert:
    def test_happy_path(self):
        s = _convert(_RECORD, "train", 0)
        assert isinstance(s, DecisionSample)
        assert s.id == "race:train:high100.txt:1:0"  # 含题级序号：RACE 一文多题
        assert s.modality == "text"
        assert s.media_path is None
        assert s.state == _RECORD["article"]
        assert s.question.type == "choice"
        assert [o.id for o in s.options] == ["opt1", "opt2", "opt3", "opt4"]
        assert s.answer == "opt2"  # B → 第二个选项
        assert s.answer_distribution is None
        assert s.license == "research-only"

    def test_lineage_propagates_split(self):
        assert _convert(_RECORD, "validation", 0).split == "validation"

    def test_answer_letter_maps_to_correct_option_text(self):
        s = _convert(_RECORD, "train", 0)
        text_by_id = {o.id: o.text for o in s.options}
        assert text_by_id[s.answer] == "Played football"


class TestRegistry:
    def test_race_registered(self):
        names = [m.name for m in list_adapters()]
        assert "race" in names

    def test_get_returns_adapter(self):
        assert isinstance(get("race"), RaceAdapter)


@pytest.mark.integration
class TestLoadIntegration:
    """真实拉取（首次下载 ~100MB，走 HF cache）。手动跑：pytest -m integration"""

    def test_load_small_slice(self):
        # 用默认 cache（data/hf_cache 已由目检导出预置）：同一源一天内重复拉会被镜像限流
        adapter = RaceAdapter(config="middle")
        samples = list(adapter.load(split="train", limit=20))
        assert len(samples) == 20
        assert all(isinstance(s, DecisionSample) for s in samples)
        assert len({s.id for s in samples}) == 20  # id 全局唯一
