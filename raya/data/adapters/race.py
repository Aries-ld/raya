"""RACE 英语阅读理解 → 统一 schema（文本 choice，首个 adapter，定 adapter 写法范式）。

分层：`_convert()` 是纯函数（单条 HF record → DecisionSample），单测不碰网络；
`RaceAdapter.load()` 只负责拉数据 + 逐条转换 + 限速。
"""

from __future__ import annotations

from typing import Iterator

from loguru import logger

from raya.data.registry import AdapterMeta, BaseAdapter, register
from raya.data.schema import DecisionSample, Option

_SOURCE = "race"
# RACE 官方仅声明研究用途，商用前需确权（设计文档 §5.7 红线台账）
_LICENSE = "research-only"
_LETTERS = "ABCDEFGH"


def _convert(record: dict, split: str, seq: int) -> DecisionSample:
    # RACE 的 example_id 是文章级（一文多题），必须带序号才是题级唯一
    options = [
        Option(id=f"opt{i + 1}", text=text)
        for i, text in enumerate(record["options"])
    ]
    answer_idx = _LETTERS.index(record["answer"])
    return DecisionSample(
        id=f"{_SOURCE}:{split}:{record['example_id']}:{seq}",
        modality="text",
        state=record["article"],
        question={"type": "choice", "text": record["question"]},
        options=options,
        answer=options[answer_idx].id,
        answer_distribution=None,
        source=_SOURCE,
        split=split,
        license=_LICENSE,
    )


class RaceAdapter(BaseAdapter):
    meta = AdapterMeta(
        name=_SOURCE,
        modality="text",
        license=_LICENSE,
        has_soft_labels=False,
        description="RACE 英语阅读理解 MC（高中+初中），S8 通用锚文本主力",
    )

    def __init__(self, cache_dir: str = "data/hf_cache", config: str = "all"):
        self.cache_dir = cache_dir
        self.config = config

    def load(self, split: str = "train", limit: int | None = None) -> Iterator[DecisionSample]:
        from datasets import load_dataset  # 惰性导入：adapter 注册不拖重依赖

        ds = load_dataset("ehovy/race", self.config, split=split, cache_dir=self.cache_dir)
        n = len(ds) if limit is None else min(limit, len(ds))
        logger.info("race adapter: split={} config={} yielding {} / {} records", split, self.config, n, len(ds))
        for i in range(n):
            yield _convert(ds[i], split, i)


register(RaceAdapter.meta, RaceAdapter)
