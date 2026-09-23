"""NExT-QA 视频问答 → 统一 schema（视频 choice，5 选项 MC，S8 视频主力之一）。

数据源：官方 GitHub 标注 csv（train/val；test 无答案不可用）。
字段：video, frame_count, width, height, question, answer(0-4), qid, type, a0..a4。
视频物化（官方 zip，~7.5GB）由 scripts/m1/download.py 负责；VidOR 源有部分视频死链，
物化阶段统计覆盖率，缺失视频的样本在训练加载时跳过（不在 adapter 层做文件存在性 IO）。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator
from urllib.request import urlretrieve

from loguru import logger

from raya.data.registry import AdapterMeta, BaseAdapter, register
from raya.data.schema import DecisionSample, Option

_SOURCE = "nextqa"
_LICENSE = "research-only (NExT-QA / VidOR terms)"
_CSV_URL = "https://raw.githubusercontent.com/doc-doc/NExT-QA/main/dataset/nextqa/{split}.csv"
_VALID_SPLITS = ("train", "val")


def _convert(row: dict, split: str) -> DecisionSample:
    options = [Option(id=f"opt{i + 1}", text=row[f"a{i}"]) for i in range(5)]
    answer_idx = int(row["answer"])
    return DecisionSample(
        id=f"{_SOURCE}:{split}:{row['video']}:{row['qid']}",
        modality="video",
        media_path=f"data/media/nextqa/{row['video']}.mp4",
        question={"type": "choice", "text": row["question"]},
        options=options,
        answer=options[answer_idx].id,
        answer_distribution=None,
        source=_SOURCE,
        split=split,
        license=_LICENSE,
    )


class NextQAAdapter(BaseAdapter):
    meta = AdapterMeta(
        name=_SOURCE,
        modality="video",
        license=_LICENSE,
        has_soft_labels=False,
        description="NExT-QA 因果/时序视频 MC（train 34K），S8 视频锚；软标签走 M3 教师补",
    )

    def __init__(self, cache_dir: str = "data/hf_cache"):
        self.cache_dir = Path(cache_dir)

    def _csv_path(self, split: str) -> Path:
        # 调用方可能用 str 覆盖 cache_dir（如目检脚本），这里统一再包一层 Path
        path = Path(self.cache_dir) / "nextqa" / f"{split}.csv"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("nextqa adapter: downloading {} csv", split)
            urlretrieve(_CSV_URL.format(split=split), path)
        return path

    def load(self, split: str = "train", limit: int | None = None) -> Iterator[DecisionSample]:
        if split not in _VALID_SPLITS:
            raise ValueError(f"nextqa split must be one of {_VALID_SPLITS}, got '{split}'")
        with self._csv_path(split).open(newline="") as f:
            rows = list(csv.DictReader(f))
        n = len(rows) if limit is None else min(limit, len(rows))
        logger.info("nextqa adapter: split={} yielding {} / {} records", split, n, len(rows))
        for row in rows[:n]:
            yield _convert(row, split)


register(NextQAAdapter.meta, NextQAAdapter)
