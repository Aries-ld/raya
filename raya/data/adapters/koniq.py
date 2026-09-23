"""KonIQ-10k 图像质量评估 → 统一 schema（图，score 原语 + 真人评分分布软标签）。

转换规则：
- 选项恒为 5 级质量序数（very poor → excellent），question.type = "score"
- answer_distribution = 真人评分直方图（csv 的 c1..c5，本身已是归一化比例）——又一校准燃料
- answer = 评分分布的众数档（plurality），与分布语义一致
- mos 浮点分不写入样本（会泄漏答案）；schema 的答案语义统一为 option id

图片物化（koniq10k.tgz ~6GB）由 scripts/m1/download.py 负责，adapter 只给规范路径。
"""

from __future__ import annotations

import csv
from typing import Iterator

from loguru import logger

from raya.data.registry import AdapterMeta, BaseAdapter, register
from raya.data.schema import DecisionSample, Option

_SOURCE = "koniq"
# KonIQ-10k 官方许可为非商用研究许可，且图片继承 YFCC100M 各自的 Flickr 许可——开源发布前的确权台账项
_LICENSE = "research-only (KonIQ-10k EULA / YFCC100M image licenses)"
_METAINFO_REPO = "chaofengc/IQA-Toolbox-Datasets-metainfo"
_METAINFO_FILE = "meta_info_KonIQ10kDataset.csv"

QUALITY_LEVELS = ["very poor", "poor", "fair", "good", "excellent"]
QUESTION_TEXT = "Rate the overall technical quality of this image."
_DIST_SUM_TOL = 0.02  # csv 比例有 6 位截断，容差放宽但仍校验量级


def _convert(row: dict) -> DecisionSample:
    dist_raw = [float(row[f"c{i}"]) for i in range(1, 6)]
    total = sum(dist_raw)
    if total <= 0:
        raise ValueError(f"koniq row {row['img_name']}: empty rating distribution")
    dist = [v / total for v in dist_raw]  # 再归一吸收 csv 截断误差
    options = [Option(id=f"opt{i + 1}", text=t) for i, t in enumerate(QUALITY_LEVELS)]
    plurality = max(range(5), key=lambda i: dist[i])
    return DecisionSample(
        id=f"{_SOURCE}:{row['official_split']}:{row['img_name']}",
        modality="image",
        media_path=f"data/media/koniq/{row['img_name']}",
        question={"type": "score", "text": QUESTION_TEXT},
        options=options,
        answer=options[plurality].id,
        answer_distribution={f"opt{i + 1}": dist[i] for i in range(5)},
        source=_SOURCE,
        split=row["official_split"],
        license=_LICENSE,
    )


class KoniqAdapter(BaseAdapter):
    meta = AdapterMeta(
        name=_SOURCE,
        modality="image",
        license=_LICENSE,
        has_soft_labels=True,
        description="KonIQ-10k 真实场景图像质量（~100 人/图评分分布），score 原语主力",
    )

    def __init__(self, cache_dir: str = "data/hf_cache"):
        self.cache_dir = cache_dir

    def load(self, split: str = "train", limit: int | None = None) -> Iterator[DecisionSample]:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            _METAINFO_REPO, _METAINFO_FILE, repo_type="dataset", cache_dir=self.cache_dir
        )
        yielded = skipped = 0
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["official_split"] != split:
                    continue
                if limit is not None and yielded >= limit:
                    break
                try:
                    yield _convert(row)
                    yielded += 1
                except (ValueError, KeyError) as e:
                    skipped += 1
                    logger.warning("koniq skip: {}", e)
        logger.info("koniq adapter: split={} yielded={} skipped={}", split, yielded, skipped)


register(KoniqAdapter.meta, KoniqAdapter)
