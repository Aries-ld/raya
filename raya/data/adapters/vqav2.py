"""VQA v2 → 统一 schema（图，⭐ 人类软标签核心资产：每题 10 人标注分布）。

转换规则（与设计文档 §5.1/§5.7 对齐）：
- answer_type == "yes/no" → noul 原语，选项恒为 yes/no，分布 = 10 人中 yes/no 计数再归一
- 其他 answer_type → choice 原语，选项 = 标注者答案去重集（≥2 才成题；
  全员一致的题选项空间为 1，不构成决策 → 跳过，留给后续难负例挖掘阶段补干扰项）
- answer 一律取官方 multiple_choice_answer（10 人多数票）
- 数据源：visualqa.org 官方 S3 标注 JSON（⚠️ lmms-lab/VQAv2 只有 validation/test，
  无 train——M1 实测发现；官方 JSON 的字段与 lmms-lab 完全一致，train/val 同一条代码路径）

adapter 不下载/不落图片：media_path 只给规范路径，物化由 scripts/m1/download.py 负责。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from loguru import logger

from raya.data.registry import AdapterMeta, BaseAdapter, register
from raya.data.schema import DecisionSample, Option

_SOURCE = "vqav2"
_LICENSE = "CC-BY-4.0 (annotations) / COCO image terms"
# 官方 S3（questions + annotations 成对；test 无答案不支持）
_S3 = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa"
_ZIPS = {
    "train": (f"{_S3}/v2_Questions_Train_mscoco.zip", f"{_S3}/v2_Annotations_Train_mscoco.zip"),
    "validation": (f"{_S3}/v2_Questions_Val_mscoco.zip", f"{_S3}/v2_Annotations_Val_mscoco.zip"),
}

# HF split → COCO 目录名（VQA v2 train/val 基于 COCO 2014，test 基于 2015）
_COCO_DIR = {"train": "train2014", "validation": "val2014", "test": "test2015"}


def media_path_for(split: str, image_id: int) -> str:
    """单张图的规范存储路径（adapter 与 download 脚本共用此约定，防两处漂移）。"""
    return f"data/media/vqav2/{split}/COCO_{_COCO_DIR[split]}_{image_id:012d}.jpg"


def _normalize(answers: list[str]) -> list[str]:
    """剥掉空标注；保持原始大小写/措辞（人类分布原样保留，归一化是训练期的事）。"""
    return [a for a in answers if a and a.strip()]


def _convert(record: dict, split: str) -> Optional[DecisionSample]:
    """单条转换。返回 None = 该题不构成有效决策样本（跳过原因由调用侧统计）。"""
    raw = _normalize([a["answer"] for a in record["answers"]])
    mca = record["multiple_choice_answer"]
    base = dict(
        id=f"{_SOURCE}:{split}:{record['question_id']}",
        modality="image",
        media_path=media_path_for(split, record["image_id"]),
        source=_SOURCE,
        split=split,
        license=_LICENSE,
    )

    if record["answer_type"] == "yes/no":
        yes_count = sum(1 for a in raw if a.lower() == "yes")
        no_count = sum(1 for a in raw if a.lower() == "no")
        if yes_count + no_count == 0:
            return None
        total = yes_count + no_count
        answer = mca if mca in ("yes", "no") else ("yes" if yes_count >= no_count else "no")
        return DecisionSample(
            **base,
            question={"type": "noul", "text": record["question"]},
            options=[Option(id="opt1", text="yes"), Option(id="opt2", text="no")],
            answer="opt1" if answer == "yes" else "opt2",
            answer_distribution={"opt1": yes_count / total, "opt2": no_count / total},
        )

    # choice：选项 = 标注者答案去重集（保序），天然硬负例
    distinct = list(dict.fromkeys(raw))
    if len(distinct) < 2 or mca not in distinct:
        return None
    options = [Option(id=f"opt{i + 1}", text=t) for i, t in enumerate(distinct)]
    total = len(raw)
    return DecisionSample(
        **base,
        question={"type": "choice", "text": record["question"]},
        options=options,
        answer=options[distinct.index(mca)].id,
        answer_distribution={
            f"opt{i + 1}": raw.count(t) / total for i, t in enumerate(distinct)
        },
    )


class VQAv2Adapter(BaseAdapter):
    meta = AdapterMeta(
        name=_SOURCE,
        modality="image",
        license=_LICENSE,
        has_soft_labels=True,
        description="VQA v2（COCO 2014 图 + 10 人标注软分布），S8 图像锚 + RLCD 校准燃料",
    )

    def __init__(self, cache_dir: str = "data/hf_cache"):
        self.cache_dir = Path(cache_dir)

    def _load_records(self, split: str) -> list[dict]:
        """下载并合并 questions/annotations（按 question_id join，保持问题顺序）。"""
        import json
        import zipfile
        from urllib.request import urlretrieve

        zip_dir = self.cache_dir / "vqav2"
        zip_dir.mkdir(parents=True, exist_ok=True)
        payloads: list[dict] = []
        for url in _ZIPS[split]:
            zip_path = zip_dir / url.rsplit("/", 1)[-1]
            if not zip_path.exists():
                logger.info("vqav2 adapter: downloading {}", url)
                urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                payloads.append(json.loads(zf.read(zf.namelist()[0])))
        questions, annotations = payloads[0]["questions"], payloads[1]["annotations"]
        q_by_id = {q["question_id"]: q["question"] for q in questions}
        return [
            {**ann, "question": q_by_id[ann["question_id"]]}
            for ann in annotations
            if ann["question_id"] in q_by_id
        ]

    def load(self, split: str = "train", limit: int | None = None) -> Iterator[DecisionSample]:
        if split not in _ZIPS:
            raise ValueError(f"vqav2 split must be one of {sorted(_ZIPS)}, got '{split}'")
        records = self._load_records(split)
        yielded = skipped = 0
        for record in records:
            if limit is not None and yielded >= limit:
                break
            sample = _convert(record, split)
            if sample is None:
                skipped += 1
                continue
            yielded += 1
            yield sample
        logger.info(
            "vqav2 adapter: split={} yielded={} skipped={} (skip=全员一致/无效标注，留给难负例阶段)",
            split, yielded, skipped,
        )


register(VQAv2Adapter.meta, VQAv2Adapter)
