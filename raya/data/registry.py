"""adapter 注册表：新增数据集 = 新增一个 adapter 类并 @register，不改框架代码（开闭原则）。

adapter 职责边界：只负责「源数据 → DecisionSample 迭代器」，
下载/抽帧/缓存由 scripts/m1/download.py 统一管，adapter 不做 IO 优化。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

from raya.data.schema import DecisionSample, Modality


@dataclass(frozen=True)
class AdapterMeta:
    name: str
    modality: Modality
    license: str
    has_soft_labels: bool
    description: str = ""


# (meta, factory) 二元组；factory 惰性构造 adapter，避免 import 即触发重依赖
_REGISTRY: dict[str, tuple[AdapterMeta, Callable[[], "BaseAdapter"]]] = {}


class BaseAdapter:
    """所有 adapter 的协议。load 产出经过 schema 校验的样本流。"""

    meta: AdapterMeta

    def load(self, split: str = "train", limit: int | None = None) -> Iterator[DecisionSample]:
        raise NotImplementedError


def register(meta: AdapterMeta, factory: Callable[[], BaseAdapter]) -> None:
    if meta.name in _REGISTRY:
        raise ValueError(f"adapter '{meta.name}' already registered")
    _REGISTRY[meta.name] = (meta, factory)


def get(name: str) -> BaseAdapter:
    if name not in _REGISTRY:
        raise KeyError(f"unknown adapter '{name}', available: {sorted(_REGISTRY)}")
    return _REGISTRY[name][1]()


def list_adapters() -> list[AdapterMeta]:
    return [meta for meta, _ in _REGISTRY.values()]
