"""导出 adapter 前 N 条样本为 JSONL，供人工目检（M1 验收项：每集 20 条）。

用法：PYTHONPATH=. python scripts/m1/export_eyeball.py --adapter race --split train --limit 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

import raya.data.adapters.race  # noqa: F401  # 触发注册；新增 adapter 在此追加 import
import raya.data.adapters.vqav2  # noqa: F401
import raya.data.adapters.koniq  # noqa: F401
import raya.data.adapters.nextqa  # noqa: F401
from raya.data.registry import get

OUT_DIR = Path("data/eyeball")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--cache-dir", default="data/hf_cache")
    args = parser.parse_args()

    adapter = get(args.adapter)
    if hasattr(adapter, "cache_dir"):
        adapter.cache_dir = args.cache_dir
    # 目检只要 20 条：有 streaming 开关的 adapter 一律流式，不为抽检下载整集
    if hasattr(adapter, "streaming"):
        adapter.streaming = True

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.adapter}_{args.split}_{args.limit}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for sample in adapter.load(split=args.split, limit=args.limit):
            f.write(json.dumps(sample.model_dump(), ensure_ascii=False) + "\n")
    logger.info("eyeball export: {} -> {}", args.adapter, out_path)


if __name__ == "__main__":
    main()
