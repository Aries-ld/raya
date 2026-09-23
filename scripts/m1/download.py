"""M1 媒体物化：adapter 只给规范路径，这里把原始媒体真正落盘（与 adapter 的路径约定严格一致）。

用法（服务器上跑，数据盘 /root/autodl-tmp/raya；务必先 source .env 拿 HF_ENDPOINT）：
  PYTHONPATH=. python scripts/m1/download.py koniq                 # KonIQ 512x384 → data/media/koniq/
  PYTHONPATH=. python scripts/m1/download.py vqav2 --split train   # COCO 2014 → data/media/vqav2/train/
  PYTHONPATH=. python scripts/m1/download.py nextqa                # NExT-QA 视频（Google Drive，可能被墙则本机中转）

纪律：只物化 adapter 引用的媒体；中间产物（zip/tgz）解压后即删，防数据盘膨胀。
"""

from __future__ import annotations

import argparse
import subprocess
import tarfile
import zipfile
from pathlib import Path

from loguru import logger

MEDIA_ROOT = Path("data/media")


def _extract_flat(archive: Path, members_prefix: str, out_dir: Path, kind: str) -> int:
    """只解指定前缀的成员，拍平为 basename 落 out_dir，返回文件数。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    opener = tarfile.open if kind == "tgz" else zipfile.ZipFile
    count = 0
    with opener(archive) as zf:  # type: ignore[arg-type]
        names = (
            m for m in zf.getmembers()
            if (m.name if hasattr(m, "name") else m).startswith(members_prefix)
        )
        for member in names:
            name = member.name if hasattr(member, "name") else member
            if name.endswith("/"):
                continue
            src = zf.extractfile(member)
            if src is None:
                continue
            (out_dir / Path(name).name).write_bytes(src.read())
            count += 1
    return count


def download_koniq() -> None:
    from huggingface_hub import snapshot_download

    tgz_dir = snapshot_download(
        "chaofengc/IQA-Toolbox-Datasets", repo_type="dataset",
        allow_patterns="koniq10k.tgz", local_dir="data/hf_cache/iqa_toolbox",
    )
    tgz = Path(tgz_dir) / "koniq10k.tgz"
    n = _extract_flat(tgz, "koniq10k/512x384/", MEDIA_ROOT / "koniq", "tgz")
    logger.info("koniq materialized: {} images -> {}", n, MEDIA_ROOT / "koniq")
    tgz.unlink()  # 解压即删，~6GB 中间产物不留
    logger.info("koniq intermediate tgz deleted")


def download_vqav2(split: str) -> None:
    coco_name = {"train": "train2014", "validation": "val2014"}[split]
    url = f"http://images.cocodataset.org/zips/{coco_name}.zip"
    zip_path = Path("data/hf_cache") / f"{coco_name}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        logger.info("vqav2: downloading {} (COCO {})", url, coco_name)
        subprocess.run(["curl", "-L", "--retry", "3", "-o", str(zip_path), url], check=True)
    n = _extract_flat(zip_path, f"{coco_name}/", MEDIA_ROOT / "vqav2" / split, "zip")
    logger.info("vqav2 materialized: {} images -> {}", n, MEDIA_ROOT / "vqav2" / split)
    zip_path.unlink()
    logger.info("vqav2 intermediate zip deleted")


def download_nextqa() -> None:
    zip_path = Path("data/hf_cache/nextqa_videos.zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        # Google Drive 大文件；服务器（北京）可能被墙，此时在本机（SG）gdown 后 rsync 上来
        file_id = "1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH"
        logger.info("nextqa: gdown file_id={}", file_id)
        subprocess.run(
            ["gdown", "--fuzzy", f"https://drive.google.com/uc?id={file_id}", "-O", str(zip_path)],
            check=True,
        )
    out_dir = MEDIA_ROOT / "nextqa"
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    # zip 内可能有嵌套目录，拍平 {video}.mp4 到规范路径
    moved = 0
    for mp4 in out_dir.rglob("*.mp4"):
        target = out_dir / mp4.name
        if mp4 != target:
            mp4.replace(target)
            moved += 1
    logger.info("nextqa materialized: {} videos flattened -> {}", moved, out_dir)
    zip_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter", choices=["koniq", "vqav2", "nextqa"])
    parser.add_argument("--split", default="train", help="vqav2 only: train | validation")
    args = parser.parse_args()
    if args.adapter == "koniq":
        download_koniq()
    elif args.adapter == "vqav2":
        download_vqav2(args.split)
    else:
        download_nextqa()


if __name__ == "__main__":
    main()
