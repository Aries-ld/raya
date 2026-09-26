"""Copy only three existing bench samples; never download or retain training datasets."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-root",
        type=Path,
        required=True,
        help="Existing mneme checkout containing bench media",
    )
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures"))
    args = parser.parse_args()
    sources = {
        "m3bench_living.jpg": "docs/figures/m3bench-real-v2/assets/living.jpg",
        "m3bench_kitchen.mp4": "联调/ingest6_runs/v6_kitchen15/kitchen_15/clips/clip_00.mp4",
        "m3bench_living.mp4": "联调/workbench_runs/"
        "living_room_22_0s_1clip_v2_20260714-194718/clips/clip000.mp4",
    }
    for relative in sources.values():
        if not (args.bench_root / relative).is_file():
            raise SystemExit(f"Missing existing sample: {args.bench_root / relative}")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, relative in sources.items():
        source, destination = args.bench_root / relative, args.output / name
        entry = {"file": name, "source": str(source.resolve())}
        if name == "m3bench_living.mp4":
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source),
                    "-t",
                    "12",
                    "-an",
                    "-c:v",
                    "copy",
                    str(destination),
                ],
                check=True,
            )
            entry["transform"] = "First 12 seconds, stream copy, audio removed"
        else:
            shutil.copyfile(source, destination)
        entry["sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
        manifest.append(entry)
    (args.output / "provenance.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Prepared three fixtures in {args.output}")


if __name__ == "__main__":
    main()
