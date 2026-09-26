"""Recreate the small manually inspected video set used for sampling-policy comparison."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/fps1-cases"))
    args = parser.parse_args()
    sources = {
        "food": "联调/workbench_runs/bedroom_01_0s_20clip_v2_20260713-155631/clips/clip002.mp4",
        "kitchen": "联调/ingest6_runs/v6_kitchen15/kitchen_15/clips/clip_00.mp4",
        "living": (
            "联调/workbench_runs/living_room_22_0s_1clip_v2_20260714-194718/clips/clip000.mp4"
        ),
    }
    specs = {
        "food": (
            "视频中红色纸盒里装着什么食物？",
            {"fries": "薯条", "dumplings": "饺子", "sushi": "寿司", "noodle_soup": "汤面"},
            "fries",
        ),
        # Source ID is kitchen_15, but the opening frames visibly show a bed.
        "kitchen": (
            "视频开头主要在什么场所？",
            {
                "bedroom": "卧室，可以看到床",
                "kitchen": "厨房，有灶台或厨房操作台",
                "beach": "海滩",
                "street": "街道",
            },
            "bedroom",
        ),
        "living": (
            "视频主要发生在什么环境？",
            {"indoor": "室内房间", "beach": "海滩", "forest": "森林", "street": "街道"},
            "indoor",
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for name, lengths in [("food", [3, 5, 10, 20]), ("kitchen", [3, 20]), ("living", [3, 20])]:
        source = (args.bench_root / sources[name]).resolve()
        if not source.is_file():
            parser.error(f"Missing existing bench video: {source}")
        for seconds in lengths:
            case_id = f"{name}-{seconds}s"
            path = args.output / f"{case_id}.mp4"
            start = 8 if name == "food" and seconds < 10 else 0
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    str(start),
                    "-i",
                    str(source),
                    "-t",
                    str(seconds),
                    "-an",
                    "-vf",
                    "fps=30",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(path),
                ],
                check=True,
            )
            question, criteria, expected = specs[name]
            request = {
                "model": "raya-decision-v1",
                "state": {
                    "text": "请根据视频画面作判断。",
                    "media": [{"type": "video", "url": f"file:./{path.name}"}],
                },
                "questions": {
                    "decision": {"type": "choice", "instructions": question, "criteria": criteria}
                },
            }
            request_path = args.output / f"{case_id}.json"
            request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n")
            cases.append(
                {
                    "id": case_id,
                    "request": request_path.name,
                    "source": str(source),
                    "source_start_seconds": start,
                    "duration_seconds": seconds,
                    "expected": expected,
                    "label_basis": "Manually inspected video frames",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    (args.output / "manifest.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Prepared {len(cases)} clips in {args.output}; training overlap is unknown")


if __name__ == "__main__":
    main()
