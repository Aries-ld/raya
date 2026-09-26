"""Compare sparse seek with bounded sequential decoding of the same eight timestamps."""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from raya_maas.config import Settings
from raya_maas.media import sample_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("artifacts/video-benchmark.json"))
    args = parser.parse_args()
    settings = Settings()
    rows = []
    for path in args.paths:
        data = path.read_bytes()
        by_strategy = {}
        for strategy in ("seek", "sequential"):
            samples = [sample_video(data, settings, strategy) for _ in range(args.runs)]
            by_strategy[strategy] = samples[-1]
            rows.append(
                {
                    "file": path.name,
                    "strategy": strategy,
                    "runs": args.runs,
                    "median_ms": round(statistics.median(x.decode_ms for x in samples), 3),
                    "decoded_frames": samples[-1].decoded_frames,
                    "sampled_frames": len(samples[-1].frames),
                    "indices": samples[-1].indices,
                }
            )
        sparse, sequential = by_strategy["seek"], by_strategy["sequential"]
        assert sparse.indices == sequential.indices, "Sampling timestamps changed"
        np.testing.assert_array_equal(sparse.frames, sequential.frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
