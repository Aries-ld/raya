"""Compare fixed-8 and 1fps on labelled local bench clips; no inference result cache."""

import argparse
import json
import statistics
import time
from pathlib import Path
from unittest.mock import patch

from raya_maas.config import Settings
from raya_maas.engine import DecisionEngine
from raya_maas.local_test import load_request_file
from raya_maas.media import sample_video
from scripts.audit_video import sync, trace_request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/fps1-cases/manifest.json"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("artifacts/fps1-benchmark.json"))
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    cases = json.loads(args.manifest.read_text())
    settings = Settings()
    engine = DecisionEngine(settings)
    report = {
        "device": None,
        "decoder": settings.video_decoder,
        "decoder_threads": settings.video_decode_threads,
        "runs_per_mode": args.runs,
        "timing": "engine request to output; no model load, file/base64 wrapping or network",
        "cases": [],
        "traces": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        engine.load()
        sync(engine.device)
        report["device"] = engine.device
        for case in cases:
            request = load_request_file(args.manifest.parent / case["request"], settings)
            item = {**case, "modes": {}}
            report["cases"].append(item)
            for mode in ("fixed8", "fps1"):
                rows = []

                def decode(data, settings):
                    return sample_video(data, settings, num_frames=8 if mode == "fixed8" else None)

                with patch("raya_maas.engine.sample_video", side_effect=decode):
                    for _ in range(args.runs):
                        sync(engine.device)
                        start = time.perf_counter()
                        response = engine.predict(request)
                        sync(engine.device)
                        rt = round((time.perf_counter() - start) * 1000, 3)
                        answer = response["answers"]["decision"]
                        assert abs(sum(answer["probabilities"].values()) - 1) < 1e-5
                        rows.append(
                            {
                                "rt_ms": rt,
                                "correct": answer["choice"] == case["expected"],
                                "response": response,
                                "diagnostics": engine.last_diagnostics,
                            }
                        )
                summary = {
                    "median_rt_ms": statistics.median(row["rt_ms"] for row in rows),
                    "median_decode_ms": statistics.median(
                        row["diagnostics"]["video"][0]["decode_ms"] for row in rows
                    ),
                    "median_forward_ms": statistics.median(
                        row["diagnostics"]["questions"]["decision"]["timing_ms"]["forward"]
                        for row in rows
                    ),
                    "all_correct": all(row["correct"] for row in rows),
                    "sampled_frames": rows[0]["diagnostics"]["video"][0]["sampled_frames"],
                    "processor_frames": rows[0]["diagnostics"]["video"][0]["processor_frames"],
                    "tokens": rows[0]["response"]["usage"]["input_tokens"],
                }
                item["modes"][mode] = {"summary": summary, "runs": rows}
                print(case["id"], mode, json.dumps(summary), flush=True)
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            if case["id"] in ("food-3s", "food-5s", "food-20s"):
                report["traces"][case["id"]] = trace_request(engine, request, "video")
        report["traces"]["image"] = trace_request(
            engine, load_request_file(Path("local_tests/image.json"), settings), "image"
        )
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"Completed {len(cases)} clips with both policies; {args.output}")
    finally:
        engine._release()


if __name__ == "__main__":
    main()
