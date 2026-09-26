"""Audit the real Qwen image/video branches, then record reproducible video decision RTs."""

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import torch

from raya_maas.config import Settings
from raya_maas.engine import DecisionEngine
from raya_maas.local_test import load_request_file


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def trace_request(engine, request, modality):
    """Trace separately from timed runs so hooks and tensor copies don't inflate RT."""
    captured = {}
    inner = engine.model.model
    original_rope = inner.get_rope_index

    def capture_inputs(module, args, kwargs):
        captured["input_keys"] = sorted(kwargs)
        captured["shapes"] = {
            key: list(value.shape) for key, value in kwargs.items() if torch.is_tensor(value)
        }
        captured["grid"] = kwargs[f"{modality}_grid_thw"].detach().cpu().tolist()
        ids = kwargs["input_ids"].detach().cpu()
        captured["image_tokens"] = int((ids == engine.model.config.image_token_id).sum())
        captured["video_tokens"] = int((ids == engine.model.config.video_token_id).sum())
        captured["modality_ids"] = kwargs["mm_token_type_ids"].unique().cpu().tolist()
        text = engine.tokenizer.decode(ids[0], skip_special_tokens=False)
        captured["timestamps_seconds"] = [float(x) for x in re.findall(r"<([\d.]+) seconds>", text)]

    def capture_rope(*args, **kwargs):
        output = original_rope(*args, **kwargs)
        captured["mrope_position_shape"] = list(output[0].shape)
        return output

    hook = engine.model.register_forward_pre_hook(capture_inputs, with_kwargs=True)
    try:
        with (
            patch.object(inner, "get_rope_index", side_effect=capture_rope),
            patch.object(inner, "get_video_features", wraps=inner.get_video_features) as video,
            patch.object(inner, "get_image_features", wraps=inner.get_image_features) as image,
        ):
            response = engine.predict(request)
            captured["video_feature_calls"] = video.call_count
            captured["shared_vision_feature_calls"] = image.call_count
    finally:
        hook.remove()
    assert captured["shared_vision_feature_calls"] == 1
    assert captured["mrope_position_shape"][0] == 3
    expected_tokens = sum(t * h * w // 4 for t, h, w in captured["grid"])
    if modality == "video":
        assert captured["video_feature_calls"] == 1
        assert "pixel_values_videos" in captured["input_keys"]
        assert "pixel_values" not in captured["input_keys"]
        assert captured["video_tokens"] == expected_tokens
        assert captured["image_tokens"] == 0
        assert captured["modality_ids"] == [0, 2]
        metadata = engine.last_diagnostics["video"][0]
        assert captured["grid"][0][0] == metadata["processor_frames"] // 2
        indices, fps = list(metadata["sampled_indices"]), metadata["source_fps"]
        if len(indices) % 2:
            indices.append(indices[-1])
        expected_times = [
            float(f"{(indices[i] + indices[i + 1]) / (2 * fps):.1f}")
            for i in range(0, len(indices), 2)
        ]
        assert captured["timestamps_seconds"] == expected_times
        captured["sampling"] = metadata
    else:
        assert captured["video_feature_calls"] == 0
        assert "pixel_values" in captured["input_keys"]
        assert "pixel_values_videos" not in captured["input_keys"]
        assert captured["grid"][0][0] == 1
        assert captured["image_tokens"] == expected_tokens
        assert captured["video_tokens"] == 0
        assert captured["modality_ids"] == [0, 1]
        assert captured["timestamps_seconds"] == []
    captured["response"] = response
    return captured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cases", type=Path, default=Path("local_tests"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/video-fps1-audit.json"))
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    settings = Settings()
    video_request = load_request_file(args.cases / "video.json", settings)
    image_request = load_request_file(args.cases / "image.json", settings)
    engine = DecisionEngine(settings)
    try:
        engine.load()
        sync(engine.device)
        report = {
            "device": engine.device,
            "timing_definition": (
                "engine.predict to output; synchronized; excludes model load, file/base64 and HTTP"
            ),
            "runs": [],
        }
        # Do not warm up or trace before measuring: record first and subsequent requests explicitly.
        for index in range(args.runs):
            sync(engine.device)
            started = time.perf_counter()
            response = engine.predict(video_request)
            sync(engine.device)
            rt_ms = round((time.perf_counter() - started) * 1000, 3)
            row = {
                "run": index + 1,
                "response_ms": rt_ms,
                "correct": response["answers"]["food_in_box"]["choice"] == "fries",
                "response": response,
                "diagnostics": engine.last_diagnostics,
            }
            report["runs"].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        report["median_response_ms"] = statistics.median(r["response_ms"] for r in report["runs"])
        report["image_trace"] = trace_request(engine, image_request, "image")
        report["video_trace"] = trace_request(engine, video_request, "video")
        report["pipeline_checks_passed"] = True
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        assert all(row["correct"] for row in report["runs"]), (
            "Default video answer differs from expectation"
        )
        print(f"PASS: video branch, timestamps, modality IDs, grid/token alignment; {args.output}")
    finally:
        engine._release()


if __name__ == "__main__":
    main()
