import base64
import io
from functools import lru_cache

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from raya_maas.app import create_app
from raya_maas.config import Settings
from raya_maas.decisions import evaluate
from raya_maas.engine import DecisionEngine
from raya_maas.errors import APIError
from raya_maas.media import sample_video


@lru_cache
def video_bytes(seconds, gop=10):
    output = io.BytesIO()
    with av.open(output, "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.codec_context.gop_size = gop
        for i in range(round(seconds * 10)):
            frame = av.VideoFrame.from_ndarray(
                np.full((48, 64, 3), i % 256, dtype=np.uint8), "rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()


@pytest.mark.parametrize("seconds", [3, 20])
def test_inclusive_video_duration_boundaries(seconds):
    sample = sample_video(video_bytes(seconds), Settings(_env_file=None))
    assert sample.duration == seconds
    assert sample.frames.shape[0] == max(seconds, 4)


@pytest.mark.parametrize("seconds", [2.9, 20.1])
def test_outside_video_duration_boundaries(seconds):
    with pytest.raises(APIError) as caught:
        sample_video(video_bytes(seconds), Settings(_env_file=None))
    assert caught.value.status == 422
    assert caught.value.code == "video_duration_out_of_range"


def test_auto_decoder_avoids_repeated_long_gop_decoding():
    settings = Settings(_env_file=None)
    data = video_bytes(10, gop=100)
    automatic = sample_video(data, settings)
    forced_seek = sample_video(data, settings, "seek")
    sequential = sample_video(data, settings, "sequential")
    assert automatic.strategy == "sequential"
    assert automatic.decoded_frames < forced_seek.decoded_frames
    assert automatic.indices == forced_seek.indices == sequential.indices
    np.testing.assert_array_equal(automatic.frames, forced_seek.frames)
    np.testing.assert_array_equal(automatic.frames, sequential.frames)


@pytest.mark.parametrize("seconds", [3, 5, 10, 20])
def test_one_fps_matches_qwen_sampler_and_preserves_source_metadata(seconds):
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    sample = sample_video(video_bytes(seconds), Settings(_env_file=None))
    processor = Qwen3VLVideoProcessor()
    metadata = sample.metadata()
    expected = processor.sample_frames(metadata, fps=1, num_frames=None).tolist()
    assert sample.indices == expected
    assert sample.fps == 10  # source fps; never overwrite it with sampling fps=1.
    assert sample.sampling_fps == 1
    # Qwen's timestamp builder pads odd index lists in place. Keep our source diagnostic intact.
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    Qwen3VLProcessor._calculate_timestamps(None, metadata.frames_indices, metadata.fps, 2)
    assert sample.indices == expected
    assert len(sample.frames) == len(expected)


def test_odd_sampling_count_is_padded_only_by_video_processor():
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    sample = sample_video(video_bytes(5), Settings(_env_file=None))
    assert len(sample.frames) == 5
    processor = Qwen3VLVideoProcessor(size={"shortest_edge": 4096, "longest_edge": 65536})
    encoded = processor(
        videos=[sample.frames],
        video_metadata=[sample.metadata()],
        do_sample_frames=False,
        fps=1,
        num_frames=None,
        cap_pixels_per_frame=True,
        return_tensors="pt",
    )
    assert encoded["video_grid_thw"][0, 0].item() == 3  # five sampled + one repeated padding frame.
    assert len(sample.indices) == 5


def test_hardware_failure_falls_back_to_software(monkeypatch):
    from av.codec import hwaccel

    from raya_maas import media

    original = media._sample
    attempts = []

    def decode(data, settings, strategy, started, backend, num_frames):
        attempts.append(backend)
        if backend != "software":
            raise RuntimeError("Hardware decoder unavailable")
        return original(data, settings, strategy, started, backend, num_frames)

    monkeypatch.setattr(hwaccel, "hwdevices_available", lambda: ["videotoolbox"])
    monkeypatch.setattr(media, "_sample", decode)
    sample = sample_video(video_bytes(3), Settings(video_decoder="auto", _env_file=None))
    assert attempts == ["videotoolbox", "software"]
    assert sample.decoder == "software"


@pytest.mark.parametrize(
    "config",
    [
        {"min_video_seconds": 2},
        {"max_video_seconds": 21},
        {"min_video_seconds": 10, "max_video_seconds": 5},
    ],
)
def test_service_configuration_cannot_relax_api_duration_contract(config):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **config)


@pytest.mark.parametrize("seconds,expected_status", [(2.9, 422), (3, 200), (20, 200), (20.1, 422)])
def test_http_checks_real_media_duration_before_model_forward(seconds, expected_status):
    class ProbeEngine:
        def __init__(self):
            self.settings = Settings(_env_file=None)
            self.tokenizer = type("Tokenizer", (), {"encode": lambda self, value: [1]})()
            self.forward_calls = 0

        def load(self):
            pass

        def prepare_media(self, media):
            return DecisionEngine.prepare_media(self, media)

        def predict(self, request, *, prepared_media=None):
            if prepared_media is None:
                return evaluate(self, request)
            self.forward_calls += 1
            return {
                "index": 0,
                "probabilities": {"A": 0.8, "B": 0.2},
                "prompt_tokens": 1,
                "device": "test",
                "timing_ms": {"total": 1},
            }

    engine = ProbeEngine()
    data = {
        "model": "raya-decision-v1",
        "state": {
            "text": "Check the video",
            "media": [
                {
                    "type": "video",
                    "url": "data:video/mp4;base64,"
                    + base64.b64encode(video_bytes(seconds)).decode(),
                }
            ],
        },
        "questions": {
            "check": {
                "type": "choice",
                "instructions": "What is shown?",
                "criteria": {"yes": "Yes", "no": "No"},
            }
        },
    }
    with TestClient(create_app(Settings(api_keys="test", _env_file=None), engine)) as client:
        response = client.post("/v1/systemone", json=data, headers={"Authorization": "Bearer test"})
    assert response.status_code == expected_status
    if expected_status == 422:
        assert response.json()["error"]["code"] == "video_duration_out_of_range"
        assert engine.forward_calls == 0
    else:
        assert engine.forward_calls == 1
