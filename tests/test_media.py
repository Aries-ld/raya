import base64
import io

import av
import numpy as np
import pytest
from PIL import Image

from raya_maas.config import Settings
from raya_maas.errors import APIError
from raya_maas.media import load_image, read_media, sample_video


@pytest.fixture
def settings():
    return Settings(_env_file=None)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://127.0.0.1/x",
        "https://example.org/x",
        "data:image/png;base64,invalid",
        "data:video/mp4;base64,YQ==",
    ],
)
def test_rejects_invalid_media_sources(settings, url):
    with pytest.raises(APIError):
        read_media(url, "image", settings)


def test_private_host_is_blocked(settings):
    settings.media_hosts = "127.0.0.1"
    with pytest.raises(APIError, match="Private"):
        read_media("https://127.0.0.1/image.png", "image", settings)


def test_image_decode_limits(settings):
    output = io.BytesIO()
    Image.new("RGB", (40, 30)).save(output, "PNG")
    url = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
    assert load_image(read_media(url, "image", settings), settings).size == (40, 30)
    settings.max_image_pixels = 20
    with pytest.raises(APIError, match="pixel"):
        load_image(output.getvalue(), settings)
    settings.max_media_bytes = 2
    with pytest.raises(APIError):
        read_media(url, "image", settings)


@pytest.fixture
def video():
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.codec_context.gop_size = 12
        for i in range(96):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i, dtype=np.uint8), "rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def test_sparse_decode_matches_sequential_and_has_eight_frames(settings, video):
    sparse = sample_video(video, settings)
    baseline = sample_video(video, settings, "sequential")
    assert sparse.frames.shape == (8, 48, 64, 3)
    assert sparse.indices == baseline.indices
    np.testing.assert_array_equal(sparse.frames, baseline.frames)
    assert sparse.decoded_frames < baseline.decoded_frames
    assert sparse.fps == 24
    assert sparse.duration == 4


def test_video_duration_budget(settings, video):
    settings.max_video_seconds = 1
    with pytest.raises(APIError, match="seconds"):
        sample_video(video, settings)


def test_invalid_video(settings):
    with pytest.raises(APIError):
        sample_video(b"not a video", settings)
