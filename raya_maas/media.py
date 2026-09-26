"""Bounded media input and sparse, timestamp-preserving eight-frame video decoding."""

import base64
import binascii
import io
import ipaddress
import math
import socket
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

import av
import httpx
import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import Settings
from .errors import APIError


def read_media(url: str, kind: str, settings: Settings) -> bytes:
    if url.startswith("data:"):
        header, separator, payload = url.partition(",")
        if (
            not separator
            or not header.startswith(f"data:{kind}/")
            or not header.endswith(";base64")
        ):
            raise APIError(f"Expected a base64 data:{kind}/... URL")
        if len(payload) > 4 * math.ceil(settings.max_media_bytes / 3):
            raise APIError("Media exceeds byte limit", 413, "payload_too_large")
        try:
            data = base64.b64decode(payload, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise APIError("Invalid base64 media") from exc
    else:
        try:
            parsed = urlsplit(url)
            port = parsed.port
        except ValueError as exc:
            raise APIError("Invalid media URL") from exc
        allowed = {host.strip().lower() for host in settings.media_hosts.split(",") if host.strip()}
        if (
            parsed.scheme != "https"
            or parsed.hostname not in allowed
            or parsed.username
            or parsed.password
            or port not in (None, 443)
        ):
            raise APIError("Use a data URL or HTTPS URL on RAYA_MEDIA_HOSTS")
        try:
            addresses = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
            if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
                raise APIError("Private/reserved media destinations are prohibited")
            # Administrators must only allow trusted hosts and enforce outbound network policy.
            # No proxy environment, redirects, local file access, or credentials are forwarded.
            start = time.monotonic()
            with httpx.Client(timeout=settings.media_timeout, trust_env=False) as client:
                with client.stream("GET", url, follow_redirects=False) as response:
                    if response.status_code != 200:
                        raise APIError("Media server did not return HTTP 200")
                    data = bytearray()
                    for chunk in response.iter_bytes(65536):
                        if len(data) + len(chunk) > settings.max_media_bytes:
                            raise APIError("Media exceeds byte limit", 413, "payload_too_large")
                        if time.monotonic() - start > settings.media_timeout:
                            raise APIError("Media download timed out", 422, "media_timeout")
                        data.extend(chunk)
                    data = bytes(data)
        except (httpx.HTTPError, OSError) as exc:
            raise APIError("Media could not be downloaded") from exc
    if not data or len(data) > settings.max_media_bytes:
        raise APIError("Empty or oversized media", 413, "payload_too_large")
    return data


def load_image(data: bytes, settings: Settings) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > settings.max_image_pixels:
                raise APIError("Image exceeds pixel limit")
            image = source.convert("RGB")
            width, height = bounded_size(image.width, image.height, settings.image_max_pixels)
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            return image
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise APIError("Invalid image") from exc


def bounded_size(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    scale = min(1.0, math.sqrt(max_pixels / (width * height)))
    return max(1, int(width * scale)), max(1, int(height * scale))


@dataclass
class VideoSample:
    frames: np.ndarray
    fps: float
    total_frames: int
    duration: float
    indices: list[int]
    decoded_frames: int
    strategy: str
    decode_ms: float

    def metadata(self):
        from transformers.video_utils import VideoMetadata

        return VideoMetadata(
            total_num_frames=self.total_frames,
            fps=self.fps,
            duration=self.duration,
            frames_indices=self.indices,
        )


def sample_video(data: bytes, settings: Settings, strategy: str = "seek") -> VideoSample:
    """Decode only GOPs around eight uniform timestamps, RGB-convert only selected frames.

    FFmpeg scales selected frames before conversion. Sequential fallback retains only
    eight frames, has a frame/time budget, and never materializes the complete video.
    """
    started = time.monotonic()
    try:
        return _sample(data, settings, strategy, started)
    except APIError:
        raise
    except (av.FFmpegError, ValueError, EOFError):
        if strategy == "seek":
            try:
                return _sample(data, settings, "sequential", started)
            except APIError:
                raise
            except (av.FFmpegError, ValueError, EOFError) as exc:
                raise APIError("Video could not be decoded") from exc
        raise APIError("Video could not be decoded") from None


def _sample(data: bytes, settings: Settings, strategy: str, started: float) -> VideoSample:
    with av.open(io.BytesIO(data)) as container:
        if not container.streams.video:
            raise APIError("Media contains no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "SLICE"
        stream.codec_context.thread_count = 2
        fps = float(stream.average_rate or stream.guessed_rate or 0)
        duration = float(stream.duration * stream.time_base) if stream.duration else 0
        if not duration and container.duration:
            duration = container.duration / av.time_base
        if not fps or not math.isfinite(fps) or not duration or not math.isfinite(duration):
            raise APIError("Video requires valid frame rate and duration metadata")
        if duration > settings.max_video_seconds:
            raise APIError(f"Video exceeds {settings.max_video_seconds:g} seconds")
        if stream.width * stream.height > settings.max_image_pixels:
            raise APIError("Video frame exceeds pixel limit")
        total = stream.frames or max(1, round(duration * fps))
        targets = np.linspace(0, max(0, duration - 1 / fps), 8)
        origin = float((stream.start_time or 0) * stream.time_base)
        width, height = bounded_size(stream.width, stream.height, settings.video_max_pixels)
        frames, indices = [], []
        decoded = 0

        def check_budget():
            if decoded > settings.max_video_decode_frames:
                raise APIError("Video exceeded decode frame budget")
            if time.monotonic() - started > settings.media_timeout:
                raise APIError("Video decode timed out", 422, "media_timeout")

        def save(frame, timestamp):
            frames.append(frame.reformat(width=width, height=height, format="rgb24").to_ndarray())
            indices.append(max(0, round(timestamp * fps)))

        if strategy == "seek":
            for target in targets:
                check_budget()
                container.seek(
                    int((origin + target) / stream.time_base),
                    stream=stream,
                    backward=True,
                    any_frame=False,
                )
                selected = None
                timestamp = 0.0
                for frame in container.decode(stream):
                    decoded += 1
                    check_budget()
                    if frame.pts is None:
                        continue
                    timestamp = float(frame.pts * stream.time_base) - origin
                    selected = frame
                    if timestamp + 0.5 / fps >= target:
                        break
                if selected is None:
                    raise ValueError("Seek produced no frame")
                save(selected, timestamp)
        else:
            for frame in container.decode(stream):
                decoded += 1
                check_budget()
                timestamp = (
                    float(frame.pts * stream.time_base) - origin
                    if frame.pts is not None
                    else (decoded - 1) / fps
                )
                while len(frames) < 8 and timestamp + 0.5 / fps >= targets[len(frames)]:
                    save(frame, timestamp)
                if len(frames) == 8:
                    break
            if not frames:
                raise ValueError("No video frames")
            while len(frames) < 8:
                frames.append(frames[-1].copy())
                indices.append(indices[-1])
        return VideoSample(
            np.stack(frames),
            fps,
            total,
            duration,
            indices,
            decoded,
            strategy,
            round((time.monotonic() - started) * 1000, 3),
        )
