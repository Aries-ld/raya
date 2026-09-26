"""Bounded media input and timestamp-preserving video decoding at Qwen's 1fps setting."""

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
    decoder: str
    sampling_fps: float | None

    def metadata(self):
        from transformers.video_utils import VideoMetadata

        return VideoMetadata(
            total_num_frames=self.total_frames,
            fps=self.fps,
            duration=self.duration,
            frames_indices=list(self.indices),  # The processor pads odd lists in place.
        )


def sample_video(
    data: bytes, settings: Settings, strategy: str = "auto", *, num_frames: int | None = None
) -> VideoSample:
    """Sample at Qwen-compatible 1fps; num_frames is for offline comparison with old sampling.

    Hardware decoding falls back to software in auto mode. Only selected frames are RGB
    converted; neither decoder materializes the complete video in memory.
    """
    from av.codec.hwaccel import hwdevices_available

    started = time.monotonic()
    backends = [settings.video_decoder]
    if settings.video_decoder == "auto":
        available = hwdevices_available()
        backends = [name for name in ("videotoolbox", "cuda") if name in available] + ["software"]
    for backend in backends:
        try:
            return _sample(data, settings, strategy, started, backend, num_frames)
        except APIError:
            raise
        except (av.FFmpegError, ValueError, EOFError, RuntimeError):
            if backend != "software" and settings.video_decoder == "auto":
                continue
            if strategy not in ("seek", "auto"):
                raise APIError("Video could not be decoded") from None
            try:
                return _sample(data, settings, "sequential", started, backend, num_frames)
            except APIError:
                raise
            except (av.FFmpegError, ValueError, EOFError, RuntimeError) as exc:
                raise APIError("Video could not be decoded") from exc
    raise APIError("No video decoder available")


def _sample(data, settings, strategy, started, backend, num_frames) -> VideoSample:
    from av.codec.hwaccel import HWAccel

    acceleration = (
        None
        if backend == "software"
        else HWAccel(
            backend,
            allow_software_fallback=False,
            is_hw_owned=True,
        )
    )
    with av.open(io.BytesIO(data), hwaccel=acceleration) as container:
        if not container.streams.video:
            raise APIError("Media contains no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = settings.video_decode_threads
        fps = float(stream.average_rate or stream.guessed_rate or 0)
        duration = float(stream.duration * stream.time_base) if stream.duration else 0
        if not duration and container.duration:
            duration = container.duration / av.time_base
        if not fps or not math.isfinite(fps) or not duration or not math.isfinite(duration):
            raise APIError("Video requires valid frame rate and duration metadata")
        # The service measures the video stream, not a client-supplied duration field.
        # A microsecond tolerance covers time-base float rounding, not extra frames.
        if (
            duration < settings.min_video_seconds - 1e-6
            or duration > settings.max_video_seconds + 1e-6
        ):
            raise APIError(
                f"Video duration {duration:.6f}s is outside the allowed inclusive range "
                f"{settings.min_video_seconds:g}–{settings.max_video_seconds:g} seconds",
                422,
                "video_duration_out_of_range",
            )
        if stream.width * stream.height > settings.max_image_pixels:
            raise APIError("Video frame exceeds pixel limit")
        total = stream.frames or max(1, round(duration * fps))
        # Match Qwen3VLVideoProcessor.sample_frames(fps=1, num_frames=None): at least
        # four frames when the source permits. Odd counts are padded by the processor.
        count = min(max(int(total / fps), 4), 768, total) if num_frames is None else num_frames
        target_indices = np.linspace(0, total - 1, count).round().astype(int)
        targets = target_indices / fps
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

        if strategy == "auto":
            # Reading compressed packet flags is cheap; no frames are decoded here.
            # Long GOPs can make independent seeks decode the same region repeatedly.
            key_times = [0.0]
            for packet_count, packet in enumerate(container.demux(stream), start=1):
                check_budget()
                if packet_count > settings.max_video_decode_frames:
                    raise APIError("Video exceeded packet inspection budget")
                if packet.is_keyframe and packet.pts is not None:
                    key_times.append(float(packet.pts * stream.time_base) - origin)
            key_times = np.array(sorted(set(key_times)))
            preceding = np.maximum(np.searchsorted(key_times, targets, side="right") - 1, 0)
            seek_work = np.maximum((targets - key_times[preceding]) * fps + 1, 1).sum()
            sequential_work = targets[-1] * fps + 1
            strategy = "sequential" if sequential_work <= seek_work else "seek"
            container.seek(int(origin / stream.time_base), stream=stream, backward=True)

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
                while len(frames) < count and timestamp + 0.5 / fps >= targets[len(frames)]:
                    save(frame, timestamp)
                if len(frames) == count:
                    break
            if not frames:
                raise ValueError("No video frames")
            while len(frames) < count:
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
            backend if stream.codec_context.is_hwaccel else "software",
            1.0 if num_frames is None else None,
        )
