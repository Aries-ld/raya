import gc
import logging
import string
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from .config import Settings
from .errors import APIError
from .media import load_image, read_media, sample_video
from .schemas import ChatRequest, render

logger = logging.getLogger(__name__)


def available_devices(settings: Settings) -> list[str]:
    if settings.device != "auto":
        return [settings.device]
    return (
        (["cuda"] if torch.cuda.is_available() else [])
        + (["mps"] if torch.backends.mps.is_available() else [])
        + ["cpu"]
    )


def dtype_for(device: str, settings: Settings):
    if settings.dtype != "auto":
        return getattr(torch, settings.dtype)
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32 if device == "cpu" else torch.float16


class DecisionEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.device = "unloaded"
        self.devices = available_devices(settings)
        self.fallbacks = 0

    def load(self):
        for path in (self.settings.model_path, self.settings.processor_path):
            if not Path(path).is_dir():
                raise RuntimeError(f"Missing local model assets at {path}; run raya-download first")
        torch.set_num_threads(self.settings.cpu_threads)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.settings.model_path, local_files_only=True
        )
        self.processor = AutoProcessor.from_pretrained(
            self.settings.processor_path, local_files_only=True
        )
        # The fine-tuned tokenizer is authoritative, the base repo supplies vision config.
        self.processor.tokenizer = self.tokenizer
        self.label_ids = []
        for label in string.ascii_uppercase:
            ids = self.tokenizer.encode(f" {label}", add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"Candidate label {label} does not map to one token")
            self.label_ids.append(ids[0])
        if len(set(self.label_ids)) != 26:
            raise RuntimeError("Candidate token IDs collide")
        self._load_device(0)

    def _load_device(self, index: int):
        for device in self.devices[index:]:
            try:
                self.device = device
                self.model = (
                    AutoModelForImageTextToText.from_pretrained(
                        self.settings.model_path,
                        local_files_only=True,
                        trust_remote_code=False,
                        dtype=dtype_for(device, self.settings),
                        attn_implementation="sdpa",
                    )
                    .to(device)
                    .eval()
                )
                logger.info("Raya loaded on %s", device)
                return
            except (RuntimeError, NotImplementedError):
                self._release()
                if device == self.devices[-1]:
                    raise
                self.fallbacks += 1
                logger.warning("Cannot load on %s; trying next device", device)

    def _release(self):
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _forward(self, inputs):
        # No KV cache and no full [sequence, vocab] output: only project the last position.
        with torch.inference_mode():
            output = self.model(**inputs.to(self.device), use_cache=False, logits_to_keep=1)
            logits = output.logits[0, -1, self.label_ids].float().cpu()
        return logits

    def predict(self, request: ChatRequest) -> dict:
        started = time.perf_counter()
        try:
            prompt, candidates, media = render(request)
        except ValueError as exc:
            raise APIError(str(exc)) from exc
        if len(media) > self.settings.max_media or sum(kind == "video" for kind, _ in media) > 1:
            raise APIError("Too many media inputs (at most one video)")
        # Bound text before doing any remote download or video work.
        if len(self.tokenizer.encode(prompt)) > self.settings.max_input_tokens:
            raise APIError("Input exceeds token limit", 400, "context_length_exceeded")
        images, videos, metadata, video_stats = [], [], [], []
        for kind, url in media:
            data = read_media(url, kind, self.settings)
            if kind == "image":
                images.append(load_image(data, self.settings))
            else:
                sample = sample_video(data, self.settings)
                videos.append(sample.frames)
                metadata.append(sample.metadata())
                video_stats.append(
                    {
                        "sampled_frames": 8,
                        "decoded_frames": sample.decoded_frames,
                        "strategy": sample.strategy,
                        "decode_ms": sample.decode_ms,
                    }
                )
        kwargs = {}
        if images:
            kwargs.update(
                images=images,
                images_kwargs={
                    "size": {"shortest_edge": 65536, "longest_edge": self.settings.image_max_pixels}
                },
            )
        if videos:
            kwargs.update(
                videos=videos,
                videos_kwargs={
                    "video_metadata": metadata,
                    "do_sample_frames": False,
                    "fps": None,
                    "num_frames": 8,
                    "cap_pixels_per_frame": True,
                    "size": {
                        "shortest_edge": 65536,
                        "longest_edge": self.settings.video_max_pixels * 8,
                    },
                },
            )
        try:
            inputs = (
                self.processor(text=[prompt], return_tensors="pt", **kwargs)
                if media
                else self.tokenizer(prompt, return_tensors="pt")
            )
        except (ValueError, TypeError) as exc:
            raise APIError("Media cannot be processed with the supplied question") from exc
        token_count = inputs["input_ids"].shape[-1]
        if token_count > self.settings.max_input_tokens:
            raise APIError(
                "Input exceeds token limit after vision expansion", 400, "context_length_exceeded"
            )
        prepared = time.perf_counter()
        while True:
            try:
                logits = self._forward(inputs)
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Non-finite device output")
                break
            except (RuntimeError, NotImplementedError) as exc:
                index = self.devices.index(self.device)
                # Only backend/OOM/operator failures should degrade, not arbitrary input errors.
                message = str(exc).lower()
                backend_error = any(
                    word in message
                    for word in (
                        "out of memory",
                        "mps",
                        "cuda",
                        "not implemented",
                        "not supported",
                        "non-finite device output",
                    )
                ) or isinstance(exc, NotImplementedError)
                if not backend_error or index == len(self.devices) - 1:
                    raise
                inputs = inputs.to("cpu")
                # Drop failed operator frames holding device tensors before allocating a new model.
                exc.__traceback__ = None
                self._release()
                self.fallbacks += 1
                logger.warning("Inference backend failed; falling back from %s", self.device)
                self._load_device(index + 1)
        probabilities = torch.softmax(logits[: len(candidates)], dim=-1).tolist()
        winner = max(range(len(probabilities)), key=probabilities.__getitem__)
        labels = string.ascii_uppercase[: len(candidates)]
        return {
            "label": labels[winner],
            "index": winner,
            "selected": candidates[winner],
            "confidence": probabilities[winner],
            "probabilities": dict(zip(labels, probabilities)),
            "candidates": dict(zip(labels, candidates)),
            "needs_review": probabilities[winner] < request.confidence_threshold,
            "threshold": request.confidence_threshold,
            "prompt_tokens": token_count,
            "device": self.device,
            "timing_ms": {
                "preprocess": round((prepared - started) * 1000, 3),
                "forward": round((time.perf_counter() - prepared) * 1000, 3),
                "total": round((time.perf_counter() - started) * 1000, 3),
            },
            "video": video_stats,
        }
