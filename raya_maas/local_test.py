"""Shared CLI for local model tests: no HTTP server, API key or network required."""

import argparse
import base64
import json
import mimetypes
import time
from pathlib import Path

from pydantic import ValidationError

from .config import Settings
from .errors import APIError
from .schemas import ChatRequest


def build_request(args, settings: Settings, modality: str) -> ChatRequest:
    content = [{"type": "text", "text": args.question}]
    if modality != "text":
        path = getattr(args, modality).expanduser()
        if not path.is_file():
            raise ValueError(f"找不到{modality}文件：{path}")
        if path.stat().st_size > settings.max_media_bytes:
            raise ValueError(f"媒体文件超过 {settings.max_media_bytes / 1024**2:g} MiB 限制")
        with path.open("rb") as source:
            data = source.read(settings.max_media_bytes + 1)
        if not data or len(data) > settings.max_media_bytes:
            raise ValueError("媒体文件为空或超过大小限制")
        mime = mimetypes.guess_type(path.name)[0]
        if not mime or not mime.startswith(f"{modality}/"):
            mime = f"{modality}/octet-stream"  # Actual format is validated by Pillow/PyAV.
        url = f"data:{mime};base64," + base64.b64encode(data).decode()
        content.insert(0, {"type": f"{modality}_url", f"{modality}_url": {"url": url}})
    return ChatRequest(
        model=settings.model_name,
        messages=[{"role": "user", "content": content}],
        candidates=args.candidates,
        confidence_threshold=args.threshold,
    )


def run_local_test(
    modality: str, question: str, candidates: list[str], media_path: Path | None = None
):
    parser = argparse.ArgumentParser(description=f"Raya 本地{modality}决策测试，无需启动服务")
    parser.add_argument("--question", default=question, help="问题或文本上下文")
    parser.add_argument("--candidates", nargs="+", default=candidates, help="2–26 个候选，逐个传入")
    if modality != "text":
        parser.add_argument(f"--{modality}", type=Path, default=media_path, help="本地媒体路径")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=None)
    parser.add_argument("--threshold", type=float, default=0.6, help="低置信度门控阈值")
    parser.add_argument("--max-input-tokens", type=int, default=None, help="覆盖输入 token 上限")
    if modality == "video":
        parser.add_argument(
            "--max-video-seconds", type=float, default=None, help="覆盖视频时长上限"
        )
    parser.add_argument("--output", type=Path, help="可选：将 JSON 结果写入文件")
    args = parser.parse_args()
    overrides = {
        name: getattr(args, name, None)
        for name in ("device", "max_input_tokens", "max_video_seconds")
        if getattr(args, name, None) is not None
    }
    try:
        settings = Settings(**overrides)
        request = build_request(args, settings, modality)
    except (OSError, ValueError, ValidationError) as exc:
        parser.exit(2, f"输入错误：{exc}\n")

    # Import/load only after validating CLI input. --help does not initialize torch.
    from .engine import DecisionEngine

    engine = DecisionEngine(settings)
    try:
        started = time.perf_counter()
        engine.load()
        load_ms = round((time.perf_counter() - started) * 1000, 3)
        result = {"modality": modality, "model_load_ms": load_ms, **engine.predict(request)}
    except APIError as exc:
        parser.exit(2, f"输入错误：{exc.message}\n")
    except (OSError, RuntimeError) as exc:
        parser.exit(1, f"模型推理失败：{exc}\n请确认已运行 uv run raya-download。\n")
    finally:
        engine._release()
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(output, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
