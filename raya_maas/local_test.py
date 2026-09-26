"""Run the same System One protocol locally; only file: media references are expanded."""

import base64
import json
import mimetypes
import time
from pathlib import Path

from .config import Settings
from .decisions import SystemOneRequest
from .schemas import Media


def load_request_file(path: Path, settings: Settings | None = None) -> SystemOneRequest:
    settings = settings or Settings(_env_file=path.resolve().parent.parent / ".env")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    # The HTTP protocol accepts data/HTTPS URLs. file: is an offline-test convenience only.
    state = payload.get("state") if isinstance(payload, dict) else None
    media = state.get("media", []) if isinstance(state, dict) else []
    if isinstance(media, list):
        for item in media:
            if not isinstance(item, dict) or not isinstance(item.get("url"), str):
                continue
            url = item["url"]
            if not url.startswith("file:"):
                continue
            kind = item.get("type")
            if kind not in ("image", "video"):
                continue
            source = Path(url[5:]).expanduser()
            source = source if source.is_absolute() else path.resolve().parent / source
            if not source.is_file():
                raise ValueError(f"Media file not found: {source}")
            if source.stat().st_size > settings.max_media_bytes:
                raise ValueError("Media file exceeds byte limit")
            with source.open("rb") as stream:
                data = stream.read(settings.max_media_bytes + 1)
            if not data or len(data) > settings.max_media_bytes:
                raise ValueError("Empty or oversized media file")
            mime = mimetypes.guess_type(source.name)[0] or f"{kind}/octet-stream"
            item["url"] = f"data:{mime};base64," + base64.b64encode(data).decode()
            Media.model_validate(item)
    return SystemOneRequest.model_validate(payload)


def run_local_decision(request: SystemOneRequest, project_root: Path) -> dict:
    from .engine import DecisionEngine

    settings = Settings(_env_file=project_root / ".env")
    for name in ("model_path", "processor_path"):
        path = Path(getattr(settings, name)).expanduser()
        setattr(settings, name, str(path if path.is_absolute() else project_root / path))
    engine = DecisionEngine(settings)
    try:
        started = time.perf_counter()
        engine.load()  # IDE breakpoint: model initialization.
        model_load_ms = (time.perf_counter() - started) * 1000
        result = engine.predict(request)  # IDE breakpoint: the same request used by /v1/systemone.
        diagnostics = {"model_load_ms": round(model_load_ms, 3), **engine.last_diagnostics}
        # Diagnostics are local debug data, not extra fields in the public protocol.
        output = project_root / "artifacts" / "local-decision-diagnostics.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n")
    finally:
        engine._release()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
