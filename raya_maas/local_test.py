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


class LocalDecisionSession:
    """Own one model for multiple decisions in the same Run/Debug process."""

    def __init__(self, project_root: Path, engine=None):
        self.project_root = project_root.resolve()
        self.settings = Settings(_env_file=self.project_root / ".env")
        for name in ("model_path", "processor_path"):
            path = Path(getattr(self.settings, name)).expanduser()
            setattr(
                self.settings, name, str(path if path.is_absolute() else self.project_root / path)
            )
        self.engine = engine
        self.loaded = False
        self.closed = False
        self.model_load_ms = 0.0
        self.predictions = 0

    def load(self):
        if self.closed:
            raise RuntimeError("This local decision session is closed")
        if self.loaded:
            return
        if self.engine is None:
            from .engine import DecisionEngine

            self.engine = DecisionEngine(self.settings)
        started = time.perf_counter()
        try:
            self.engine.load()  # IDE breakpoint: runs once for this session.
        except BaseException:
            self.close()
            raise
        self.model_load_ms = round((time.perf_counter() - started) * 1000, 3)
        self.loaded = True

    def predict(self, request: SystemOneRequest) -> dict:
        self.load()
        result = self.engine.predict(request)  # IDE breakpoint: runs for every decision.
        diagnostics = {
            "model_load_ms": self.model_load_ms if self.predictions == 0 else 0.0,
            "session_model_load_ms": self.model_load_ms,
            "model_reused": self.predictions > 0,
            "session_request_number": self.predictions + 1,
            **self.engine.last_diagnostics,
        }
        self.predictions += 1
        # Debug information stays outside the public decision response.
        output = self.project_root / "artifacts" / "local-decision-diagnostics.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    def close(self):
        if not self.closed:
            self.closed = True
            self.loaded = False
            if self.engine is not None:
                self.engine._release()

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def run_local_decision(request: SystemOneRequest, project_root: Path) -> dict:
    """One-shot entry used by the individual scripts; session runner reuses the class directly."""
    with LocalDecisionSession(project_root) as session:
        return session.predict(request)
