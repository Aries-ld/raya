"""Loopback-only gesture demo. Keeps the MaaS key on the server, outside browser code."""

import socket
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import Settings
from .decisions import SystemOneRequest

CRITERIA = {
    "fist": "A closed fist: zero extended fingers; all fingers folded into the palm.",
    "one": (
        "Exactly ONE finger extended; the other four fingers folded. The thumb counts if extended."
    ),
    "two": "Exactly TWO fingers extended, such as a V sign; the other three fingers folded.",
    "three": "Exactly THREE fingers extended; two fingers folded. Count the thumb if extended.",
    "four": "Exactly FOUR fingers extended; one finger folded, usually the thumb.",
    "five": "An open palm with all FIVE fingers extended, including the thumb.",
    "none": (
        "No clear single hand: no hand visible, multiple hands, severe occlusion, "
        "or no identifiable gesture."
    ),
}
INSTRUCTIONS = (
    "Examine the single hand in this camera image and count its fully extended fingers. "
    "The thumb counts as one finger when extended. "
    "Choose fist for zero, or one/two/three/four/five. "
    "Ignore the face and background. If no complete single hand is clearly visible, choose none."
)


class DemoSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAYA_DEMO_", env_file=".env", extra="ignore")
    port: int = Field(8765, ge=1, le=65535)
    maas_url: str = "http://127.0.0.1:8000"


def create_demo_app(settings=None, demo_settings=None, transport=None):
    settings = settings or Settings()
    demo_settings = demo_settings or DemoSettings()
    address = urlsplit(demo_settings.maas_url)
    if address.scheme not in ("http", "https") or address.username or address.password:
        raise ValueError("RAYA_DEMO_MAAS_URL must be an HTTP(S) address without credentials")
    headers = {"Authorization": f"Bearer {settings.keys[0]}"} if settings.keys else {}

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(
            base_url=demo_settings.maas_url.rstrip("/"),
            headers=headers,
            timeout=30,
            trust_env=False,
            transport=transport,
        ) as client:
            app.state.client = client
            yield

    app = FastAPI(title="Raya Gesture Lab", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]"])
    assets = Path(__file__).parent / "demo_web"

    @app.middleware("http")
    async def local_guard(request: Request, call_next):
        # Protect the credential-bearing proxy from cross-site requests and DNS rebinding.
        host = request.url.hostname
        if host not in ("localhost", "127.0.0.1", "::1"):
            return JSONResponse({"error": {"message": "Local demo only"}}, 403)
        origin = request.headers.get("origin")
        if origin:
            parsed = urlsplit(origin)
            if parsed.scheme != request.url.scheme or parsed.netloc != request.url.netloc:
                return JSONResponse(
                    {"error": {"message": "Cross-origin requests are not allowed"}}, 403
                )
        if request.method == "POST":
            if request.headers.get("x-raya-demo") != "gesture-lab":
                return JSONResponse({"error": {"message": "Missing demo request header"}}, 403)
            # Camera crops are small; cap actual body bytes, not just Content-Length.
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 2 * 1024 * 1024:
                    return JSONResponse({"error": {"message": "Camera frame is too large"}}, 413)
            request._body = bytes(body)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Permissions-Policy"] = "camera=(self), microphone=()"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; "
            "object-src 'none'; frame-ancestors 'none'; base-uri 'self'"
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse({"error": {"message": "Invalid decision request"}}, 422)

    @app.get("/")
    async def index():
        return FileResponse(assets / "index.html")

    @app.get("/api/config")
    async def config():
        return {
            "model": settings.model_name,
            "sample_interval_ms": 1000,
            "capture_size": 384,
            "stale_ms": 2500,
            "confidence_threshold": 0.55,
            "instructions": INSTRUCTIONS,
            "criteria": CRITERIA,
            "endpoint": "/v1/systemone",
        }

    @app.get("/api/health")
    async def health():
        try:
            response = await app.state.client.get("/v1/models")
        except httpx.HTTPError:
            return JSONResponse(
                {"ready": False, "message": "无法连接 MaaS，请先运行 uv run raya-serve。"}, 503
            )
        if response.status_code != 200:
            return JSONResponse(
                {"ready": False, "message": "MaaS 未就绪或 API key 不匹配，请检查 .env。"}, 503
            )
        try:
            model_ids = [model["id"] for model in response.json().get("models", [])]
        except (ValueError, KeyError, TypeError):
            model_ids = []
        if settings.model_name not in model_ids:
            return JSONResponse({"ready": False, "message": "MaaS 未提供当前配置的模型。"}, 503)
        return {"ready": True, "model": settings.model_name}

    @app.post("/api/systemone")
    async def decide(body: SystemOneRequest):
        started = time.perf_counter()
        try:
            response = await app.state.client.post("/v1/systemone", json=body.model_dump())
        except httpx.TimeoutException:
            return JSONResponse({"error": {"message": "模型响应超时，请重试。"}}, 504)
        except httpx.HTTPError:
            return JSONResponse({"error": {"message": "无法连接 MaaS 服务。"}}, 503)
        try:
            payload = response.json()
        except ValueError:
            return JSONResponse({"error": {"message": "MaaS 返回了无效响应。"}}, 502)
        timings = {
            name: value
            for name, value in response.headers.items()
            if name.lower().startswith("x-raya-") or name.lower() == "x-request-id"
        }
        timings["X-Demo-Upstream-Ms"] = f"{(time.perf_counter() - started) * 1000:.3f}"
        return JSONResponse(payload, response.status_code, headers=timings)

    app.mount("/assets", StaticFiles(directory=assets), name="assets")
    return app


def serve():
    config = DemoSettings()
    # Deliberately bind only loopback. Public demos need an authenticated deployment design.
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", config.port))
        except OSError as exc:
            raise RuntimeError(f"Demo port {config.port} is in use; set RAYA_DEMO_PORT") from exc
    print(f"Gesture demo: http://127.0.0.1:{config.port}")
    uvicorn.run(create_demo_app(demo_settings=config), host="127.0.0.1", port=config.port)


if __name__ == "__main__":
    serve()
