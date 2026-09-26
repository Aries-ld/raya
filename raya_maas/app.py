import asyncio
import hmac
import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException

from .config import Settings
from .errors import APIError
from .schemas import ChatRequest, render
from .worker import InferenceWorker


class RequestGuard:
    """Authenticate before parsing bodies, and cap actual bytes including chunked uploads."""

    def __init__(self, app, settings):
        self.app, self.settings = app, settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        if scope["path"] not in ("/healthz", "/readyz"):
            authorization = headers.get(b"authorization", b"").decode("latin1")
            supplied = authorization[7:] if authorization.startswith("Bearer ") else ""
            valid = False
            for key in self.settings.keys:
                valid |= hmac.compare_digest(supplied.encode(), key.encode())
            if not valid and not self.settings.allow_anonymous:
                error = APIError("Invalid or missing API key", 401, "authentication_error")
                return await JSONResponse(error.body(), 401)(scope, receive, send)
        request_id = "req_" + uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        body, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > self.settings.max_body_bytes:
                error = APIError("Request body exceeds byte limit", 413, "payload_too_large")
                return await JSONResponse(error.body(), 413)(scope, receive, send)
            body.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        delivered = False

        async def buffered_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": b"".join(body), "more_body": False}
            return await receive()

        async def response_send(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).append((b"x-request-id", request_id.encode()))
            await send(message)

        await self.app(scope, buffered_receive, response_send)


def create_app(settings: Settings | None = None, engine=None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        if not settings.keys and not settings.allow_anonymous:
            raise RuntimeError("Set RAYA_API_KEYS, or explicitly RAYA_ALLOW_ANONYMOUS=true locally")
        if engine is None:
            from .engine import DecisionEngine

            active_engine = DecisionEngine(settings)
        else:
            active_engine = engine
        await asyncio.to_thread(active_engine.load)
        app.state.engine = active_engine
        app.state.worker = InferenceWorker(
            active_engine, settings.queue_size, settings.request_timeout
        )
        await app.state.worker.start()
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            await app.state.worker.close()

    app = FastAPI(title="Raya MaaS", version="0.1.0", lifespan=lifespan)
    app.state.ready = False
    app.add_middleware(RequestGuard, settings=settings)

    @app.exception_handler(APIError)
    async def api_error(request, exc):
        headers = {"Retry-After": "1"} if exc.status == 429 else None
        return JSONResponse(exc.body(), exc.status, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Validation's input field can contain entire base64 payloads; never reflect it.
        problems = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
        return JSONResponse(APIError("; ".join(problems)).body(), 400)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return JSONResponse(APIError(str(exc.detail), exc.status_code).body(), exc.status_code)

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready():
        return JSONResponse(
            {"status": "ready" if app.state.ready else "loading"}, 200 if app.state.ready else 503
        )

    def model_card():
        return {"id": settings.model_name, "object": "model", "created": 0, "owned_by": "raya"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [model_card()]}

    @app.get("/v1/models/{model_id}")
    async def model(model_id: str):
        if model_id != settings.model_name:
            raise APIError("Model not found", 404, "model_not_found")
        return model_card()

    @app.post("/v1/chat/completions")
    async def completions(body: ChatRequest, request: Request):
        if body.model != settings.model_name:
            raise APIError("Model not found", 404, "model_not_found")
        try:
            render(body)
        except ValueError as exc:
            raise APIError(str(exc)) from exc
        if not app.state.ready:
            raise APIError("Model not ready", 503, "service_unavailable")
        result = await app.state.worker.submit(body)
        prompt_tokens = result.pop("prompt_tokens")
        content = (
            json.dumps(result, ensure_ascii=False)
            if body.response_format.type == "json_object"
            else result["label"]
        )
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": prompt_tokens,
        }
        common = {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "created": int(time.time()),
            "model": settings.model_name,
        }
        if not body.stream:
            return {
                **common,
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content, "refusal": None},
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
                "usage": usage,
                "decision": result,
            }

        async def chunks():
            common["object"] = "chat.completion.chunk"
            for delta, finish, extra in [
                ({"role": "assistant", "content": ""}, None, {}),
                ({"content": content}, None, {}),
                ({}, "stop", {"decision": result}),
            ]:
                chunk = {
                    **common,
                    "choices": [
                        {"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}
                    ],
                    **extra,
                }
                if body.stream_options and body.stream_options.include_usage:
                    chunk["usage"] = None
                yield "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"
            if body.stream_options and body.stream_options.include_usage:
                yield "data: " + json.dumps({**common, "choices": [], "usage": usage}) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            chunks(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
