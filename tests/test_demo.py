import json

import httpx
from fastapi.testclient import TestClient

from raya_maas.config import Settings
from raya_maas.demo_server import CRITERIA, INSTRUCTIONS, DemoSettings, create_demo_app


def request_body():
    return {
        "model": "raya-decision-v1",
        "state": {
            "text": "Camera crop",
            "media": [{"type": "image", "url": "data:image/jpeg;base64,YQ=="}],
        },
        "questions": {
            "gesture": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": CRITERIA}
        },
    }


def test_demo_keeps_credentials_on_server_and_forwards_response_timings():
    calls = []

    def upstream(request):
        assert request.headers["authorization"] == "Bearer server-only-secret"
        calls.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"models": [{"id": "raya-decision-v1"}]})
        body = json.loads(request.content)
        assert body == request_body()
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "answers": {
                    "gesture": {
                        "type": "choice",
                        "choice": "fist",
                        "confidence": 1,
                        "probabilities": {key: int(key == "fist") for key in CRITERIA},
                    }
                },
                "usage": {"input_tokens": 200, "output_tokens": 0},
            },
            headers={
                "X-Raya-Forward-Ms": "220.5",
                "X-Raya-Processing-Ms": "240.7",
                "X-Raya-Queue-Ms": "0.2",
                "X-Request-Id": "req_demo",
            },
        )

    app = create_demo_app(
        Settings(api_keys="server-only-secret", _env_file=None),
        DemoSettings(_env_file=None),
        httpx.MockTransport(upstream),
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "server-only-secret" not in page.text
        config = client.get("/api/config")
        assert "server-only-secret" not in config.text
        assert config.json()["sample_interval_ms"] == 1000
        assert len(config.json()["criteria"]) == 7
        assert client.get("/api/health").json()["ready"]
        response = client.post(
            "/api/systemone",
            json=request_body(),
            headers={"X-Raya-Demo": "gesture-lab", "Origin": "http://127.0.0.1"},
        )
        assert response.status_code == 200
        assert set(response.json()) == {"model", "answers", "usage"}
        assert response.headers["x-raya-forward-ms"] == "220.5"
        assert response.headers["x-raya-processing-ms"] == "240.7"
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["permissions-policy"] == "camera=(self), microphone=()"
    assert len(calls) == 2


def test_demo_rejects_cross_origin_rebinding_and_large_uploads_without_proxying():
    def forbidden(request):
        raise AssertionError("Must not reach upstream")

    app = create_demo_app(
        Settings(_env_file=None), DemoSettings(_env_file=None), httpx.MockTransport(forbidden)
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/", headers={"Host": "evil.example"}).status_code == 403
        assert client.post("/api/systemone", json=request_body()).status_code == 403
        assert (
            client.post(
                "/api/systemone",
                json=request_body(),
                headers={"X-Raya-Demo": "gesture-lab", "Origin": "https://evil.example"},
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/systemone",
                content=b"x" * (2 * 1024 * 1024 + 1),
                headers={"X-Raya-Demo": "gesture-lab"},
            ).status_code
            == 413
        )


def test_demo_handles_unavailable_backend_without_exposing_key():
    def unavailable(request):
        raise httpx.ConnectError("Cannot connect", request=request)

    app = create_demo_app(
        Settings(api_keys="private-key", _env_file=None),
        DemoSettings(_env_file=None),
        httpx.MockTransport(unavailable),
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/health").status_code == 503
        result = client.post(
            "/api/systemone", json=request_body(), headers={"X-Raya-Demo": "gesture-lab"}
        )
        assert result.status_code == 503
        assert "private-key" not in result.text
