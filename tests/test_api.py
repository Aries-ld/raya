import json

import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from raya_maas.app import create_app
from raya_maas.config import Settings
from raya_maas.schemas import ChatRequest, render


class FakeEngine:
    def load(self):
        pass

    def predict(self, request):
        return {
            "label": "A",
            "selected": "yes",
            "index": 0,
            "confidence": 0.75,
            "probabilities": {"A": 0.75, "B": 0.25},
            "needs_review": False,
            "prompt_tokens": 12,
            "device": "test",
        }


@pytest.fixture
def client():
    with TestClient(
        create_app(Settings(api_keys="test-secret", _env_file=None), FakeEngine())
    ) as c:
        yield c


def payload(**kwargs):
    return {
        "model": "raya-decision-v1",
        "messages": [{"role": "user", "content": "Proceed?"}],
        "candidates": ["yes", "no"],
        **kwargs,
    }


def sdk_payload(**kwargs):
    data = payload(**kwargs)
    data["extra_body"] = {"candidates": data.pop("candidates")}
    return data


def test_openai_sdk(client):
    sdk = OpenAI(base_url="http://testserver/v1", api_key="test-secret", http_client=client)
    response = sdk.chat.completions.create(**sdk_payload())
    assert response.choices[0].message.content == "A"
    assert response.decision["probabilities"] == {"A": 0.75, "B": 0.25}
    assert response.usage.completion_tokens == 0
    assert sdk.models.list().data[0].id == "raya-decision-v1"
    assert sdk.models.retrieve("raya-decision-v1").owned_by == "raya"


def test_stream_and_json_sdk(client):
    sdk = OpenAI(base_url="http://testserver/v1", api_key="test-secret", http_client=client)
    chunks = list(
        sdk.chat.completions.create(
            **sdk_payload(
                stream=True,
                stream_options={"include_usage": True},
                response_format={"type": "json_object"},
            )
        )
    )
    content = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert json.loads(content)["label"] == "A"
    assert chunks[-2].choices[0].finish_reason == "stop"
    assert chunks[-1].usage.prompt_tokens == 12
    assert chunks[-1].choices == []


def test_auth_and_health(client):
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 200
    response = client.post("/v1/chat/completions", content=b"not json")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "authentication_error"


@pytest.mark.parametrize(
    "change",
    [
        {"candidates": ["only one"]},
        {"candidates": ["same", "same"]},
        {"temperature": 0.7},
        {"n": 2},
        {"tools": []},
        {"candidates": None},
        {"confidence_threshold": -0.1},
        {"candidates": ["a\nb", "c"]},
        {"messages": [{"role": "user", "content": "<|image_pad|>"}]},
        {"messages": [{"role": "assistant", "content": "hello"}]},
    ],
)
def test_invalid_requests(client, change):
    response = client.post(
        "/v1/chat/completions",
        json=payload(**change),
        headers={"Authorization": "Bearer test-secret"},
    )
    assert response.status_code == 400
    assert "error" in response.json()


def test_model_not_found(client):
    response = client.post(
        "/v1/chat/completions",
        json=payload(model="unknown"),
        headers={"Authorization": "Bearer test-secret"},
    )
    assert response.status_code == 404


def test_body_limit_chunked():
    settings = Settings(allow_anonymous=True, max_body_bytes=20, _env_file=None)
    with TestClient(create_app(settings, FakeEngine())) as client:
        response = client.post("/v1/chat/completions", content=iter([b"a" * 11, b"b" * 11]))
        assert response.status_code == 413


def test_inline_options_match_structured_training_prompt():
    structured = ChatRequest(**payload())
    inline = ChatRequest(
        **payload(
            candidates=None, messages=[{"role": "user", "content": "Proceed?\n\nA. yes\nB. no"}]
        )
    )
    assert render(structured) == render(inline)


def test_no_silent_anonymous_startup():
    with pytest.raises(RuntimeError, match="RAYA_API_KEYS"):
        with TestClient(create_app(Settings(_env_file=None), FakeEngine())):
            pass
