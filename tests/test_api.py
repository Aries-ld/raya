import pytest
from fastapi.testclient import TestClient

from raya_maas.app import create_app
from raya_maas.config import Settings
from raya_maas.decisions import SystemOneRequest, evaluate


class FakeEngine:
    def __init__(self):
        self.settings = Settings(_env_file=None)
        self.tokenizer = type("Tokenizer", (), {"encode": lambda self, text: [1, 2, 3]})()
        self.preparations = 0
        self.inputs = []

    def load(self):
        pass

    def prepare_media(self, media):
        self.preparations += 1
        return {}, []

    def predict(self, request, *, prepared_media=None):
        if isinstance(request, SystemOneRequest):
            return evaluate(self, request)
        self.inputs.append(request)
        probabilities = [0.25, 0.75] if len(request.candidates) == 2 else [0.1, 0.2, 0.7]
        return {
            "index": len(probabilities) - 1,
            "probabilities": {chr(65 + i): p for i, p in enumerate(probabilities)},
            "prompt_tokens": 12,
            "device": "test",
            "timing_ms": {"total": 1},
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
        "state": "The customer cannot log in.",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Where should this go?",
                "criteria": {"billing": "Payments", "account": "Login"},
            }
        },
        **kwargs,
    }


def post(client, data):
    return client.post("/v1/systemone", json=data, headers={"Authorization": "Bearer test-secret"})


def test_systemone_mixed_questions_and_exact_response_shape(client):
    data = payload()
    data["questions"].update(
        {
            "urgent": {"type": "noul", "instructions": "Is this urgent?"},
            "severity": {
                "type": "score",
                "instructions": "How severe?",
                "criteria": ["low", "medium", "high"],
            },
        }
    )
    response = post(client, data)
    assert response.status_code == 200
    result = response.json()
    assert set(result) == {"model", "answers", "usage"}
    assert list(result["answers"]) == ["route", "urgent", "severity"]
    assert result["answers"]["route"] == {
        "type": "choice",
        "choice": "account",
        "confidence": 0.75,
        "probabilities": {"billing": 0.25, "account": 0.75},
    }
    assert result["answers"]["urgent"] == {"type": "noul", "noul": 0.25}
    score = result["answers"]["severity"]
    assert score["score"] == pytest.approx(1.6)
    assert score["legend"] == {"0": "low", "1": "medium", "2": "high"}
    assert result["usage"] == {"input_tokens": 36, "output_tokens": 0}
    assert response.headers["x-request-id"].startswith("req_")
    assert client.app.state.engine.preparations == 1


def test_auth_health_and_no_chat_endpoint(client):
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 200
    assert client.post("/v1/systemone", content=b"not json").status_code == 401
    headers = {"Authorization": "Bearer test-secret"}
    assert client.post("/v1/chat/completions", json={}, headers=headers).status_code == 404
    models = client.get("/v1/models", headers=headers).json()["models"]
    assert models[0]["question_types"] == ["choice", "score", "noul"]


@pytest.mark.parametrize(
    "change",
    [
        {"questions": {}},
        {"questions": {"x": {"type": "choice", "criteria": {"a": "yes", "b": "no"}}}},
        {"questions": {"x": {"type": "choice", "instructions": "Q?"}}},
        {"questions": {"x": {"type": "choice", "instructions": "Q?", "criteria": {"a": "yes"}}}},
        {
            "questions": {
                "x": {"type": "choice", "instructions": "", "criteria": {"a": "y", "b": "n"}}
            }
        },
        {"questions": {"x": {"type": "score", "instructions": "Q?", "criteria": ["low"]}}},
        {"questions": {"x": {"type": "score", "instructions": "Q?", "criteria": ["x"] * 11}}},
        {"questions": {"x": {"type": "noul", "instructions": "Q?", "criteria": {"yes": "y"}}}},
        {"questions": {"x": {"type": "other", "instructions": "Q?"}}},
        {"state": None},
        {"state": ""},
        {"state": "<|image_pad|>"},
        {"messages": []},
        {"state": {"media": [{"type": "image", "url": "file:./image.jpg"}]}},
        {"state": {"media": [{"type": "image", "url": "data:video/mp4;base64,YQ=="}]}},
    ],
)
def test_invalid_requests(client, change):
    response = post(client, payload(**change))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request_error"


def test_missing_state_and_model(client):
    for missing in ("state", "model", "questions"):
        data = payload()
        del data[missing]
        assert post(client, data).status_code == 422


def test_options_and_question_limits(client):
    data = payload()
    data["questions"]["route"]["criteria"] = {f"key{i}": str(i) for i in range(27)}
    assert post(client, data).status_code == 422
    data = payload()
    q = data["questions"]["route"]
    data["questions"] = {f"question{i}": q for i in range(17)}
    assert post(client, data).status_code == 422


def test_multimodal_state_does_not_replace_the_question(client):
    data = payload(
        state={
            "text": "Evaluate the picture.",
            "media": [{"type": "image", "url": "data:image/png;base64,YQ=="}],
        }
    )
    assert post(client, data).status_code == 200
    value = client.app.state.engine.inputs[-1]
    assert "Evaluate the picture." in value.text and "Where should this go?" in value.text
    assert len(value.media) == 1
    assert value.candidates == ["billing: Payments", "account: Login"]
    del data["questions"]
    assert post(client, data).status_code == 422


def test_question_ids_do_not_change_the_model_input(client):
    first = payload()
    second = payload(questions={"arbitrary_new_id": first["questions"]["route"]})
    post(client, first)
    response = post(client, second)
    assert "arbitrary_new_id" in response.json()["answers"]
    assert client.app.state.engine.inputs[-1] == client.app.state.engine.inputs[-2]


def test_structured_state_instructions_and_criteria(client):
    data = payload(state=[{"speaker": "user", "message": "refund"}])
    question = data["questions"]["route"]
    question["instructions"] = {"question": "Choose a route", "policy": ["Be precise"]}
    question["criteria"] = {"billing": {"meaning": "payment problems"}, "other": None}
    response = post(client, data)
    assert response.status_code == 200
    assert response.json()["answers"]["route"]["choice"] == "other"


def test_unknown_model(client):
    assert post(client, payload(model="unknown")).status_code == 404


def test_body_limit_chunked():
    settings = Settings(allow_anonymous=True, max_body_bytes=20, _env_file=None)
    with TestClient(create_app(settings, FakeEngine())) as client:
        response = client.post("/v1/systemone", content=iter([b"a" * 11, b"b" * 11]))
        assert response.status_code == 413


def test_no_silent_anonymous_startup():
    with pytest.raises(RuntimeError, match="RAYA_API_KEYS"):
        with TestClient(create_app(Settings(_env_file=None), FakeEngine())):
            pass


def test_openapi_exposes_typed_decision_contract(client):
    spec = client.app.openapi()
    assert "/v1/systemone" in spec["paths"]
    assert "/v1/chat/completions" not in spec["paths"]
    schemas = spec["components"]["schemas"]
    assert set(schemas["SystemOneResponse"]["properties"]) == {"model", "answers", "usage"}
    assert set(schemas["SystemOneRequest"]["required"]) == {"model", "state", "questions"}
