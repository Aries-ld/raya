import json

import pytest

from raya_maas.decisions import state_parts
from raya_maas.local_test import LocalDecisionSession, load_request_file


class FakeEngine:
    def __init__(self):
        self.loads = 0
        self.releases = 0
        self.seen = []
        self.last_diagnostics = {"total_ms": 1}

    def load(self):
        self.loads += 1

    def predict(self, request):
        self.seen.append(request)
        return {
            "model": request.model,
            "answers": {},
            "usage": {"input_tokens": 1, "output_tokens": 0},
        }

    def _release(self):
        self.releases += 1


def write_request(path, state, media=None):
    payload = {
        "model": "raya-decision-v1",
        "state": state if media is None else {"text": state, "media": media},
        "questions": {
            "check": {
                "type": "choice",
                "instructions": "Which?",
                "criteria": {"yes": "Yes", "no": "No"},
            }
        },
    }
    path.write_text(json.dumps(payload))


def test_session_reuses_model_but_rereads_edited_json_and_media(tmp_path):
    path, media = tmp_path / "image.json", tmp_path / "image.jpg"
    media.write_bytes(b"first media contents")
    write_request(path, "first question context", [{"type": "image", "url": "file:./image.jpg"}])
    engine = FakeEngine()
    with LocalDecisionSession(tmp_path, engine) as session:
        session.predict(load_request_file(path, session.settings))
        media.write_bytes(b"second media contents")
        write_request(path, "updated context", [{"type": "image", "url": "file:./image.jpg"}])
        session.predict(load_request_file(path, session.settings))
        assert engine.loads == 1 and engine.releases == 0
        first_text, first_media = state_parts(engine.seen[0].state)
        second_text, second_media = state_parts(engine.seen[1].state)
        assert first_text != second_text
        assert first_media[0].url != second_media[0].url
        diagnostics = json.loads(
            (tmp_path / "artifacts/local-decision-diagnostics.json").read_text()
        )
        assert diagnostics["model_reused"] is True
        assert diagnostics["model_load_ms"] == 0
        assert diagnostics["session_request_number"] == 2
    assert engine.releases == 1
    session.close()
    assert engine.releases == 1
    with pytest.raises(RuntimeError, match="closed"):
        session.load()


def test_invalid_file_does_not_discard_loaded_model(tmp_path):
    path = tmp_path / "text.json"
    engine = FakeEngine()
    with LocalDecisionSession(tmp_path, engine) as session:
        path.write_text("not JSON")
        with pytest.raises(ValueError):
            load_request_file(path, session.settings)
        write_request(path, "corrected")
        session.predict(load_request_file(path, session.settings))
        assert engine.loads == 1
    assert engine.releases == 1


def test_failed_load_releases_session_resources(tmp_path):
    class BrokenEngine(FakeEngine):
        def load(self):
            super().load()
            raise RuntimeError("load failed")

    engine = BrokenEngine()
    with pytest.raises(RuntimeError, match="load failed"):
        with LocalDecisionSession(tmp_path, engine):
            pass
    assert engine.loads == 1
    assert engine.releases == 1
