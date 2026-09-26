from types import SimpleNamespace

import torch
from transformers import BatchEncoding

from raya_maas.config import Settings
from raya_maas.engine import DecisionEngine, available_devices
from raya_maas.schemas import ChatRequest


def test_device_priority(monkeypatch):
    settings = Settings(_env_file=None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert available_devices(settings) == ["cuda", "mps", "cpu"]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert available_devices(settings) == ["mps", "cpu"]
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert available_devices(settings) == ["cpu"]


class Tokenizer:
    def encode(self, prompt):
        return [1, 2, 3]

    def __call__(self, prompt, **kwargs):
        return BatchEncoding({"input_ids": torch.tensor([[1, 2, 3]])})


def request():
    return ChatRequest(
        model="raya-decision-v1",
        messages=[{"role": "user", "content": "Q?"}],
        candidates=["yes", "no"],
        confidence_threshold=0.8,
    )


def test_one_forward_last_position_and_candidate_mask():
    class Model:
        calls = 0

        def __call__(self, **kwargs):
            self.calls += 1
            assert kwargs["logits_to_keep"] == 1
            assert kwargs["use_cache"] is False
            assert not torch.is_grad_enabled()
            # Huge logits outside the two valid options must have no effect on probability.
            logits = torch.full((1, 1, 30), 1000.0)
            logits[0, 0, 0], logits[0, 0, 1] = 1.0, 2.0
            return SimpleNamespace(logits=logits)

    engine = DecisionEngine(Settings(device="cpu", _env_file=None))
    engine.device = "cpu"
    engine.tokenizer, engine.model, engine.label_ids = Tokenizer(), Model(), list(range(26))
    result = engine.predict(request())
    assert engine.model.calls == 1
    assert result["label"] == "B"
    assert abs(result["confidence"] - 0.7310586) < 1e-6
    assert result["needs_review"]


def test_runtime_fallback_retries_backend_but_does_not_generate(monkeypatch):
    engine = DecisionEngine(Settings(_env_file=None))
    engine.devices, engine.device, engine.tokenizer = ["cuda", "mps", "cpu"], "cuda", Tokenizer()
    tried = []

    def forward(inputs):
        tried.append(engine.device)
        if engine.device != "cpu":
            raise RuntimeError(f"{engine.device} backend not supported")
        return torch.tensor([1.0, 2.0] + [0.0] * 24)

    monkeypatch.setattr(engine, "_forward", forward)
    monkeypatch.setattr(engine, "_release", lambda: None)
    monkeypatch.setattr(
        engine, "_load_device", lambda i: setattr(engine, "device", engine.devices[i])
    )
    result = engine.predict(request())
    assert tried == ["cuda", "mps", "cpu"]
    assert result["device"] == "cpu"
    assert engine.fallbacks == 2
