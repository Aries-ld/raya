"""Raya decision protocol: Jev-shaped questions, with explicit multimodal extensions."""

import json
import math
import time
from typing import Annotated, Literal

from pydantic import Field, JsonValue, StringConstraints, field_validator, model_validator

from .errors import APIError
from .schemas import InferenceInput, Media, StrictModel, render

Content = str | dict[str, JsonValue] | list[JsonValue]
Identifier = Annotated[str, StringConstraints(min_length=1, max_length=128)]


def content_text(value: Content) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def check_content(value, limit=65536):
    if not value or not content_text(value).strip():
        raise ValueError("Content must not be empty")
    text = content_text(value)
    if len(text) > limit:
        raise ValueError(f"Content exceeds {limit} characters")
    if "<|" in text:
        raise ValueError("Raw model special tokens are not allowed")
    return value


class QuestionBase(StrictModel):
    instructions: Content

    @field_validator("instructions")
    @classmethod
    def check_instructions(cls, value):
        return check_content(value)


class ChoiceQuestion(QuestionBase):
    type: Literal["choice"]
    criteria: dict[Identifier, Content | None] = Field(min_length=2, max_length=26)

    @field_validator("criteria")
    @classmethod
    def check_criteria(cls, value):
        for key, description in value.items():
            if "<|" in key or "\n" in key:
                raise ValueError("Option IDs cannot contain special tokens or newlines")
            if description is not None:
                check_content(description, 1900)
        return value


class ScoreQuestion(QuestionBase):
    type: Literal["score"]
    criteria: list[Content] = Field(min_length=2, max_length=10)

    @field_validator("criteria")
    @classmethod
    def check_criteria(cls, value):
        for description in value:
            check_content(description, 1900)
        return value


class NoulCriteria(StrictModel):
    true: Content = "Yes, the statement is true."
    false: Content = "No, the statement is false."

    @field_validator("true", "false")
    @classmethod
    def check_criteria(cls, value):
        return check_content(value, 1900)


class NoulQuestion(QuestionBase):
    type: Literal["noul"]
    criteria: NoulCriteria = Field(default_factory=NoulCriteria)


Question = Annotated[ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")]


class MultimodalState(StrictModel):
    text: Content = ""
    media: list[Media] = Field(min_length=1, max_length=8)


def state_parts(state: Content | MultimodalState) -> tuple[Content, list[Media]]:
    if isinstance(state, MultimodalState):
        return state.text, state.media
    if isinstance(state, dict) and "media" in state:
        multimodal = MultimodalState.model_validate(state)
        return multimodal.text, multimodal.media
    return state, []


class SystemOneRequest(StrictModel):
    model: str
    state: MultimodalState | Content
    questions: dict[Identifier, Question] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def check_state(self):
        context, media = state_parts(self.state)
        if context or not media:
            check_content(context)
        # Construct every question before inference: malformed criteria fail atomically.
        for question in self.questions.values():
            make_input(self, question)
        return self


def one_line(value: Content) -> str:
    return " ".join(content_text(value).splitlines())


def make_input(request: SystemOneRequest, question: Question) -> InferenceInput:
    if isinstance(question, ChoiceQuestion):
        candidates = [
            key if value is None else f"{key}: {one_line(value)}"
            for key, value in question.criteria.items()
        ]
    elif isinstance(question, ScoreQuestion):
        candidates = [f"{i}: {one_line(value)}" for i, value in enumerate(question.criteria)]
    else:
        candidates = [
            f"Yes: {one_line(question.criteria.true)}",
            f"No: {one_line(question.criteria.false)}",
        ]
    context, media = state_parts(request.state)
    return InferenceInput(
        text=f"State:\n{content_text(context)}\n\nQuestion:\n{content_text(question.instructions)}",
        candidates=candidates,
        media=media,
    )


def to_answer(question: Question, result: dict) -> dict:
    probabilities = list(result["probabilities"].values())
    if isinstance(question, NoulQuestion):
        return {"type": "noul", "noul": probabilities[0]}
    confidence = max(probabilities)
    if isinstance(question, ChoiceQuestion):
        keys = list(question.criteria)
        return {
            "type": "choice",
            "choice": keys[result["index"]],
            "probabilities": dict(zip(keys, probabilities)),
            "confidence": confidence,
        }
    return {
        "type": "score",
        "score": sum(i * p for i, p in enumerate(probabilities)),
        "legend": {str(i): content_text(value) for i, value in enumerate(question.criteria)},
        "probabilities": {str(i): p for i, p in enumerate(probabilities)},
        "confidence": confidence,
    }


class ChoiceAnswer(StrictModel):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0, le=1)


class ScoreAnswer(StrictModel):
    type: Literal["score"]
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float = Field(ge=0, le=1)


class NoulAnswer(StrictModel):
    type: Literal["noul"]
    noul: float = Field(ge=0, le=1)


Answer = Annotated[ChoiceAnswer | ScoreAnswer | NoulAnswer, Field(discriminator="type")]


class Usage(StrictModel):
    input_tokens: int
    output_tokens: Literal[0] = 0


class SystemOneResponse(StrictModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage


def evaluate(engine, request: SystemOneRequest) -> dict:
    started = time.perf_counter()
    if request.model != engine.settings.model_name:
        raise APIError("Model not found", 404, "model_not_found")
    inputs = {key: make_input(request, question) for key, question in request.questions.items()}
    for value in inputs.values():
        if len(engine.tokenizer.encode(render(value)[0])) > engine.settings.max_input_tokens:
            raise APIError(
                "State plus question exceeds token limit", 422, "context_length_exceeded"
            )
    prepared_media = engine.prepare_media(
        [(item.type, item.url) for item in state_parts(request.state)[1]]
    )
    media_ms = (time.perf_counter() - started) * 1000
    # Media decoding is shared; each question still has its own forward pass.
    answers, diagnostics, total_tokens = {}, {}, 0
    for key, question in request.questions.items():
        result = engine.predict(inputs[key], prepared_media=prepared_media)
        probabilities = list(result["probabilities"].values())
        if not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities) or not math.isclose(
            sum(probabilities), 1.0, abs_tol=1e-5
        ):
            raise RuntimeError("Invalid probability distribution")
        answers[key] = to_answer(question, result)
        total_tokens += result["prompt_tokens"]
        diagnostics[key] = {
            "input_tokens": result["prompt_tokens"],
            "device": result["device"],
            "timing_ms": result["timing_ms"],
        }
    engine.last_diagnostics = {
        "execution": "sequential",
        "confidence_method": "max_probability",
        "question_count": len(answers),
        "media_prepare_ms": round(media_ms, 3),
        "video": prepared_media[1],
        "total_ms": round((time.perf_counter() - started) * 1000, 3),
        "questions": diagnostics,
    }
    return {
        "model": request.model,
        "answers": answers,
        "usage": {"input_tokens": total_tokens, "output_tokens": 0},
    }
