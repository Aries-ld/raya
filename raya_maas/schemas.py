import re
import string
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=65536)]
Candidate = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2048)]
SUFFIX = "Answer with exactly one label from the options above."


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MediaURL(StrictModel):
    url: str = Field(min_length=1)
    detail: Literal["auto", "low", "high"] = "auto"


class TextPart(StrictModel):
    type: Literal["text"]
    text: Text


class ImagePart(StrictModel):
    type: Literal["image_url"]
    image_url: MediaURL


class VideoPart(StrictModel):
    type: Literal["video_url"]
    video_url: MediaURL


Part = Annotated[TextPart | ImagePart | VideoPart, Field(discriminator="type")]


class Message(StrictModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: Text | list[Part] = Field(min_length=1)


class ResponseFormat(StrictModel):
    type: Literal["text", "json_object"] = "text"


class StreamOptions(StrictModel):
    include_usage: bool = False


class ChatRequest(StrictModel):
    model: str
    messages: list[Message] = Field(min_length=1, max_length=32)
    candidates: list[Candidate] | None = Field(None, min_length=2, max_length=26)
    confidence_threshold: float = Field(0.6, ge=0, le=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    response_format: ResponseFormat = Field(default_factory=ResponseFormat)
    temperature: Literal[0, 1] | None = None
    top_p: Literal[1] | None = None
    n: Literal[1] = 1
    max_tokens: int | None = Field(None, ge=1)
    max_completion_tokens: int | None = Field(None, ge=1)
    user: str | None = Field(None, max_length=256)

    @model_validator(mode="after")
    def check_request(self):
        if self.messages[-1].role != "user":
            raise ValueError("The last message must be a user decision question")
        if self.candidates is not None:
            if len(set(self.candidates)) != len(self.candidates):
                raise ValueError("Candidates must be unique")
            if any("\n" in item or "<|" in item for item in self.candidates):
                raise ValueError("Candidates must be single-line text without special tokens")
        for message in self.messages:
            parts = [message.content] if isinstance(message.content, str) else message.content
            for part in parts:
                text = part if isinstance(part, str) else getattr(part, "text", "")
                if "<|" in text:
                    raise ValueError("Raw model special tokens are not allowed")
                if isinstance(part, ImagePart | VideoPart) and message.role != "user":
                    raise ValueError("Media is only supported in user messages")
        return self


def render(request: ChatRequest) -> tuple[str, list[str], list[tuple[str, str]]]:
    """Keep the training prompt format; do not apply a chat template."""
    blocks, media = [], []
    for message in request.messages:
        if isinstance(message.content, str):
            block = message.content
        else:
            pieces = []
            for part in message.content:
                if isinstance(part, TextPart):
                    pieces.append(part.text)
                else:
                    kind = "image" if isinstance(part, ImagePart) else "video"
                    media.append((kind, getattr(part, f"{kind}_url").url))
                    pieces.append(f"<|vision_start|><|{kind}_pad|><|vision_end|>")
            block = "\n\n".join(pieces)
        blocks.append(block if len(request.messages) == 1 else f"{message.role}: {block}")
    prompt = "\n\n".join(blocks)
    if request.candidates is not None:
        options = request.candidates
        prompt += "\n\n" + "\n".join(
            f"{label}. {option}" for label, option in zip(string.ascii_uppercase, options)
        )
    else:
        last = blocks[-1].removesuffix(SUFFIX).rstrip()
        match = re.search(r"(?:^|\n)(A\. [^\n]+(?:\n[A-Z]\. [^\n]+)+)$", last)
        if match is None:
            raise ValueError("Provide candidates or end the question with A. / B. option lines")
        lines = match.group(1).splitlines()
        labels, options = [line[0] for line in lines], [line[3:].strip() for line in lines]
        if labels != list(string.ascii_uppercase[: len(lines)]) or not 2 <= len(lines) <= 26:
            raise ValueError("Option labels must be contiguous A through Z (2–26 options)")
        if any(not option for option in options) or len(set(options)) != len(options):
            raise ValueError("Options must be non-empty and unique")
        prompt = prompt.removesuffix(SUFFIX).rstrip()
    return prompt + "\n\n" + SUFFIX, options, media
