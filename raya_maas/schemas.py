"""Internal single-forward input; HTTP clients use decisions.SystemOneRequest."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=65536)]
Candidate = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2048)]
SUFFIX = "Answer with exactly one label from the options above."


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Media(StrictModel):
    type: Literal["image", "video"]
    url: str = Field(
        min_length=1,
        description="Matching base64 data URL or trusted HTTPS URL. Video duration must be "
        "3–20 seconds inclusive; sampled with Qwen fps=1 rules. The server checks stream metadata.",
    )

    @model_validator(mode="after")
    def check_url(self):
        if not (self.url.startswith(f"data:{self.type}/") or self.url.startswith("https://")):
            raise ValueError("Media url must be a matching base64 data URL or trusted HTTPS URL")
        return self


class InferenceInput(StrictModel):
    text: Text
    candidates: list[Candidate] = Field(min_length=2, max_length=26)
    media: list[Media] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def check_tokens(self):
        if "<|" in self.text or any("<|" in item for item in self.candidates):
            raise ValueError("Raw model special tokens are not allowed")
        if any("\n" in item for item in self.candidates):
            raise ValueError("Candidate descriptions must be single-line text")
        return self


def render(request: InferenceInput) -> tuple[str, list[str], list[tuple[str, str]]]:
    media = [(item.type, item.url) for item in request.media]
    pieces = [f"<|vision_start|><|{kind}_pad|><|vision_end|>" for kind, _ in media]
    pieces.append(request.text)
    pieces.append("\n".join(f"{chr(65 + i)}. {item}" for i, item in enumerate(request.candidates)))
    pieces.append(SUFFIX)
    return "\n\n".join(pieces), request.candidates, media
