from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAYA_", env_file=".env", extra="ignore")

    model_path: str = "models/raya-decision-v1"
    processor_path: str = "models/qwen3.5-2b-processor"
    model_name: str = "raya-decision-v1"
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    api_keys: SecretStr = SecretStr("")
    allow_anonymous: bool = False
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)
    max_input_tokens: int = Field(4096, ge=128, le=32768)
    max_body_bytes: int = Field(96 * 1024 * 1024, gt=0)
    max_media_bytes: int = Field(64 * 1024 * 1024, gt=0)
    max_image_pixels: int = Field(20_000_000, gt=0)
    image_max_pixels: int = Field(262144, ge=65536)
    video_max_pixels: int = Field(131072, ge=65536)
    max_video_seconds: float = Field(60, gt=0, le=600)
    max_video_decode_frames: int = Field(18000, gt=0)
    max_media: int = Field(4, ge=1, le=8)
    media_hosts: str = ""
    queue_size: int = Field(8, ge=1, le=128)
    request_timeout: float = Field(120, gt=0)
    media_timeout: float = Field(15, gt=0)
    cpu_threads: int = Field(4, ge=1)

    @property
    def keys(self) -> list[str]:
        return [key.strip() for key in self.api_keys.get_secret_value().split(",") if key.strip()]
