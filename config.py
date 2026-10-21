from typing import ClassVar

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GroqSettings(BaseModel):
    api_key: str = Field(default="")
    base_url: str = Field(default="https://api.groq.com/openai/v1")
    model: str = Field(default="openai/gpt-oss-20b")


class Settings(BaseSettings):
    groq: GroqSettings = GroqSettings()

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=[".env"],
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
        case_sensitive=False,
    )


settings = Settings()
