from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
    )
    groq_api_key: str
    groq_model: str
    embedding_model: str
    chroma_path: str
    chroma_collection: str
    chunk_size: int
    chunk_overlap: int
    top_k: int

    @field_validator("chroma_path")
    @classmethod
    def resolve_chroma_path(cls, v: str) -> str:
        """Resolve relative path against the project root to make it absolute."""
        path = Path(v)
        if path.is_absolute():
            return str(path)
        return str(PROJECT_ROOT / path)


settings = Settings()  # type: ignore[call-arg]
