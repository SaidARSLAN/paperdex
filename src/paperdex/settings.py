from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PATH = Path(__file__).parent.parent.parent / ".env"


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


settings = Settings()  # type: ignore[call-arg]
