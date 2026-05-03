from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
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
