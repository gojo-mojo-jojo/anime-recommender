from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str
    tavily_api_key: str
    tmdb_api_key: str = ""
    port: int = 8000

    model_config = {"env_file": ".env"}


settings = Settings()  # type: ignore[call-arg]
