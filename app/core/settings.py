from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Tech Challenge 04 - LSTM Stocks"
    API_PREFIX: str = "/api/v1"
    MODEL_DIR: str = "app/data"
    LOG_LEVEL: str = "INFO"
    ENABLE_PROMETHEUS_METRICS: bool = True
    DEFAULT_TICKER: str = "DIS"
    DEFAULT_START_DATE: str = "2018-01-01"
    DEFAULT_END_DATE: str | None = None
    SEQUENCE_LENGTH: int = 60

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
