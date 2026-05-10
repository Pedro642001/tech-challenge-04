from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.settings import settings


class TrainingRequestDto(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "ticker": "DIS",
                "start_date": "2022-01-01",
                "end_date": "2023-12-31",
                "lookback_window_size": 20,
                "training_data_ratio": 0.8,
                "training_epochs": 5,
                "samples_per_batch": 32,
                "lstm_hidden_units": 50,
                "learning_rate": 0.001,
            }
        },
    )

    ticker: str = Field(default=settings.DEFAULT_TICKER, description="Ticker da ação (ex.: AAPL)")
    start_date: str = Field(
        default=settings.DEFAULT_START_DATE,
        description="Data inicial para coleta histórica (YYYY-MM-DD)",
    )
    end_date: str | None = Field(
        default=settings.DEFAULT_END_DATE,
        description="Data final para coleta histórica (YYYY-MM-DD). Nulo usa data atual.",
    )
    lookback_window_size: int = Field(
        default=settings.SEQUENCE_LENGTH,
        ge=5,
        le=365,
        description="Quantidade de fechamentos passados usados para prever o próximo valor.",
    )
    training_data_ratio: float = Field(
        default=0.8,
        gt=0.5,
        lt=0.95,
        description="Percentual do dataset reservado para treino (restante para validação).",
    )
    training_epochs: int = Field(
        default=20,
        ge=1,
        le=300,
        description="Número de épocas de treinamento.",
    )
    samples_per_batch: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Quantidade de amostras processadas por lote durante o treino.",
    )
    lstm_hidden_units: int = Field(
        default=50,
        ge=8,
        le=512,
        description="Número de neurônios (unidades de memória) na camada LSTM.",
    )
    learning_rate: float = Field(default=0.001, gt=0.0, le=0.1)


class TrainingResultDto(BaseModel):
    ticker: str
    start_date: str
    end_date: str | None
    lookback_window_size: int
    mae: float
    rmse: float
    mape: float
    trained_at: str
    validation_samples: int


class PredictionInputDto(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "recent_closing_prices": [
                    96.3,
                    96.7,
                    97.1,
                    97.0,
                    97.6,
                    98.0,
                    97.9,
                    98.3,
                    98.7,
                    99.1,
                    99.4,
                    99.8,
                    100.1,
                    100.4,
                    100.8,
                    101.0,
                    101.3,
                    101.7,
                    102.1,
                    102.5,
                ]
            }
        },
    )

    recent_closing_prices: list[float] = Field(
        ...,
        description="Lista de fechamentos recentes com tamanho igual à janela usada no treino.",
    )

    @field_validator("recent_closing_prices")
    @classmethod
    def validate_recent_closing_prices(cls, value: list[float]):
        if not value:
            raise ValueError("recent_closing_prices não pode ser vazio.")
        return value


class PredictionOutputDto(BaseModel):
    predicted_close: float
    ticker: str
    lookback_window_size: int


class ModelMetricsDto(BaseModel):
    ticker: str
    start_date: str
    end_date: str | None
    lookback_window_size: int
    mae: float
    rmse: float
    mape: float
    trained_at: str
    validation_samples: int
