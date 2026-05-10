from fastapi import APIRouter, Depends, HTTPException

from app.dtos.ml import (
    ModelMetricsDto,
    PredictionInputDto,
    PredictionOutputDto,
    TrainingRequestDto,
    TrainingResultDto,
)
from app.services.ml_service import MachineLearningService

router = APIRouter(prefix="/ml")


@router.post(
    "/train",
    description="Treina um modelo LSTM com histórico de fechamento de ações.",
    response_model=TrainingResultDto,
    responses={
        200: {
            "description": "Treinamento concluído com sucesso.",
            "content": {
                "application/json": {
                    "example": {
                        "ticker": "DIS",
                        "start_date": "2022-01-01",
                        "end_date": "2023-12-31",
                        "lookback_window_size": 20,
                        "mae": 1.82,
                        "rmse": 2.31,
                        "mape": 1.74,
                        "trained_at": "2026-05-10T05:50:00+00:00",
                        "validation_samples": 96,
                    }
                }
            },
        },
        400: {"description": "Payload inválido ou erro de coleta de dados."},
    },
)
async def train_model(
    payload: TrainingRequestDto,
    ml_service: MachineLearningService = Depends(),
):
    try:
        return ml_service.train_model(payload)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post(
    "/predict",
    description="Prevê próximo valor de fechamento com base no histórico recente.",
    response_model=PredictionOutputDto,
    responses={
        200: {
            "description": "Predição realizada com sucesso.",
            "content": {
                "application/json": {
                    "example": {
                        "predicted_close": 103.42,
                        "ticker": "DIS",
                        "lookback_window_size": 20,
                    }
                }
            },
        },
        400: {"description": "Payload inválido ou modelo ainda não treinado."},
    },
)
async def predict_price(
    payload: PredictionInputDto,
    ml_service: MachineLearningService = Depends(),
):
    try:
        return ml_service.predict(payload)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get(
    "/metrics",
    description="Recupera métricas do último treinamento executado.",
    response_model=ModelMetricsDto,
    responses={
        200: {
            "description": "Métricas do último treino.",
            "content": {
                "application/json": {
                    "example": {
                        "ticker": "DIS",
                        "start_date": "2022-01-01",
                        "end_date": "2023-12-31",
                        "lookback_window_size": 20,
                        "mae": 1.82,
                        "rmse": 2.31,
                        "mape": 1.74,
                        "trained_at": "2026-05-10T05:50:00+00:00",
                        "validation_samples": 96,
                    }
                }
            },
        },
        404: {"description": "Nenhum treinamento encontrado."},
    },
)
async def get_model_metrics(ml_service: MachineLearningService = Depends()):
    try:
        return ml_service.get_latest_metrics()
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
