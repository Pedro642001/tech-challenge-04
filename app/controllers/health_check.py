from fastapi import APIRouter, Depends, Request

from app.services.health_service import HealthService

router = APIRouter(prefix="/health")


@router.get(
    "/",
    description="Verificação de disponibilidade da API.",
    responses={
        200: {
            "description": "API disponível.",
            "content": {"application/json": {"example": {"status": "ok"}}},
        }
    },
)
async def health_check(health_service: HealthService = Depends()):
    return health_service.check_health()


@router.get(
    "/metrics",
    description="Métricas de monitoramento da aplicação.",
    responses={
        200: {
            "description": "Métricas de runtime da API.",
            "content": {
                "application/json": {
                    "example": {
                        "total_requests": 42,
                        "average_latency_ms": 8.317,
                        "cpu_percent": 11.2,
                        "memory_percent": 63.4,
                        "process_memory_mb": 345.128,
                    }
                }
            },
        }
    },
)
async def health_metrics(request: Request, health_service: HealthService = Depends()):
    return health_service.get_metrics(request)
