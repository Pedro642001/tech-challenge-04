from fastapi import APIRouter, HTTPException

from app.core.observability import render_prometheus_metrics
from app.core.settings import settings

router = APIRouter()


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    if not settings.ENABLE_PROMETHEUS_METRICS:
        raise HTTPException(status_code=404, detail="Metrics endpoint is disabled.")
    return render_prometheus_metrics()
