from fastapi import FastAPI

from app.controllers.health_check import router as health_router
from app.controllers.machine_learning import router as machine_learning_router
from app.controllers.observability import router as observability_router
from app.core.settings import settings


def register_routers(app: FastAPI):
    prefix = settings.API_PREFIX
    app.include_router(health_router, tags=["Health"], prefix=prefix)
    app.include_router(machine_learning_router, tags=["Machine Learning"], prefix=prefix)
    app.include_router(observability_router, tags=["Observability"])
