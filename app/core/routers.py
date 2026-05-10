from fastapi import FastAPI

from app.controllers import health_check, machine_learning, observability
from app.core.settings import settings


def register_routers(app: FastAPI):
    prefix = settings.API_PREFIX
    app.include_router(health_check.router, tags=["Health"], prefix=prefix)
    app.include_router(machine_learning.router, tags=["Machine Learning"], prefix=prefix)
    app.include_router(observability.router, tags=["Observability"])
