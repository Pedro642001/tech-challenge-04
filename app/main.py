from time import perf_counter
from uuid import uuid4

import logging

from fastapi import FastAPI, Request

from app.core.logging import configure_logging, log_event
from app.core.observability import record_http_metrics, resolve_route_path
from app.core.routers import register_routers
from app.core.settings import settings


def create_app() -> FastAPI:
    configure_logging(settings.LOG_LEVEL)
    logger = logging.getLogger("app.http")

    app = FastAPI(
        title=settings.APP_NAME,
        description="API para treinamento e inferência de LSTM para previsão de fechamento de ações.",
        swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    )

    app.state.total_requests = 0
    app.state.total_latency_seconds = 0.0

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start_time = perf_counter()
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        request.state.request_id = request_id

        response = None
        status_code = 500
        error_type = None
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception as error:
            error_type = type(error).__name__
            raise
        finally:
            elapsed = perf_counter() - start_time
            route_path = resolve_route_path(request)

            if error_type is None and status_code >= 400:
                error_type = "HTTPStatusError"

            app.state.total_requests += 1
            app.state.total_latency_seconds += elapsed
            record_http_metrics(
                method=request.method,
                route=route_path,
                status_code=status_code,
                duration_seconds=elapsed,
                error_type=error_type,
            )

            level = "info"
            if status_code >= 500 or error_type:
                level = "error"
            elif status_code >= 400:
                level = "warning"

            log_event(
                logger=logger,
                level=level,
                event="http_request",
                request_id=request_id,
                method=request.method,
                route=route_path,
                status_code=status_code,
                duration_ms=round(elapsed * 1000, 3),
            )

            if response is not None:
                response.headers["X-Process-Time"] = f"{elapsed:.6f}"

    register_routers(app)
    return app


app = create_app()
