import os
from time import time
from datetime import datetime, timezone

import psutil
from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

_APP_STARTED_AT = time()

HTTP_REQUESTS_TOTAL = Counter(
    "app_http_requests_total",
    "Total HTTP requests handled by the application.",
    ("method", "route", "status_code"),
)

HTTP_REQUEST_ERRORS_TOTAL = Counter(
    "app_http_request_errors_total",
    "Total HTTP request errors handled by the application.",
    ("method", "route", "status_code", "error_type"),
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "app_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ("method", "route", "status_code"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)

HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "app_http_requests_in_progress",
    "Number of in-progress HTTP requests.",
)

SYSTEM_CPU_PERCENT = Gauge(
    "app_system_cpu_percent",
    "Current system CPU usage percentage.",
)

SYSTEM_MEMORY_PERCENT = Gauge(
    "app_system_memory_percent",
    "Current system memory usage percentage.",
)

PROCESS_MEMORY_MB = Gauge(
    "app_process_memory_mb",
    "Current process memory usage in MB.",
)

APP_START_TIME_TIMESTAMP = Gauge(
    "app_start_time_timestamp",
    "UNIX timestamp for when the application started.",
)

APP_UPTIME_SECONDS = Gauge(
    "app_uptime_seconds",
    "Uptime of the application in seconds.",
)

ML_TRAINING_TOTAL = Counter(
    "app_ml_training_total",
    "Total number of ML training events.",
    ("status",),
)

ML_TRAINING_DURATION_SECONDS = Histogram(
    "app_ml_training_duration_seconds",
    "Training duration in seconds.",
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1200, 1800),
)

ML_PREDICTION_TOTAL = Counter(
    "app_ml_prediction_total",
    "Total number of ML prediction events.",
    ("status",),
)

ML_PREDICTION_DURATION_SECONDS = Histogram(
    "app_ml_prediction_duration_seconds",
    "Prediction duration in seconds.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

ML_LAST_TRAINING_TIMESTAMP = Gauge(
    "app_ml_last_training_timestamp",
    "UNIX timestamp for the last successful training.",
)

ML_ACTIVE_OPERATIONS = Gauge(
    "app_ml_active_operations",
    "Current number of active ML operations by type.",
    ("operation",),
)


def resolve_route_path(request: Request) -> str:
    route = request.scope.get("route")
    if route and getattr(route, "path", None):
        return str(route.path)
    return request.url.path


def refresh_system_metrics() -> None:
    process = psutil.Process(os.getpid())
    PROCESS_MEMORY_MB.set(process.memory_info().rss / 1024 / 1024)
    SYSTEM_CPU_PERCENT.set(psutil.cpu_percent(interval=None))
    SYSTEM_MEMORY_PERCENT.set(psutil.virtual_memory().percent)
    APP_UPTIME_SECONDS.set(max(0.0, time() - _APP_STARTED_AT))


def mark_app_started() -> None:
    global _APP_STARTED_AT
    _APP_STARTED_AT = time()
    APP_START_TIME_TIMESTAMP.set(_APP_STARTED_AT)


def mark_http_request_started() -> None:
    HTTP_REQUESTS_IN_PROGRESS.inc()


def mark_http_request_finished() -> None:
    HTTP_REQUESTS_IN_PROGRESS.dec()


def record_http_metrics(
    method: str,
    route: str,
    status_code: int,
    duration_seconds: float,
    error_type: str | None = None,
) -> None:
    status = str(status_code)
    HTTP_REQUESTS_TOTAL.labels(method=method, route=route, status_code=status).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(method=method, route=route, status_code=status).observe(
        duration_seconds
    )
    if error_type:
        HTTP_REQUEST_ERRORS_TOTAL.labels(
            method=method,
            route=route,
            status_code=status,
            error_type=error_type,
        ).inc()


def mark_ml_training_started() -> None:
    ML_TRAINING_TOTAL.labels(status="started").inc()
    ML_ACTIVE_OPERATIONS.labels(operation="training").inc()


def mark_ml_training_finished(duration_seconds: float, success: bool) -> None:
    ML_TRAINING_DURATION_SECONDS.observe(duration_seconds)
    status = "succeeded" if success else "failed"
    ML_TRAINING_TOTAL.labels(status=status).inc()
    ML_ACTIVE_OPERATIONS.labels(operation="training").dec()
    if success:
        ML_LAST_TRAINING_TIMESTAMP.set(datetime.now(timezone.utc).timestamp())


def mark_ml_prediction_started() -> None:
    ML_PREDICTION_TOTAL.labels(status="started").inc()
    ML_ACTIVE_OPERATIONS.labels(operation="prediction").inc()


def mark_ml_prediction_finished(duration_seconds: float, success: bool) -> None:
    ML_PREDICTION_DURATION_SECONDS.observe(duration_seconds)
    status = "succeeded" if success else "failed"
    ML_PREDICTION_TOTAL.labels(status=status).inc()
    ML_ACTIVE_OPERATIONS.labels(operation="prediction").dec()


def render_prometheus_metrics() -> Response:
    refresh_system_metrics()
    payload = generate_latest()
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)
