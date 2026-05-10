import os

import psutil
from fastapi import Request


class HealthService:
    def check_health(self) -> dict[str, str]:
        return {"status": "ok"}

    def get_metrics(self, request: Request) -> dict[str, float | int]:
        total_requests = request.app.state.total_requests
        total_latency = request.app.state.total_latency_seconds
        avg_latency_ms = (total_latency / total_requests) * 1000 if total_requests else 0.0

        process = psutil.Process(os.getpid())
        process_memory_mb = process.memory_info().rss / 1024 / 1024

        return {
            "total_requests": total_requests,
            "average_latency_ms": round(avg_latency_ms, 3),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "process_memory_mb": round(process_memory_mb, 3),
        }
