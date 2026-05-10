import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Any


def configure_logging(level: str, log_file_path: str | None = None) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(numeric_level)

    formatter = logging.Formatter("%(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    if log_file_path:
        file_path = Path(log_file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def log_event(logger: logging.Logger, level: str, event: str, **fields: Any) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level.lower(),
        "logger": logger.name,
        "event": event,
        **fields,
    }
    message = json.dumps(payload, ensure_ascii=False, default=str)
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(message)
