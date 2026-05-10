import json
import logging
from datetime import datetime, timezone
from typing import Any


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="%(message)s")


def log_event(logger: logging.Logger, level: str, event: str, **fields: Any) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    message = json.dumps(payload, ensure_ascii=False, default=str)
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(message)
