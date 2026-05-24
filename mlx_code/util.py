import json
import logging
import uuid
from datetime import datetime, timezone

request_id_var = uuid.uuid4()
_LOG_RECORD_BUILTIN_KEYS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "file": record.pathname,
            "function": record.funcName,
            "line": record.lineno,
            "request_id": str(request_id_var),
        }
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _LOG_RECORD_BUILTIN_KEYS
        }
        if extras:
            log_entry["extra"] = extras
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


def setup_logger(log_file: str = ".log.json", console: bool = False) -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if not root.handlers:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(JsonFormatter())
        root.addHandler(fh)
        if console:
            sh = logging.StreamHandler()
            sh.setFormatter(JsonFormatter())
            root.addHandler(sh)
    return root
