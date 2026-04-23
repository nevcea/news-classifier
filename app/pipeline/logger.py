import atexit
from contextlib import suppress
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
from typing import TextIO

from app.classifier.interface import ClassificationResult

_LOG_DIR = Path("logs")
_MAX_STR_LEN = 5000

logger = logging.getLogger(__name__)

# Fixed once per process so all items in a single run share the same log file.
_SESSION_TS = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")

# File handle kept open for the lifetime of the process (or until _LOG_DIR changes).
# Opened lazily on first log_item call; closed via atexit and when the dir changes.
_handle: TextIO | None = None
_handle_dir: Path | None = None  # tracks which _LOG_DIR the open handle belongs to


def _get_handle() -> TextIO:
    """Return the open log file handle, (re)creating it when _LOG_DIR has changed.

    _LOG_DIR is replaced by tests via unittest.mock.patch, so we compare the current
    value against the directory the handle was created for.  If they differ, we close
    the old handle and open a new one in the new directory.
    """
    global _handle, _handle_dir
    if _handle is None or _handle_dir != _LOG_DIR:
        if _handle is not None:
            with suppress(OSError):
                _handle.close()
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = _LOG_DIR / f"classified_{_SESSION_TS}.jsonl"
        # buffering=1 → line-buffered: each newline flushes the buffer,
        # so records are visible on disk even if the process is interrupted.
        _handle = path.open("a", encoding="utf-8", buffering=1)
        _handle_dir = _LOG_DIR
        atexit.register(_handle.close)
    return _handle


def _safe_str(value: object) -> str:
    return str(value)[:_MAX_STR_LEN] if value is not None else ""


def log_item(
    item: dict,
    result: ClassificationResult,
    extra: dict | None = None,
) -> None:
    """Append a structured JSONL record for future training dataset use.

    Args:
        item: normalized item dict.
        result: classification result.
        extra: optional context — supports keys: input_text, used_summary,
               model_info, classifier_version, threshold, margin_threshold.
    """
    extra = extra or {}
    record = {
        "title": _safe_str(item.get("title")),
        "summary": _safe_str(item.get("summary")),
        "input_text": _safe_str(extra.get("input_text")),
        "predicted_label": _safe_str(result.predicted_label),
        "top_label": _safe_str(result.top_label),
        "second_label": _safe_str(result.second_label),
        "top_score": round(result.top_score, 6),
        "second_score": round(result.second_score, 6),
        "margin": round(result.margin, 6),
        "applied_min_score": result.applied_min_score,
        "applied_min_margin": result.applied_min_margin,
        "reject_reason": _safe_str(result.reject_reason),
        "source": _safe_str(item.get("source")),
        "source_group": _safe_str(item.get("source_group")),
        "link": _safe_str(item.get("link")),
        "published_at": _safe_str(item.get("published_at")),
        "model_info": _safe_str(extra.get("model_info")),
        "classifier_version": _safe_str(extra.get("classifier_version")),
        "threshold": extra.get("threshold"),
        "margin_threshold": extra.get("margin_threshold"),
        "used_summary": bool(extra.get("used_summary", False)),
        "logged_at": datetime.now(UTC).isoformat(),
    }
    try:
        _get_handle().write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.error("Failed to write log record: %s", exc)
