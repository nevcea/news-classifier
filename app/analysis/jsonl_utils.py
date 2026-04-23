import json
from pathlib import Path
from typing import Any

_MAX_JSONL_FILE_BYTES = 100 * 1024 * 1024  # 100 MB


def load_jsonl_records(
    path: Path,
    max_file_bytes: int = _MAX_JSONL_FILE_BYTES,
) -> list[dict[str, Any]]:
    """Load a JSONL file as a list of JSON objects."""
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    if path.stat().st_size > max_file_bytes:
        raise ValueError(f"JSONL file exceeds size limit: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Line {lineno} must be a JSON object")
            records.append(item)
    return records
