import json
from pathlib import Path
import shutil
from uuid import uuid4

import pytest

from app.analysis.jsonl_utils import load_jsonl_records


def _case_dir() -> Path:
    path = Path("tests") / "_tmp_runtime" / f"jsonl_utils_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_load_jsonl_records_reads_objects():
    case_dir = _case_dir()
    path = case_dir / "records.jsonl"
    path.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")

    try:
        records = load_jsonl_records(path)
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)

    assert records == [{"a": 1}, {"b": 2}]


def test_load_jsonl_records_rejects_non_object_lines():
    case_dir = _case_dir()
    path = case_dir / "records.jsonl"
    path.write_text('"text"\n', encoding="utf-8")

    try:
        with pytest.raises(ValueError, match="JSON object"):
            load_jsonl_records(path)
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_load_jsonl_records_rejects_invalid_json():
    case_dir = _case_dir()
    path = case_dir / "records.jsonl"
    path.write_text("{bad json}\n", encoding="utf-8")

    try:
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_jsonl_records(path)
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_load_jsonl_records_enforces_size_limit():
    case_dir = _case_dir()
    path = case_dir / "records.jsonl"
    path.write_text(json.dumps({"a": "x" * 20}), encoding="utf-8")

    try:
        with pytest.raises(ValueError, match="size limit"):
            load_jsonl_records(path, max_file_bytes=5)
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)
