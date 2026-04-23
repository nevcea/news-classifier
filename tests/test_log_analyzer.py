import json
import os
from pathlib import Path
import shutil
import time
from uuid import uuid4

from app.analysis.log_analyzer import build_report, compare_logs, summarize_log


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _case_dir() -> Path:
    path = Path("tests") / "_tmp_runtime" / f"log_analyzer_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_summarize_log_reports_basic_metrics():
    case_dir = _case_dir()
    path = case_dir / "classified_1.jsonl"
    _write_jsonl(
        path,
        [
            {
                "title": "a",
                "predicted_label": "AI",
                "top_label": "AI",
                "top_score": 0.5,
                "second_score": 0.2,
                "margin": 0.3,
            },
            {
                "title": "b",
                "predicted_label": "OTHER",
                "top_label": "BUSINESS",
                "top_score": 0.4,
                "second_score": 0.38,
                "margin": 0.02,
                "reject_reason": "margin",
            },
        ],
    )

    try:
        summary = summarize_log(path)
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)

    assert summary["count"] == 2
    assert summary["other_count"] == 1
    assert round(summary["other_rate"], 4) == 0.5
    assert summary["predicted_counts"]["AI"] == 1
    assert summary["reject_counts"]["margin"] == 1
    assert summary["top_label_other_counts"]["BUSINESS"] == 1


def test_compare_logs_reports_transitions():
    case_dir = _case_dir()
    old_path = case_dir / "classified_old.jsonl"
    new_path = case_dir / "classified_new.jsonl"
    _write_jsonl(
        old_path,
        [
            {
                "title": "one",
                "link": "https://example.com/1",
                "predicted_label": "OTHER",
                "top_label": "AI",
                "top_score": 0.41,
                "second_label": "DEV",
                "margin": 0.02,
            },
            {
                "title": "two",
                "link": "https://example.com/2",
                "predicted_label": "AI",
                "top_label": "AI",
                "top_score": 0.55,
                "second_label": "BUSINESS",
                "margin": 0.04,
            },
        ],
    )
    _write_jsonl(
        new_path,
        [
            {
                "title": "one",
                "link": "https://example.com/1",
                "predicted_label": "AI",
                "top_label": "AI",
                "top_score": 0.46,
                "second_label": "DEV",
                "margin": 0.05,
            },
            {
                "title": "two",
                "link": "https://example.com/2",
                "predicted_label": "OTHER",
                "top_label": "AI",
                "top_score": 0.50,
                "second_label": "BUSINESS",
                "margin": 0.02,
                "reject_reason": "margin",
            },
        ],
    )

    try:
        compare = compare_logs(old_path, new_path)
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)

    assert compare["transition_counts"][("OTHER", "AI")] == 1
    assert compare["transition_counts"][("AI", "OTHER")] == 1
    assert compare["restored_examples"][0]["title"] == "one"
    assert compare["dropped_examples"][0]["reject_reason"] == "margin"


def test_build_report_includes_summary_and_compare():
    case_dir = _case_dir()
    first = case_dir / "classified_1.jsonl"
    second = case_dir / "classified_2.jsonl"
    _write_jsonl(
        first,
        [
            {
                "title": "old",
                "link": "https://example.com/old",
                "predicted_label": "OTHER",
                "top_label": "BUSINESS",
                "top_score": 0.4,
                "second_label": "AI",
                "second_score": 0.39,
                "margin": 0.01,
                "reject_reason": "margin",
            }
        ],
    )
    now = time.time()
    os.utime(first, (now - 10, now - 10))
    _write_jsonl(
        second,
        [
            {
                "title": "old",
                "link": "https://example.com/old",
                "predicted_label": "BUSINESS",
                "top_label": "BUSINESS",
                "top_score": 0.52,
                "second_label": "AI",
                "second_score": 0.49,
                "margin": 0.03,
            }
        ],
    )
    os.utime(second, (now, now))

    try:
        report = build_report(case_dir, recent=2)
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)

    assert "Tech News Log Analysis" in report
    assert "Compare:" in report
    assert "OTHER->BUSINESS:1" in report
