import json
from unittest.mock import patch

from app.classifier.interface import ClassificationResult
from app.pipeline.logger import log_item


def _result(**kwargs) -> ClassificationResult:
    defaults = {
        "predicted_label": "AI",
        "top_label": "AI",
        "second_label": "DEV",
        "top_score": 0.85,
        "second_score": 0.60,
        "margin": 0.25,
    }
    defaults.update(kwargs)
    return ClassificationResult(**defaults)


def _item() -> dict:
    return {
        "title": "Test title",
        "summary": "Test summary",
        "source": "https://example.com/feed",
        "source_group": "general_sources",
        "link": "https://example.com/article/1",
        "published_at": "2026-01-01T00:00:00+00:00",
    }


# --- Record structure ---


def test_all_required_fields_present(tmp_path):
    required = [
        "title",
        "summary",
        "input_text",
        "predicted_label",
        "top_label",
        "second_label",
        "top_score",
        "second_score",
        "margin",
        "applied_min_score",
        "applied_min_margin",
        "reject_reason",
        "source",
        "source_group",
        "link",
        "published_at",
        "model_info",
        "classifier_version",
        "threshold",
        "margin_threshold",
        "used_summary",
        "logged_at",
    ]
    extra = {
        "input_text": "Test title Test summary",
        "used_summary": True,
        "model_info": "all-MiniLM-L6-v2",
        "classifier_version": "1.0",
        "threshold": 0.3,
        "margin_threshold": 0.05,
    }
    with patch("app.pipeline.logger._LOG_DIR", tmp_path):
        log_item(_item(), _result(), extra=extra)

    records = list(tmp_path.glob("*.jsonl"))
    assert len(records) == 1
    data = json.loads(records[0].read_text(encoding="utf-8"))
    for field in required:
        assert field in data, f"Missing field: {field}"


def test_scores_are_floats(tmp_path):
    with patch("app.pipeline.logger._LOG_DIR", tmp_path):
        log_item(_item(), _result())
    data = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
    assert isinstance(data["top_score"], float)
    assert isinstance(data["second_score"], float)
    assert isinstance(data["margin"], float)


# --- Append mode ---


def test_append_produces_multiple_lines(tmp_path):
    with patch("app.pipeline.logger._LOG_DIR", tmp_path):
        log_item(_item(), _result())
        log_item(_item(), _result(predicted_label="CLOUD", top_label="CLOUD"))

    lines = next(tmp_path.glob("*.jsonl")).read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["predicted_label"] == "AI"
    assert json.loads(lines[1])["predicted_label"] == "CLOUD"


# --- Safety ---


def test_long_string_truncated(tmp_path):
    item = _item()
    item["title"] = "X" * 10_000
    with patch("app.pipeline.logger._LOG_DIR", tmp_path):
        log_item(item, _result())
    data = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
    assert len(data["title"]) <= 5000


def test_no_extra_still_logs(tmp_path):
    with patch("app.pipeline.logger._LOG_DIR", tmp_path):
        log_item(_item(), _result())  # extra=None
    data = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
    assert data["input_text"] == ""
    assert data["used_summary"] is False


def test_used_summary_flag(tmp_path):
    with patch("app.pipeline.logger._LOG_DIR", tmp_path):
        log_item(_item(), _result(), extra={"used_summary": True})
    data = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
    assert data["used_summary"] is True


def test_threshold_null_when_no_extra(tmp_path):
    """threshold and margin_threshold must be JSON null when extra is omitted."""
    with patch("app.pipeline.logger._LOG_DIR", tmp_path):
        log_item(_item(), _result())
    data = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
    assert data["threshold"] is None
    assert data["margin_threshold"] is None


def test_threshold_values_logged_when_provided(tmp_path):
    with patch("app.pipeline.logger._LOG_DIR", tmp_path):
        log_item(_item(), _result(), extra={"threshold": 0.3, "margin_threshold": 0.05})
    data = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
    assert data["threshold"] == 0.3
    assert data["margin_threshold"] == 0.05


def test_reject_reason_and_applied_thresholds_logged(tmp_path):
    with patch("app.pipeline.logger._LOG_DIR", tmp_path):
        log_item(
            _item(),
            _result(
                predicted_label="OTHER",
                applied_min_score=0.37,
                applied_min_margin=0.07,
                reject_reason="margin",
            ),
        )
    data = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
    assert data["applied_min_score"] == 0.37
    assert data["applied_min_margin"] == 0.07
    assert data["reject_reason"] == "margin"
