import json
from pathlib import Path

import pytest

from app.analysis.evaluator import evaluate
from app.classifier.interface import BaseClassifier, ClassificationResult


class _FixedClassifier(BaseClassifier):
    """Always returns the same label."""

    def __init__(self, label: str) -> None:
        self._label = label

    def classify(self, text: str) -> ClassificationResult:
        return ClassificationResult(
            predicted_label=self._label,
            top_label=self._label,
            second_label="OTHER",
            top_score=0.9,
            second_score=0.2,
            margin=0.7,
        )


def _write(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


# --- Accuracy ---


def test_perfect_accuracy(tmp_path):
    p = tmp_path / "eval.jsonl"
    _write(
        p,
        [
            {"title": "a", "summary": "", "expected_label": "AI"},
            {"title": "b", "summary": "", "expected_label": "AI"},
        ],
    )
    metrics = evaluate(_FixedClassifier("AI"), p)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_zero_accuracy(tmp_path):
    p = tmp_path / "eval.jsonl"
    _write(p, [{"title": "x", "summary": "", "expected_label": "SECURITY"}])
    metrics = evaluate(_FixedClassifier("AI"), p)
    assert metrics["accuracy"] == 0.0


# --- OTHER rate ---


def test_other_rate_all_other(tmp_path):
    p = tmp_path / "eval.jsonl"
    _write(
        p,
        [
            {"title": "a", "summary": "", "expected_label": "AI"},
            {"title": "b", "summary": "", "expected_label": "CLOUD"},
        ],
    )
    metrics = evaluate(_FixedClassifier("OTHER"), p)
    assert metrics["other_rate"] == 1.0


def test_other_rate_none_other(tmp_path):
    p = tmp_path / "eval.jsonl"
    _write(p, [{"title": "a", "summary": "", "expected_label": "AI"}])
    metrics = evaluate(_FixedClassifier("AI"), p)
    assert metrics["other_rate"] == 0.0


# --- Skipping ---


def test_missing_expected_label_skipped(tmp_path):
    p = tmp_path / "eval.jsonl"
    _write(
        p,
        [
            {"title": "a", "summary": ""},  # no expected_label
            {"title": "b", "summary": "", "expected_label": ""},  # empty
            {"title": "c", "summary": "", "expected_label": "AI"},
        ],
    )
    metrics = evaluate(_FixedClassifier("AI"), p)
    assert metrics["n"] == 1


def test_empty_file_returns_error(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    metrics = evaluate(_FixedClassifier("AI"), p)
    assert "error" in metrics


# --- Confusion matrix ---


def test_confusion_matrix_off_diagonal(tmp_path):
    p = tmp_path / "eval.jsonl"
    _write(p, [{"title": "a", "summary": "", "expected_label": "AI"}])
    metrics = evaluate(_FixedClassifier("SECURITY"), p)
    assert metrics["confusion_matrix"]["AI"]["SECURITY"] == 1
    assert metrics["confusion_matrix"]["AI"]["AI"] == 0


def test_confusion_matrix_diagonal(tmp_path):
    p = tmp_path / "eval.jsonl"
    _write(p, [{"title": "a", "summary": "", "expected_label": "AI"}])
    metrics = evaluate(_FixedClassifier("AI"), p)
    assert metrics["confusion_matrix"]["AI"]["AI"] == 1


# --- Per-label metrics ---


def test_per_label_f1_perfect(tmp_path):
    p = tmp_path / "eval.jsonl"
    _write(p, [{"title": "a", "summary": "", "expected_label": "AI"}])
    metrics = evaluate(_FixedClassifier("AI"), p)
    assert metrics["per_label"]["AI"]["f1"] == 1.0
    assert metrics["per_label"]["AI"]["precision"] == 1.0
    assert metrics["per_label"]["AI"]["recall"] == 1.0


# --- Macro F1 semantics ---


def test_macro_f1_excludes_prediction_only_labels(tmp_path):
    """Classifier predicts 'OTHER' for everything, but expected labels never include OTHER.
    macro_f1 must be computed over expected labels only, not deflated by OTHER having f1=0.
    """
    p = tmp_path / "eval.jsonl"
    _write(
        p,
        [
            {"title": "a", "summary": "", "expected_label": "AI"},
            {"title": "b", "summary": "", "expected_label": "CLOUD"},
        ],
    )
    metrics = evaluate(_FixedClassifier("OTHER"), p)

    # "OTHER" appears in per_label (it was predicted) but has support=0 as expected
    assert "OTHER" in metrics["per_label"]
    assert metrics["per_label"]["OTHER"]["support"] == 0

    # macro_f1 must only average AI and CLOUD — not OTHER
    expected_labels = ["AI", "CLOUD"]
    expected_macro = sum(metrics["per_label"][lbl]["f1"] for lbl in expected_labels) / len(
        expected_labels
    )
    assert metrics["macro_f1"] == round(expected_macro, 4)

    # Both AI and CLOUD have recall=0 (never correctly predicted), so macro_f1=0
    assert metrics["macro_f1"] == 0.0


def test_macro_f1_over_expected_labels_only(tmp_path):
    """Verify macro_f1 ignores extra predicted labels regardless of their count."""
    p = tmp_path / "eval.jsonl"
    _write(p, [{"title": "a", "summary": "", "expected_label": "AI"}])
    # Classifier predicts correctly — "SECURITY" never appears in predictions here
    metrics = evaluate(_FixedClassifier("AI"), p)

    # Only "AI" is in expected labels → macro_f1 = AI f1 = 1.0
    assert metrics["macro_f1"] == 1.0
    assert "AI" in metrics["per_label"]


# --- File not found ---


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate(_FixedClassifier("AI"), tmp_path / "missing.jsonl")
