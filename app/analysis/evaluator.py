from collections import Counter
import logging
from pathlib import Path

from app.analysis.jsonl_utils import load_jsonl_records
from app.classifier.interface import BaseClassifier
from app.pipeline.input_builder import build_input

logger = logging.getLogger(__name__)

_MAX_FILE_BYTES = 100 * 1024 * 1024  # 100 MB


def _load_jsonl(path: Path) -> list[dict]:
    try:
        return load_jsonl_records(path, max_file_bytes=_MAX_FILE_BYTES)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Evaluation file not found: {path}") from exc
    except ValueError as exc:
        if "exceeds size limit" in str(exc):
            raise ValueError(f"Evaluation file exceeds size limit: {path}") from exc
        raise


def evaluate(
    classifier: BaseClassifier,
    jsonl_path: Path,
    input_mode: str = "title_plus_summary",
) -> dict:
    """Run classifier against labelled JSONL and return evaluation metrics.

    Input JSONL fields: title, summary, expected_label (required), plus any extras.
    Records without a non-empty expected_label are skipped.

    Returns dict with: n, accuracy, per_label, macro_f1, confusion_matrix, other_rate.

    macro_f1 is computed over expected labels only (standard macro-average definition).
    Labels that appear only in predictions are included in per_label and
    confusion_matrix but excluded from macro_f1 to avoid deflation.

    Uses classify_batch() for efficient bulk inference.
    """
    records = _load_jsonl(jsonl_path)

    items: list[dict] = []
    y_true: list[str] = []

    for record in records:
        expected = str(record.get("expected_label", "")).strip().upper()
        if not expected:
            continue
        items.append(
            {
                "title": str(record.get("title") or ""),
                "summary": str(record.get("summary") or ""),
            }
        )
        y_true.append(expected)

    if not y_true:
        return {"error": "No valid labelled records found"}

    texts = [build_input(item, mode=input_mode)[0] for item in items]
    predictions = classifier.classify_batch(texts)
    y_pred = [r.predicted_label for r in predictions]

    all_labels = sorted(set(y_true) | set(y_pred))
    return _compute_metrics(y_true, y_pred, all_labels)


def _compute_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    n = len(y_true)
    if n == 0:
        return {"error": "No valid labelled records found"}

    # Build confusion counts in a single O(n) pass.
    confusion_counts: Counter[tuple[str, str]] = Counter(zip(y_true, y_pred, strict=True))

    accuracy = sum(confusion_counts[(t, t)] for t in set(y_true)) / n

    per_label: dict[str, dict] = {}
    for label in labels:
        tp = confusion_counts[(label, label)]
        fp = sum(confusion_counts[(t, label)] for t in labels if t != label)
        fn = sum(confusion_counts[(label, p)] for p in labels if p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    # Macro F1: average only over labels that actually appear in ground truth.
    # Labels appearing only in predictions (e.g. unexpected "OTHER") are excluded
    # to match the standard sklearn macro-average definition.
    expected_labels = sorted(set(y_true))
    macro_f1 = (
        sum(per_label[lbl]["f1"] for lbl in expected_labels) / len(expected_labels)
        if expected_labels
        else 0.0
    )

    # Build confusion matrix from pre-computed counts (O(|labels|²) instead of O(n)).
    confusion_matrix: dict[str, dict[str, int]] = {lbl: dict.fromkeys(labels, 0) for lbl in labels}
    for (t, p), count in confusion_counts.items():
        if t in confusion_matrix and p in confusion_matrix[t]:
            confusion_matrix[t][p] = count

    other_rate = sum(1 for p in y_pred if p == "OTHER") / n

    return {
        "n": n,
        "accuracy": round(accuracy, 4),
        "per_label": per_label,
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": confusion_matrix,
        "other_rate": round(other_rate, 4),
    }
