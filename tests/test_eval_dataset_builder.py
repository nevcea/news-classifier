import json
from pathlib import Path
import shutil
from uuid import uuid4

from app.analysis.eval_dataset_builder import build_seed_dataset, write_seed_dataset


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _case_dir() -> Path:
    path = Path("tests") / "_tmp_runtime" / f"eval_dataset_builder_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_build_seed_dataset_keeps_consensus_non_other_labels():
    case_dir = _case_dir()
    _write_jsonl(
        case_dir / "classified_1.jsonl",
        [
            {
                "title": "Consensus item",
                "summary": "summary",
                "link": "https://example.com/a",
                "predicted_label": "AI",
                "top_score": 0.61,
            },
            {
                "title": "Other item",
                "summary": "summary",
                "link": "https://example.com/b",
                "predicted_label": "OTHER",
                "top_score": 0.31,
            },
        ],
    )
    _write_jsonl(
        case_dir / "classified_2.jsonl",
        [
            {
                "title": "Consensus item",
                "summary": "summary",
                "link": "https://example.com/a",
                "predicted_label": "AI",
                "top_score": 0.59,
            },
            {
                "title": "Other item",
                "summary": "summary",
                "link": "https://example.com/b",
                "predicted_label": "OTHER",
                "top_score": 0.28,
            },
        ],
    )

    try:
        records, stats = build_seed_dataset(
            log_dir=case_dir,
            recent=2,
            min_votes=2,
            min_avg_top_score=0.45,
        )
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)

    assert len(records) == 1
    assert records[0]["expected_label"] == "AI"
    assert records[0]["label_votes"] == 2
    assert stats["records_written"] == 1
    assert stats["skipped"]["other"] == 1


def test_build_seed_dataset_skips_conflicting_non_other_labels():
    case_dir = _case_dir()
    _write_jsonl(
        case_dir / "classified_1.jsonl",
        [
            {
                "title": "Conflict item",
                "summary": "summary",
                "link": "https://example.com/c",
                "predicted_label": "AI",
                "top_score": 0.7,
            }
        ],
    )
    _write_jsonl(
        case_dir / "classified_2.jsonl",
        [
            {
                "title": "Conflict item",
                "summary": "summary",
                "link": "https://example.com/c",
                "predicted_label": "DEV",
                "top_score": 0.71,
            }
        ],
    )

    try:
        records, stats = build_seed_dataset(
            log_dir=case_dir,
            recent=2,
            min_votes=1,
            min_avg_top_score=0.45,
        )
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)

    assert records == []
    assert stats["skipped"]["tie"] == 1


def test_write_seed_dataset_outputs_jsonl():
    case_dir = _case_dir()
    output = case_dir / "seed.jsonl"
    records = [
        {
            "title": "Saved item",
            "summary": "",
            "expected_label": "BUSINESS",
            "silver_label": True,
            "label_votes": 2,
            "logs_considered": 2,
            "avg_top_score": 0.5,
            "supporting_labels": {"BUSINESS": 2},
        }
    ]

    try:
        write_seed_dataset(output, records)
        saved = json.loads(output.read_text(encoding="utf-8").strip())
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)

    assert saved["expected_label"] == "BUSINESS"
    assert saved["silver_label"] is True
