import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

from app.analysis.jsonl_utils import load_jsonl_records

_DEFAULT_LOG_DIR = Path("logs")
_DEFAULT_OUTPUT = Path("data") / "eval_seed.jsonl"
_DEFAULT_RECENT = 3


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return load_jsonl_records(path)


def _item_key(row: dict[str, Any]) -> str:
    link = str(row.get("link") or "").strip()
    if link:
        return link
    title = str(row.get("title") or "").strip()
    published = str(row.get("published_at") or "").strip()
    return f"{title}|{published}"


def _recent_logs(log_dir: Path, recent: int) -> list[Path]:
    return sorted(log_dir.glob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)[
        : max(recent, 1)
    ]


def build_seed_dataset(
    log_dir: Path,
    recent: int,
    min_votes: int,
    min_avg_top_score: float,
    include_other: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files = _recent_logs(log_dir, recent)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for path in files:
        for row in _load_jsonl(path):
            grouped[_item_key(row)].append(row)

    records: list[dict[str, Any]] = []
    skipped = Counter[str]()

    for rows in grouped.values():
        label_counts = Counter(str(row.get("predicted_label") or "") for row in rows)
        if not label_counts:
            skipped["empty"] += 1
            continue

        consensus_label, votes = label_counts.most_common(1)[0]
        if consensus_label == "OTHER" and not include_other:
            skipped["other"] += 1
            continue
        if votes < min_votes:
            skipped["low_votes"] += 1
            continue
        if sum(1 for count in label_counts.values() if count == votes) > 1:
            skipped["tie"] += 1
            continue

        non_other_labels = {label for label in label_counts if label and label != "OTHER"}
        if len(non_other_labels) > 1 and consensus_label != "OTHER":
            skipped["conflicting_non_other"] += 1
            continue

        consensus_rows = [
            row for row in rows if str(row.get("predicted_label") or "") == consensus_label
        ]
        avg_top_score = (
            sum(float(row.get("top_score", 0.0) or 0.0) for row in consensus_rows)
            / len(consensus_rows)
        )
        if avg_top_score < min_avg_top_score:
            skipped["low_score"] += 1
            continue

        base_row = max(consensus_rows, key=lambda row: float(row.get("top_score", 0.0) or 0.0))
        records.append(
            {
                "title": str(base_row.get("title") or ""),
                "summary": str(base_row.get("summary") or ""),
                "expected_label": consensus_label,
                "source": str(base_row.get("source") or ""),
                "source_group": str(base_row.get("source_group") or ""),
                "link": str(base_row.get("link") or ""),
                "published_at": str(base_row.get("published_at") or ""),
                "silver_label": True,
                "label_votes": votes,
                "logs_considered": len(rows),
                "avg_top_score": round(avg_top_score, 6),
                "supporting_labels": dict(sorted(label_counts.items())),
            }
        )

    records.sort(key=lambda row: (row["expected_label"], -row["label_votes"], row["title"]))
    label_distribution = Counter(str(row["expected_label"]) for row in records)
    stats = {
        "files_used": [path.name for path in files],
        "items_seen": len(grouped),
        "records_written": len(records),
        "label_distribution": dict(sorted(label_distribution.items())),
        "skipped": dict(sorted(skipped.items())),
    }
    return records, stats


def write_seed_dataset(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _format_stats(output_path: Path, stats: dict[str, Any]) -> str:
    lines = [
        f"Wrote silver eval dataset: {output_path}",
        "Files used: " + ", ".join(stats["files_used"]),
        f"Items seen: {stats['items_seen']}",
        f"Records written: {stats['records_written']}",
    ]
    if stats["label_distribution"]:
        lines.append(
            "Label distribution: "
            + ", ".join(
                f"{label}:{count}" for label, count in stats["label_distribution"].items()
            )
        )
    if stats["skipped"]:
        lines.append(
            "Skipped: "
            + ", ".join(f"{reason}:{count}" for reason, count in stats["skipped"].items())
        )
    return "\n".join(lines)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Build a silver evaluation JSONL dataset from recent classification logs."
    )
    parser.add_argument("--log-dir", type=Path, default=_DEFAULT_LOG_DIR)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--recent", type=int, default=_DEFAULT_RECENT)
    parser.add_argument("--min-votes", type=int, default=2)
    parser.add_argument("--min-avg-top-score", type=float, default=0.45)
    parser.add_argument("--include-other", action="store_true")
    args = parser.parse_args()

    records, stats = build_seed_dataset(
        log_dir=args.log_dir,
        recent=args.recent,
        min_votes=args.min_votes,
        min_avg_top_score=args.min_avg_top_score,
        include_other=args.include_other,
    )
    write_seed_dataset(args.output, records)
    print(_format_stats(args.output, stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
