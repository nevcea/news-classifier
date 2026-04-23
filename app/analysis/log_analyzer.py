import argparse
from collections import Counter
from pathlib import Path
from statistics import fmean
import sys
from typing import Any

from app.analysis.jsonl_utils import load_jsonl_records

_DEFAULT_LOG_DIR = Path("logs")
_DEFAULT_RECENT = 3
_MAX_EXAMPLES = 8


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return load_jsonl_records(path)


def _item_key(row: dict[str, Any]) -> str:
    link = str(row.get("link") or "").strip()
    if link:
        return link
    title = str(row.get("title") or "").strip()
    published = str(row.get("published_at") or "").strip()
    return f"{title}|{published}"


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    values = [float(row.get(field, 0.0) or 0.0) for row in rows]
    return fmean(values) if values else 0.0


def summarize_log(path: Path) -> dict[str, Any]:
    rows = _load_jsonl(path)
    count = len(rows)
    predicted_counts = Counter(str(row.get("predicted_label") or "") for row in rows)
    other_rows = [row for row in rows if str(row.get("predicted_label") or "") == "OTHER"]
    reject_counts = Counter(str(row.get("reject_reason") or "") for row in other_rows)
    top_label_other_counts = Counter(str(row.get("top_label") or "") for row in other_rows)

    return {
        "path": path,
        "count": count,
        "other_count": predicted_counts.get("OTHER", 0),
        "other_rate": (predicted_counts.get("OTHER", 0) / count) if count else 0.0,
        "avg_top_score": _mean(rows, "top_score"),
        "avg_second_score": _mean(rows, "second_score"),
        "avg_margin": _mean(rows, "margin"),
        "predicted_counts": predicted_counts,
        "reject_counts": reject_counts,
        "top_label_other_counts": top_label_other_counts,
    }


def compare_logs(old_path: Path, new_path: Path) -> dict[str, Any]:
    old_rows = _load_jsonl(old_path)
    new_rows = _load_jsonl(new_path)
    new_by_key = {_item_key(row): row for row in new_rows}

    transition_counts: Counter[tuple[str, str]] = Counter()
    restored_examples: list[dict[str, Any]] = []
    dropped_examples: list[dict[str, Any]] = []

    for old_row in old_rows:
        key = _item_key(old_row)
        new_row = new_by_key.get(key)
        if new_row is None:
            continue
        old_label = str(old_row.get("predicted_label") or "")
        new_label = str(new_row.get("predicted_label") or "")
        if old_label == new_label:
            continue
        transition_counts[(old_label, new_label)] += 1

        example = {
            "title": str(new_row.get("title") or ""),
            "old_label": old_label,
            "new_label": new_label,
            "top_label": str(new_row.get("top_label") or ""),
            "top_score": float(new_row.get("top_score", 0.0) or 0.0),
            "second_label": str(new_row.get("second_label") or ""),
            "margin": float(new_row.get("margin", 0.0) or 0.0),
            "reject_reason": str(new_row.get("reject_reason") or ""),
        }
        if old_label == "OTHER" and new_label != "OTHER":
            restored_examples.append(example)
        elif old_label != "OTHER" and new_label == "OTHER":
            dropped_examples.append(example)

    restored_examples.sort(key=lambda item: item["top_score"], reverse=True)
    dropped_examples.sort(key=lambda item: item["top_score"], reverse=True)

    return {
        "old_path": old_path,
        "new_path": new_path,
        "transition_counts": transition_counts,
        "restored_examples": restored_examples[:_MAX_EXAMPLES],
        "dropped_examples": dropped_examples[:_MAX_EXAMPLES],
    }


def _top_counts(counter: Counter[Any], limit: int = 8) -> list[tuple[Any, int]]:
    return counter.most_common(limit)


def _format_summary(summary: dict[str, Any]) -> list[str]:
    path = Path(summary["path"])
    predicted_counts: Counter[str] = summary["predicted_counts"]
    reject_counts: Counter[str] = summary["reject_counts"]
    top_label_other_counts: Counter[str] = summary["top_label_other_counts"]

    lines = [
        f"File: {path.name}",
        (
            f"Items={summary['count']} OTHER={summary['other_count']} "
            f"({summary['other_rate']:.1%}) avg_top={summary['avg_top_score']:.4f} "
            f"avg_second={summary['avg_second_score']:.4f} avg_margin={summary['avg_margin']:.4f}"
        ),
        "Predicted labels: "
        + ", ".join(f"{label}:{count}" for label, count in _top_counts(predicted_counts)),
    ]
    if reject_counts:
        lines.append(
            "OTHER reject reasons: "
            + ", ".join(f"{label}:{count}" for label, count in _top_counts(reject_counts))
        )
    if top_label_other_counts:
        lines.append(
            "OTHER top-label candidates: "
            + ", ".join(f"{label}:{count}" for label, count in _top_counts(top_label_other_counts))
        )
    return lines


def _format_compare(compare: dict[str, Any]) -> list[str]:
    lines = [
        (
            f"Compare: {Path(compare['old_path']).name} -> "
            f"{Path(compare['new_path']).name}"
        )
    ]
    transitions: Counter[tuple[str, str]] = compare["transition_counts"]
    if not transitions:
        lines.append("No label changes detected.")
        return lines

    lines.append(
        "Transitions: "
        + ", ".join(
            f"{old}->{new}:{count}"
            for (old, new), count in transitions.most_common(_MAX_EXAMPLES)
        )
    )

    restored = compare["restored_examples"]
    if restored:
        lines.append("Top OTHER->label recoveries:")
        for item in restored:
            lines.append(
                f"- {item['new_label']} | {item['title']} "
                f"(score={item['top_score']:.2f}, margin={item['margin']:.2f})"
            )

    dropped = compare["dropped_examples"]
    if dropped:
        lines.append("Top label->OTHER drops:")
        for item in dropped:
            lines.append(
                f"- {item['old_label']} | {item['title']} "
                f"(top={item['top_label']}, score={item['top_score']:.2f}, "
                f"reason={item['reject_reason'] or 'n/a'})"
            )

    return lines


def build_report(log_dir: Path, recent: int) -> str:
    files = sorted(log_dir.glob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not files:
        return f"No .jsonl files found in {log_dir}"

    selected = files[: max(recent, 1)]
    lines = ["Tech News Log Analysis", ""]
    for path in selected:
        lines.extend(_format_summary(summarize_log(path)))
        lines.append("")

    if len(selected) >= 2:
        lines.extend(_format_compare(compare_logs(selected[1], selected[0])))

    return "\n".join(lines).rstrip()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Analyze recent classification JSONL logs.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=_DEFAULT_LOG_DIR,
        help="Directory containing classification .jsonl logs.",
    )
    parser.add_argument(
        "--recent",
        type=int,
        default=_DEFAULT_RECENT,
        help="How many recent log files to summarize.",
    )
    args = parser.parse_args()

    print(build_report(args.log_dir, args.recent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
