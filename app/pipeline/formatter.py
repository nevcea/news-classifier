from app.classifier.interface import ClassificationResult


def format_item(item: dict, result: ClassificationResult) -> str:
    title = str(item.get("title", ""))[:100]
    source = str(item.get("source", ""))
    return (
        f"[{result.predicted_label:<10}] "
        f"score={result.top_score:.3f} margin={result.margin:.3f} | "
        f"{title} | {source}"
    )


def format_all(pairs: list[tuple[dict, ClassificationResult]]) -> str:
    return "\n".join(format_item(item, result) for item, result in pairs)
