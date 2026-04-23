import logging
import sys

from app.classifier.embedding import EmbeddingClassifier
from app.classifier.interface import ClassificationResult
from app.pipeline import deduplicator, fetcher, formatter, normalizer
from app.pipeline import logger as app_logger
from app.pipeline.input_builder import build_input
from app.settings import loader as config_loader

# Reconfigure stdout to UTF-8 so titles with non-ASCII characters (e.g. emoji,
# CJK, accented letters) print correctly on Windows terminals that default to
# a legacy code page such as cp949 or cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
_log = logging.getLogger(__name__)


def main() -> int:
    try:
        feeds = config_loader.load_feeds()
    except Exception as exc:
        _log.error("Failed to load rss.yaml: %s", exc)
        return 1

    try:
        cat_config = config_loader.load_categories()
    except Exception as exc:
        _log.error("Failed to load categories.yaml: %s", exc)
        return 1

    if not feeds:
        _log.error("No valid feeds found in rss.yaml")
        return 1

    if not cat_config["categories"]:
        _log.error("No valid categories found in categories.yaml")
        return 1

    classifier = EmbeddingClassifier(
        categories=cat_config["categories"],
        min_score=cat_config["min_score"],
        min_margin=cat_config["min_margin"],
        min_score_by_label=cat_config["min_score_by_label"],
        min_margin_by_label=cat_config["min_margin_by_label"],
        min_margin_by_pair=cat_config["min_margin_by_pair"],
    )

    _log.info("Fetching %d feed(s) concurrently...", len(feeds))
    raw_items = fetcher.fetch_all(feeds)
    _log.info("Fetched %d raw entries", len(raw_items))

    items = normalizer.normalize_all(raw_items)
    _log.info("Normalized to %d items", len(items))

    items = deduplicator.deduplicate(items)
    _log.info("After deduplication: %d items", len(items))

    # Build classifier inputs, tracking per-item failures before the batch call.
    valid_items: list[dict] = []
    texts: list[str] = []
    used_summaries: list[bool] = []
    skipped = 0

    for item in items:
        try:
            text, used_summary = build_input(item, mode="title_plus_summary")
            valid_items.append(item)
            texts.append(text)
            used_summaries.append(used_summary)
        except Exception as exc:
            _log.warning(
                "Skipping item (input build failed) '%s': %s",
                str(item.get("title", ""))[:80],
                exc,
            )
            skipped += 1

    # Classify all items in a single batched encode call.
    _log.info("Classifying %d items...", len(valid_items))
    try:
        results = classifier.classify_batch(texts)
    except Exception as exc:
        _log.error("Batch classification failed: %s", exc)
        return 1

    pairs: list[tuple[dict, ClassificationResult]] = []
    for item, result, text, used_summary in zip(
        valid_items, results, texts, used_summaries, strict=True
    ):
        pairs.append((item, result))
        app_logger.log_item(
            item,
            result,
            extra={
                "input_text": text,
                "used_summary": used_summary,
                "model_info": classifier.model_info,
                "classifier_version": "1.0",
                "threshold": cat_config["min_score"],
                "margin_threshold": cat_config["min_margin"],
            },
        )

    if skipped:
        _log.warning("Skipped %d item(s) due to input build errors.", skipped)
    print(formatter.format_all(pairs))
    _log.info("Done. Classified %d items.", len(pairs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
