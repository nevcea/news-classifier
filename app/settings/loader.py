import math
from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
_MAX_FILE_BYTES = 1024 * 1024  # 1 MB
_MAX_LABEL_LEN = 100


def _validate_label_threshold_map(
    value: object,
    field_name: str,
    known_labels: set[str],
) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"categories.yaml: '{field_name}' must be a mapping when provided")

    result: dict[str, float] = {}
    for label, raw_value in value.items():
        if not isinstance(label, str) or not label:
            continue
        if label not in known_labels:
            continue
        result[label] = _validated_threshold(raw_value, f"{field_name}.{label}", 0.0)
    return result


def _validate_pair_threshold_map(
    value: object,
    field_name: str,
    known_labels: set[str],
) -> dict[str, dict[str, float]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"categories.yaml: '{field_name}' must be a mapping when provided")

    result: dict[str, dict[str, float]] = {}
    for top_label, raw_nested in value.items():
        if not isinstance(top_label, str) or top_label not in known_labels:
            continue
        if not isinstance(raw_nested, dict):
            raise ValueError(
                f"categories.yaml: '{field_name}.{top_label}' must be a mapping when provided"
            )
        nested: dict[str, float] = {}
        for second_label, raw_threshold in raw_nested.items():
            if not isinstance(second_label, str) or second_label not in known_labels:
                continue
            nested[second_label] = _validated_threshold(
                raw_threshold,
                f"{field_name}.{top_label}.{second_label}",
                0.0,
            )
        if nested:
            result[top_label] = nested
    return result


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.stat().st_size > _MAX_FILE_BYTES:
        raise ValueError(f"Config file exceeds size limit: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def _validated_threshold(value: object, name: str, default: float) -> float:
    """Parse and range-check a threshold value from config."""
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v) or not (0.0 <= v <= 1.0):
        raise ValueError(
            f"categories.yaml: '{name}' must be a finite float in [0, 1], got {value!r}"
        )
    return v


def load_feeds() -> list[dict]:
    """Return validated feed dicts from rss.yaml.

    Supports two formats:
      - Flat list:   feeds: [{url: ..., group: ...}]
      - Source groups: feeds: {source_groups: {group_name: [url, ...]}}
    Group names are stored as metadata only — never used as classification labels.
    """
    data = _load_yaml(_CONFIG_DIR / "rss.yaml")
    feeds_section = data.get("feeds", [])

    if isinstance(feeds_section, list):
        return _parse_flat_feeds(feeds_section)
    if isinstance(feeds_section, dict):
        return _parse_grouped_feeds(feeds_section.get("source_groups", {}))
    raise ValueError("rss.yaml: 'feeds' must be a list or mapping")


def _parse_flat_feeds(raw: list) -> list[dict]:
    result: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        url = item.get("url", "")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            continue
        group = item.get("group", "")
        result.append({"url": url, "group": str(group) if group else ""})
    return result


def _parse_grouped_feeds(source_groups: object) -> list[dict]:
    result: list[dict] = []
    if not isinstance(source_groups, dict):
        return result
    for group_name, urls in source_groups.items():
        if not isinstance(urls, list):
            continue
        group = str(group_name) if group_name else ""
        for url in urls:
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                continue
            result.append({"url": url, "group": group})
    return result


def load_categories() -> dict:
    """Return category prototypes and thresholds from categories.yaml."""
    data = _load_yaml(_CONFIG_DIR / "categories.yaml")
    raw_cats = data.get("categories", {})
    if not isinstance(raw_cats, dict):
        raise ValueError("categories.yaml: 'categories' must be a mapping")
    raw_thresh = data.get("thresholds", {})

    categories: dict[str, list[str]] = {}
    for label, cfg in raw_cats.items():
        if not isinstance(label, str) or not label:
            continue
        if len(label) > _MAX_LABEL_LEN:
            continue
        prototypes: list[str] = []
        if isinstance(cfg, dict):
            raw_protos = cfg.get("prototypes", [])
            if isinstance(raw_protos, list):
                prototypes = [str(p) for p in raw_protos if isinstance(p, str) and p]
        categories[label] = prototypes

    thresh = raw_thresh if isinstance(raw_thresh, dict) else {}
    min_score = _validated_threshold(thresh.get("min_score", 0.30), "min_score", 0.30)
    min_margin = _validated_threshold(thresh.get("min_margin", 0.05), "min_margin", 0.05)
    min_score_by_label = _validate_label_threshold_map(
        thresh.get("min_score_by_label"),
        "min_score_by_label",
        set(categories),
    )
    min_margin_by_label = _validate_label_threshold_map(
        thresh.get("min_margin_by_label"),
        "min_margin_by_label",
        set(categories),
    )
    min_margin_by_pair = _validate_pair_threshold_map(
        thresh.get("min_margin_by_pair"),
        "min_margin_by_pair",
        set(categories),
    )

    return {
        "categories": categories,
        "min_score": min_score,
        "min_margin": min_margin,
        "min_score_by_label": min_score_by_label,
        "min_margin_by_label": min_margin_by_label,
        "min_margin_by_pair": min_margin_by_pair,
    }
