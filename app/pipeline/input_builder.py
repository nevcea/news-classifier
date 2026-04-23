_MAX_INPUT_LEN = 512  # characters fed to the embedding model


def build_input(item: dict, mode: str = "title_plus_summary") -> tuple[str, bool]:
    """Build classifier input text from a normalized item.

    Items are expected to already have HTML stripped and whitespace normalized
    (handled by normalizer.py). This module only assembles and truncates.

    Args:
        item: normalized item dict with 'title' and 'summary' keys.
        mode: 'title_only' or 'title_plus_summary'.

    Returns:
        (text, used_summary) — the assembled text and whether summary was included.
    """
    title = str(item.get("title") or "").strip()
    summary = str(item.get("summary") or "").strip()

    if mode == "title_only":
        return title[:_MAX_INPUT_LEN], False

    if mode == "title_plus_summary":
        if summary:
            text = f"{title} {summary}"
            return text[:_MAX_INPUT_LEN], True
        return title[:_MAX_INPUT_LEN], False

    raise ValueError(f"Unknown input mode: {mode!r}")
