import re

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _normalize_title(title: str) -> str:
    return _NON_ALNUM.sub("", title.lower())


def deduplicate(items: list[dict]) -> list[dict]:
    """Remove duplicates: first by exact link, then by normalized title."""
    seen_links: set[str] = set()
    seen_titles: set[str] = set()
    result: list[dict] = []

    for item in items:
        link = item.get("link", "")
        if link and link in seen_links:
            continue

        norm_title = _normalize_title(item.get("title", ""))
        if norm_title and norm_title in seen_titles:
            continue

        if link:
            seen_links.add(link)
        if norm_title:
            seen_titles.add(norm_title)
        result.append(item)

    return result
