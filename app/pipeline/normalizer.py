from datetime import UTC, datetime
import html
import logging
import re

logger = logging.getLogger(__name__)

_MAX_TITLE_LEN = 300
_MAX_SUMMARY_LEN = 2000

# Bounded pattern prevents ReDoS on malformed HTML.
_HTML_TAG_RE = re.compile(r"<[^>]{0,500}>")
_WHITESPACE_RE = re.compile(r"\s+")


def _strip_html(text: str) -> str:
    text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _truncate(text: str, max_len: int) -> str:
    return text[:max_len] if len(text) > max_len else text


def normalize_item(raw: dict) -> dict | None:
    """Convert a raw feedparser entry dict to a clean normalized item."""
    entry = raw.get("_raw", {})
    source_url = str(raw.get("_source_url", ""))
    source_group = str(raw.get("_source_group", ""))

    title = entry.get("title", "")
    if not isinstance(title, str):
        title = ""
    title = _truncate(_strip_html(title), _MAX_TITLE_LEN)

    if not title:
        logger.debug("Skipping entry with no title from %s", source_url)
        return None

    link = entry.get("link", "")
    if not isinstance(link, str):
        link = ""
    link = link.strip()
    if link and not link.startswith(("http://", "https://")):
        link = ""

    summary = entry.get("summary", "") or entry.get("description", "")
    if not isinstance(summary, str):
        summary = ""
    summary = _truncate(_strip_html(summary), _MAX_SUMMARY_LEN)

    published_at = ""
    if entry.get("published_parsed"):
        try:
            t = entry.published_parsed
            dt = datetime(*t[:6], tzinfo=UTC)
            published_at = dt.isoformat()
        except Exception as exc:
            logger.debug("Could not parse published_parsed from %s: %s", source_url, exc)

    return {
        "title": title,
        "link": link,
        "summary": summary,
        "source": source_url,
        "source_group": source_group,
        "published_at": published_at,
    }


def normalize_all(raw_items: list[dict]) -> list[dict]:
    result: list[dict] = []
    for raw in raw_items:
        item = normalize_item(raw)
        if item is not None:
            result.append(item)
    return result
