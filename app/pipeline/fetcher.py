from concurrent.futures import ThreadPoolExecutor
import logging
import urllib.error
from urllib.parse import urlsplit
import urllib.request

import feedparser

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 10
_MAX_CONTENT_BYTES = 10 * 1024 * 1024  # 10 MB
_MAX_ENTRIES = 500
_MAX_WORKERS = 8
_USER_AGENT = "TechNewsClassifier/1.0"
_ALLOWED_URL_SCHEMES = {"http", "https"}


def fetch_feed(feed: dict) -> list[dict]:
    """Fetch a single RSS feed and return raw entry dicts.

    Network content is treated as untrusted. SSL verification is on by default.
    Content size is capped to prevent memory exhaustion from oversized feeds.
    """
    url = feed["url"]
    group = feed.get("group", "")
    scheme = urlsplit(url).scheme.lower()
    if scheme not in _ALLOWED_URL_SCHEMES:
        logger.warning("Skipping feed with unsupported URL scheme %r: %s", scheme, url)
        return []

    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})  # noqa: S310
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:  # noqa: S310
            content = resp.read(_MAX_CONTENT_BYTES)
    except urllib.error.URLError as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return []
    except OSError as exc:
        logger.warning("Network error fetching %s: %s", url, exc)
        return []

    parsed = feedparser.parse(content)
    if parsed.bozo:
        logger.warning(
            "Malformed feed %s: %s",
            url,
            parsed.get("bozo_exception", "unknown error"),
        )

    entries = parsed.get("entries", [])[:_MAX_ENTRIES]
    return [{"_raw": entry, "_source_url": url, "_source_group": group} for entry in entries]


def fetch_all(feeds: list[dict]) -> list[dict]:
    """Fetch all configured feeds concurrently; failures are logged and skipped.

    Feeds are fetched in parallel using a thread pool (up to _MAX_WORKERS threads).
    Results are yielded in the original feed order to keep output deterministic.
    """
    if not feeds:
        return []

    items: list[dict] = []
    workers = min(len(feeds), _MAX_WORKERS)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # executor.map preserves submission order, unlike as_completed.
        for result in executor.map(fetch_feed, feeds):
            items.extend(result)

    return items
