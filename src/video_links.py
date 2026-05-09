"""Utilities for loading video reference links for user-guide responses."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

_YT_URL_PATTERN = re.compile(r"https?://(?:www\.)?(?:youtube\.com|youtu\.be)/\S+", re.IGNORECASE)
_MD_LINK_PATTERN = re.compile(
    r"\[(?P<title>[^\]]+)\]\((?P<url>https?://(?:www\.)?(?:youtube\.com|youtu\.be)/[^)\s]+)\)",
    re.IGNORECASE,
)


def _clean_url(url: str) -> str:
    return url.strip().rstrip(").,;]")


def _dedupe(items: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for item in items:
        url = _clean_url(item.get("url", ""))
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append({"title": item.get("title", "").strip(), "url": url})
    return deduped


def _parse_text_lines(lines: list[str]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue

        md_match = _MD_LINK_PATTERN.search(raw)
        if md_match:
            items.append(
                {
                    "title": md_match.group("title").strip(),
                    "url": _clean_url(md_match.group("url")),
                }
            )
            continue

        urls = _YT_URL_PATTERN.findall(raw)
        if not urls:
            continue

        if "|" in raw:
            parts = [part.strip() for part in raw.split("|", 1)]
            title = parts[0]
            url = _clean_url(parts[1])
            items.append({"title": title, "url": url})
            continue

        if "," in raw and not raw.lower().startswith("http"):
            parts = [part.strip() for part in raw.split(",", 1)]
            title = parts[0]
            url = _clean_url(parts[1])
            if _YT_URL_PATTERN.search(url):
                items.append({"title": title, "url": url})
                continue

        items.append({"title": "", "url": _clean_url(urls[0])})

    return _dedupe(items)


def _load_json(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items: list[dict[str, str]] = []

    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, str):
                if _YT_URL_PATTERN.search(entry):
                    items.append({"title": "", "url": _clean_url(entry)})
            elif isinstance(entry, dict):
                url = str(entry.get("url", "")).strip()
                if url and _YT_URL_PATTERN.search(url):
                    items.append(
                        {
                            "title": str(entry.get("title", "")).strip(),
                            "url": _clean_url(url),
                        }
                    )

    return _dedupe(items)


def _load_csv(path: Path) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            url = str(row.get("url", "")).strip()
            if not url or not _YT_URL_PATTERN.search(url):
                continue
            items.append(
                {
                    "title": str(row.get("title", "")).strip(),
                    "url": _clean_url(url),
                }
            )
    return _dedupe(items)


def load_guide_video_links(file_path: str) -> list[dict[str, str]]:
    """Load YouTube links from a file path.

    Supported formats:
    - .txt/.md: one URL per line, or `title|url`, or markdown links
    - .csv: columns `title,url`
    - .json: list of strings or objects with keys `title,url`
    """
    path = Path(file_path)
    if not path.exists() or path.is_dir():
        return []

    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            return _load_json(path)
        except Exception:
            return []
    if suffix == ".csv":
        try:
            return _load_csv(path)
        except Exception:
            return []

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    return _parse_text_lines(lines)
