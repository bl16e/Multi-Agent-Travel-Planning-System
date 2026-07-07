from __future__ import annotations

import re
from typing import Any, Literal
from urllib.parse import quote_plus, urlparse


LinkConfidence = Literal["canonical", "provider_result", "search_fallback", "unavailable"]


def resolve_place_links(place: dict[str, Any]) -> dict[str, Any]:
    title = str(place.get("title") or place.get("name") or "").strip()
    provider_link = _clean_url(place.get("link") or place.get("maps_link"))
    official_url = _clean_url(place.get("website") or place.get("official_url"))
    provider_place_id = place.get("provider_place_id") or place.get("place_id") or place.get("id")
    google_place_link = _google_maps_place_url(title, provider_place_id)
    if google_place_link and not provider_link:
        provider_link = google_place_link
    search_url = f"https://ditu.amap.com/search?query={quote_plus(title)}" if title else None

    if provider_link:
        return {
            "canonical_url": provider_link,
            "official_url": official_url if (official_url and _looks_official(official_url)) else None,
            "search_url": search_url,
            "link_confidence": "provider_result",
            "provider": str(place.get("provider") or _provider_from_url(provider_link) or ""),
            "provider_place_id": str(provider_place_id) if provider_place_id else None,
        }
    if official_url and _looks_official(official_url):
        return {
            "canonical_url": official_url,
            "official_url": official_url,
            "search_url": search_url,
            "link_confidence": "canonical",
            "provider": str(place.get("provider") or _provider_from_url(official_url) or ""),
            "provider_place_id": str(provider_place_id) if provider_place_id else None,
        }
    return {
        "canonical_url": None,
        "official_url": None,
        "search_url": search_url,
        "link_confidence": "search_fallback" if search_url else "unavailable",
        "provider": str(place.get("provider") or ""),
        "provider_place_id": str(provider_place_id) if provider_place_id else None,
    }


def _clean_url(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text.startswith(("http://", "https://")):
        return None
    return text


def _looks_official(url: str) -> bool:
    """Algorithmic heuristic: does this URL look like an official site (not UGC/blog/social)?

    Strategy (no hardcoded domain lists):
      1. Parse the URL structure.
      2. Reject if the subdomain contains digits mixed with letters
         (e.g. ``nightshanghai66666.blog.163.com``) – these are user accounts.
      3. Reject if the path looks like user-generated content
         (/blog/, /u/, /~, /static/ followed by digits).
      4. Reject if the host has >3 dotted segments (deep subdomains
         are rarely official).
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    host = (parsed.hostname or "").lower()
    if not host:
        return False

    # Host with >3 segments is almost always a user/hosting subdomain
    # e.g. "nightshanghai66666.blog.163.com" has 4 segments
    parts = host.split(".")
    if len(parts) > 3:
        return False

    # Reject subdomains that mix digits with letters – typical of
    # user accounts on hosting platforms (e.g. "user123.example.com")
    if len(parts) >= 3:
        subdomain = parts[0]
        if _looks_like_user_account(subdomain):
            return False

    path = (parsed.path or "").lower()
    # Paths that indicate user-generated content
    for segment in path.split("/"):
        if segment in {"blog", "u", "user", "~"}:
            return False
    # /static/ followed by digits → blog platform attachment
    if re.search(r"/static/\d", path):
        return False
    # /blog/ anywhere in path
    if "/blog/" in path or path.startswith("/blog"):
        return False

    return True


def _looks_like_user_account(subdomain: str) -> bool:
    """Returns True if *subdomain* looks like a user/account name on a hosting platform.

    Signals: contains both letters AND digits, or is purely numeric.
    """
    has_alpha = any(c.isalpha() for c in subdomain)
    has_digit = any(c.isdigit() for c in subdomain)
    if has_alpha and has_digit:
        return True
    if subdomain.isdigit():
        return True
    return False


def _provider_from_url(value: str) -> str | None:
    lowered = value.lower()
    if "google." in lowered:
        return "google"
    if "amap.com" in lowered or "ditu." in lowered:
        return "amap"
    return None


def _google_maps_place_url(title: str, provider_place_id: Any) -> str | None:
    if not title or not provider_place_id:
        return None
    place_id = str(provider_place_id).strip()
    if not place_id.startswith("ChI"):
        return None
    return f"https://www.google.com/maps/search/?api=1&query={quote_plus(title)}&query_place_id={quote_plus(place_id)}"
