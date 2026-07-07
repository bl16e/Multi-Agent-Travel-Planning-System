from __future__ import annotations

from typing import Any, Literal
from urllib.parse import quote_plus


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
            "official_url": official_url,
            "search_url": search_url,
            "link_confidence": "provider_result",
            "provider": str(place.get("provider") or _provider_from_url(provider_link) or ""),
            "provider_place_id": str(provider_place_id) if provider_place_id else None,
        }
    if official_url:
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
