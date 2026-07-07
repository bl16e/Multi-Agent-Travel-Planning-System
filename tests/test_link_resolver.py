from utils.link_resolver import resolve_place_links


def test_resolve_place_links_prefers_provider_and_official_urls_over_search_url():
    links = resolve_place_links(
        {
            "title": "Shanghai Museum",
            "link": "https://www.google.com/maps/place/Shanghai+Museum",
            "website": "https://www.shanghaimuseum.net/",
        }
    )

    assert links["canonical_url"] == "https://www.google.com/maps/place/Shanghai+Museum"
    assert links["official_url"] == "https://www.shanghaimuseum.net/"
    assert links["link_confidence"] == "provider_result"
    assert links["search_url"] == "https://ditu.amap.com/search?query=Shanghai+Museum"


def test_resolve_place_links_marks_query_url_as_search_fallback():
    links = resolve_place_links({"title": "Shanghai Museum"})

    assert links["canonical_url"] is None
    assert links["official_url"] is None
    assert links["link_confidence"] == "search_fallback"
    assert links["search_url"] == "https://ditu.amap.com/search?query=Shanghai+Museum"


def test_resolve_place_links_builds_google_maps_url_from_serpapi_place_id():
    links = resolve_place_links(
        {
            "title": "Shanghai Museum",
            "place_id": "ChIJPWUSbWlwsjURbNvIw3tOTE0",
            "website": "http://www.shanghaimuseum.net/",
        }
    )

    assert links["canonical_url"] == (
        "https://www.google.com/maps/search/?api=1&query=Shanghai+Museum"
        "&query_place_id=ChIJPWUSbWlwsjURbNvIw3tOTE0"
    )
    assert links["official_url"] == "http://www.shanghaimuseum.net/"
    assert links["link_confidence"] == "provider_result"
    assert links["provider"] == "google"
    assert links["provider_place_id"] == "ChIJPWUSbWlwsjURbNvIw3tOTE0"
