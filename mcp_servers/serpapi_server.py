from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import httpx
from fastmcp import FastMCP

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.settings import get_settings

mcp = FastMCP("SerpApi Travel Search MCP")
settings = get_settings()


async def _search(params: dict[str, str]) -> dict:
    if not settings.serpapi_api_key:
        raise ValueError("SERPAPI_API_KEY is not configured.")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get("https://serpapi.com/search", params=params)
        if response.status_code >= 400:
            error = response.text
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    error = str(payload.get("error") or payload.get("message") or error)
            except ValueError:
                pass
            return {
                "status": "error",
                "http_status": response.status_code,
                "error": error.replace(str(settings.serpapi_api_key), "<redacted>"),
            }
        return response.json()


async def _bocha_web_search(query: str, *, freshness: str = "noLimit", count: int = 10) -> dict:
    if not settings.bocha_api_key:
        raise ValueError("BOCHA_API_KEY is not configured.")

    endpoint = "https://api.bochaai.com/v1/web-search?utm_source=bocha-mcp-local"
    payload = {
        "query": query,
        "summary": True,
        "freshness": freshness,
        "count": min(max(count, 1), 50),
    }
    headers = {
        "Authorization": f"Bearer {settings.bocha_api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            error = exc.response.text.replace(str(settings.bocha_api_key), "<redacted>")
            return {
                "status": "error",
                "http_status": exc.response.status_code,
                "error": error,
                "search_metadata": {"provider": "bocha"},
            }
        except httpx.RequestError as exc:
            return {
                "status": "error",
                "error": f"Error communicating with Bocha Web Search API: {exc}",
                "search_metadata": {"provider": "bocha"},
            }

    raw = response.json()
    values = (((raw.get("data") or {}).get("webPages") or {}).get("value") or [])
    organic_results = []
    for index, item in enumerate(values[: payload["count"]]):
        if not isinstance(item, dict):
            continue
        organic_results.append(
            {
                "position": index + 1,
                "title": item.get("name") or "",
                "link": item.get("url") or "",
                "snippet": item.get("summary") or "",
                "date": item.get("datePublished") or "",
                "source": item.get("siteName") or "",
                "thumbnail": item.get("siteIcon") or item.get("imageUrl") or "",
            }
        )

    return {
        "search_metadata": {
            "provider": "bocha",
            "status": "ok",
        },
        "search_parameters": payload,
        "organic_results": organic_results,
    }


@mcp.tool()
async def search_google_flights(
    departure_id: str,
    arrival_id: str,
    outbound_date: str,
    return_date: str = "",
    adults: int = 1,
    currency: str = "USD",
) -> dict:
    """Search Google Flights results through SerpApi."""
    params = {
        "engine": "google_flights",
        "departure_id": departure_id,
        "arrival_id": arrival_id,
        "outbound_date": outbound_date,
        "adults": str(adults),
        "currency": currency,
        "api_key": settings.serpapi_api_key,
        "hl": "en",
        "gl": "us",
        "type": "1" if return_date else "2",
    }
    if return_date:
        params["return_date"] = return_date
    return await _search(params)


@mcp.tool()
async def search_google_hotels(
    query: str,
    check_in_date: str,
    check_out_date: str,
    adults: int = 2,
    currency: str = "USD",
) -> dict:
    """Search Google Hotels results through SerpApi."""
    return await _search(
        {
            "engine": "google_hotels",
            "q": query,
            "check_in_date": check_in_date,
            "check_out_date": check_out_date,
            "adults": str(adults),
            "currency": currency,
            "api_key": settings.serpapi_api_key,
            "hl": "en",
            "gl": "us",
        }
    )


@mcp.tool()
async def search_google_maps(query: str, ll: str = "") -> dict:
    """Search Google Maps results for attractions, stations, neighborhoods, and POIs."""
    params = {
        "engine": "google_maps",
        "q": query,
        "api_key": settings.serpapi_api_key,
        "hl": "en",
        "gl": "us",
    }
    if ll:
        params["ll"] = ll
    return await _search(params)


@mcp.tool()
async def search_google_maps_directions(origin: str, destination: str) -> dict:
    """Search Google Maps directions for transfer planning."""
    return await _search(
        {
            "engine": "google_maps_directions",
            "start_addr": origin,
            "end_addr": destination,
            "api_key": settings.serpapi_api_key,
            "hl": "en",
            "gl": "us",
        }
    )


@mcp.tool()
async def search_google_travel(
    departure_id: str,
    arrival_id: str = "",
    currency: str = "USD",
) -> dict:
    """Search Google Travel Explore destinations and fares through SerpApi."""
    params = {
        "engine": "google_travel_explore",
        "departure_id": departure_id,
        "currency": currency,
        "api_key": settings.serpapi_api_key,
        "hl": "en",
        "gl": "us",
    }
    if arrival_id:
        params["arrival_id"] = arrival_id
    return await _search(params)


@mcp.tool()
async def search_local_places(query: str, location: str = "") -> dict:
    """Search local business listings useful for food, attractions, and practical stops."""
    params = {
        "engine": "google_local",
        "q": query,
        "api_key": settings.serpapi_api_key,
        "hl": "en",
        "gl": "us",
    }
    if location:
        params["location"] = location
    return await _search(params)


@mcp.tool()
async def search_google_web(query: str, num: int = 10) -> dict:
    """Search Bocha web results; kept under the historical tool name for compatibility.

    Use this to discover recommended attractions,
    top sights, and travel guides for a destination BEFORE pinpointing them on
    a map.

    For example, search \"上海必去景点\" or \"best temples in Kyoto\" to find
    curated lists of real places, then use search_google_maps or maps_text_search
    to get exact addresses and links for each place.
    """
    return await _bocha_web_search(query, count=num)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "http", "sse", "streamable-http", "streamable_http"], default="stdio")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--path", default=None)
    args = parser.parse_args(argv)

    transport = args.transport.replace("_", "-")
    if transport == "stdio":
        mcp.run(transport="stdio")
        return

    run_kwargs = {
        "transport": transport,
        "host": args.host,
        "port": args.port,
        "path": args.path,
    }
    mcp.run(**{key: value for key, value in run_kwargs.items() if value is not None})


if __name__ == "__main__":
    main()
