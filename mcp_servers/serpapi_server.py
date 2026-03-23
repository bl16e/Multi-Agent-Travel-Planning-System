from __future__ import annotations

import sys
from pathlib import Path

import httpx
from fastmcp import FastMCP

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from utils.settings import get_settings

mcp = FastMCP("SerpApi Travel Search MCP")
settings = get_settings()


async def _search(params: dict[str, str]) -> dict:
    if not settings.serpapi_api_key:
        raise ValueError("SERPAPI_API_KEY is not configured.")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        return response.json()


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


if __name__ == "__main__":
    mcp.run(transport="stdio")
