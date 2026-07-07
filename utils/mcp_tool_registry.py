from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from utils.mcp_client import load_mcp_tools


@dataclass(frozen=True)
class ToolCapability:
    name: str
    server: str
    category: str
    allowed_agents: frozenset[str]


TOOL_CAPABILITIES: tuple[ToolCapability, ...] = (
    ToolCapability("search_google_web", "serpapi", "web_search_discovery", frozenset({"ZHONGSHU"})),
    ToolCapability("search_google_maps", "serpapi", "global_place_discovery", frozenset({"ZHONGSHU", "CALENDAR"})),
    ToolCapability("search_local_places", "serpapi", "semantic_local_discovery", frozenset({"ZHONGSHU", "CALENDAR"})),
    ToolCapability("search_google_maps_directions", "serpapi", "route_context", frozenset({"CALENDAR"})),
    ToolCapability("search_google_travel", "serpapi", "travel_budget_context", frozenset({"BUDGET"})),
    ToolCapability("search_google_hotels", "serpapi", "hotel_budget_context", frozenset({"BUDGET"})),
    ToolCapability("search_google_flights", "serpapi", "flight_budget_context", frozenset({"BUDGET", "FLIGHT_TRANSPORT"})),
    ToolCapability("maps_text_search", "amap", "domestic_poi_confirmation", frozenset({"ZHONGSHU", "ACCOMMODATION"})),
    ToolCapability("maps_weather", "amap", "domestic_weather", frozenset({"WEATHER"})),
    ToolCapability("maps_geo", "amap", "domestic_geocoding", frozenset({"FLIGHT_TRANSPORT"})),
    # --- RollingGo hotel tools ---
    ToolCapability("searchHotels", "rollinggo", "hotel_search_booking", frozenset({"ACCOMMODATION", "ZHONGSHU"})),
    ToolCapability("getHotelDetail", "rollinggo", "hotel_detail_pricing", frozenset({"ACCOMMODATION"})),
    ToolCapability("getHotelSearchTags", "rollinggo", "hotel_search_metadata", frozenset({"ACCOMMODATION", "ZHONGSHU"})),
    # --- RollingGo flight tools ---
    ToolCapability("searchFlights", "rollinggo", "flight_search_booking", frozenset({"FLIGHT_TRANSPORT"})),
    ToolCapability("searchAirports", "rollinggo", "airport_code_lookup", frozenset({"FLIGHT_TRANSPORT"})),
    ToolCapability("checkFlightSeats", "rollinggo", "flight_seat_availability", frozenset({"FLIGHT_TRANSPORT"})),
    ToolCapability("checkBaggageAllowance", "rollinggo", "flight_baggage_info", frozenset({"FLIGHT_TRANSPORT"})),
)


def required_servers_for(agent: str, categories: set[str]) -> list[str]:
    names = {
        capability.server
        for capability in _matching_capabilities(agent, categories)
    }
    return sorted(names)


async def load_agent_tools(agent: str, categories: set[str]) -> list[Any]:
    servers = required_servers_for(agent, categories)
    if not servers:
        return []
    allowed_names = {capability.name for capability in _matching_capabilities(agent, categories)}
    tools = await load_mcp_tools(servers)
    return [tool for tool in tools if getattr(tool, "name", None) in allowed_names]


async def load_agent_tools_for_names(agent: str, allowed_names: set[str]) -> list[Any]:
    capabilities = [
        capability
        for capability in TOOL_CAPABILITIES
        if agent.upper() in capability.allowed_agents and capability.name in allowed_names
    ]
    servers = sorted({capability.server for capability in capabilities})
    if not servers:
        return []
    gated_names = {capability.name for capability in capabilities}
    tools = await load_mcp_tools(servers)
    return [tool for tool in tools if getattr(tool, "name", None) in gated_names]


def _matching_capabilities(agent: str, categories: set[str]) -> list[ToolCapability]:
    normalized_agent = agent.upper()
    return [
        capability
        for capability in TOOL_CAPABILITIES
        if normalized_agent in capability.allowed_agents and capability.category in categories
    ]
