import pytest
from langchain_core.tools import tool


@pytest.mark.asyncio
async def test_agent_tool_gate_filters_by_agent_and_category(monkeypatch):
    import utils.mcp_tool_registry as registry

    @tool
    async def search_google_maps(query: str = "") -> str:
        """Search Google Maps."""
        return query

    @tool
    async def search_google_flights(departure_id: str = "", arrival_id: str = "") -> str:
        """Search Google Flights."""
        return departure_id + arrival_id

    @tool
    async def maps_weather(city: str = "") -> str:
        """Search Amap weather."""
        return city

    async def fake_load_mcp_tools(server_names):
        assert server_names == ["serpapi"]
        return [search_google_maps, search_google_flights, maps_weather]

    monkeypatch.setattr(registry, "load_mcp_tools", fake_load_mcp_tools)

    tools = await registry.load_agent_tools(
        "ZHONGSHU",
        {"global_place_discovery", "domestic_weather"},
    )

    assert [item.name for item in tools] == ["search_google_maps"]


def test_agent_tool_gate_reports_required_servers():
    from utils.mcp_tool_registry import required_servers_for

    assert required_servers_for(
        "ZHONGSHU",
        {"global_place_discovery", "domestic_poi_confirmation"},
    ) == ["amap", "serpapi"]


@pytest.mark.asyncio
async def test_liubu_loader_uses_agent_gate_when_agent_is_provided(monkeypatch):
    import provinces.liubu.official_tooling as official_tooling

    calls = []

    async def fake_load_agent_tools_for_names(agent, allowed_names):
        calls.append((agent, allowed_names))
        return ["budget-flight-tool"]

    monkeypatch.setattr(official_tooling, "load_agent_tools_for_names", fake_load_agent_tools_for_names)

    tools = await official_tooling.load_allowed_liubu_tools(
        ["serpapi"],
        {"search_google_flights"},
        agent="BUDGET",
    )

    assert tools == ["budget-flight-tool"]
    assert calls == [("BUDGET", {"search_google_flights"})]
