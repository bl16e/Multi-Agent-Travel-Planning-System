import asyncio
import logging

import pytest
from langchain_core.messages import AIMessage

import main
import provinces.liubu.official_tooling as official_tooling
import provinces.liubu.weather.service as weather_service
import utils.mcp_client as mcp_client
import workflow as workflow_module
from provinces.liubu.weather.service import WeatherBureau
from tests.test_bureaus import _liubu_payload, _liubu_subtask
from utils.schemas import PlanningRequest, TravelerProfile
from utils.settings import get_settings


def make_request(request_id: str = "timeout_trip") -> PlanningRequest:
    return PlanningRequest(
        request_id=request_id,
        user_message="Plan Tokyo",
        profile=TravelerProfile(
            origin_city="Beijing",
            destination_preferences=["Tokyo"],
            start_date="2026-05-01",
            end_date="2026-05-03",
            total_budget=2500,
        ),
    )


@pytest.mark.asyncio
async def test_mcp_tool_loading_times_out_and_logs_context(monkeypatch, caplog):
    class HangingClient:
        async def get_tools(self):
            await asyncio.sleep(10)
            return []

    import mcp_servers.server as mcp_server

    monkeypatch.setenv("MCP_TOOLING_TIMEOUT_SECONDS", "0.01")
    get_settings.cache_clear()
    mcp_client._CLIENT_CACHE.clear()
    mcp_client._TOOLS_CACHE.clear()
    monkeypatch.setattr(mcp_server, "build_mcp_client", lambda server_names: HangingClient())

    with caplog.at_level(logging.WARNING):
        tools = await asyncio.wait_for(mcp_client.load_mcp_tools(["serpapi"]), timeout=1.0)

    assert tools == []
    assert any("MCP tool loading timed out" in record.getMessage() for record in caplog.records)
    assert any("serpapi" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_toolnode_execution_times_out_and_returns_timeout_evidence(monkeypatch, caplog):
    async def hanging_toolnode_ainvoke(tool_node, state):
        await asyncio.sleep(10)
        return {"messages": []}

    monkeypatch.setenv("LIUBU_TOOL_TIMEOUT_SECONDS", "0.01")
    get_settings.cache_clear()
    monkeypatch.setattr(official_tooling.ToolNode, "ainvoke", hanging_toolnode_ainvoke)
    tool_node = official_tooling.EvidenceToolNode([])
    state = {
        "messages": [
            AIMessage(
                content="search",
                tool_calls=[{"name": "search_weather_context", "args": {"q": "Tokyo"}, "id": "tool-1"}],
            )
        ]
    }

    with caplog.at_level(logging.WARNING):
        result = await asyncio.wait_for(
            official_tooling.run_tool_node_collect_evidence(tool_node, state, bureau="WEATHER", request_id="tool_timeout"),
            timeout=1.0,
        )

    assert result["tool_evidence"][0]["status"] == "timeout"
    assert result["tool_evidence"][0]["tool_name"] == "search_weather_context"
    assert any("ToolNode execution timed out" in record.getMessage() for record in caplog.records)
    assert any("tool_timeout" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_bound_tool_model_timeout_falls_back_and_logs(monkeypatch, caplog):
    class HangingToolModel:
        async def ainvoke(self, messages):
            await asyncio.sleep(10)
            return AIMessage(content="never")

    monkeypatch.setenv("LIUBU_TOOL_TIMEOUT_SECONDS", "0.01")
    get_settings.cache_clear()
    monkeypatch.setattr(weather_service, "build_qwen_chat", lambda: None)
    bureau = WeatherBureau()
    bureau.bound_tool_model = HangingToolModel()
    bureau._tooling_ready = True
    bureau.graph = bureau._build_graph()

    with caplog.at_level(logging.WARNING):
        result = await asyncio.wait_for(
            bureau.run(_liubu_subtask(_liubu_payload(), "WEATHER")),
            timeout=1.0,
        )

    assert result["bureau"] == "WEATHER"
    assert result["data_source"] == "fallback_estimate"
    assert any("Bound tool model call timed out" in record.getMessage() for record in caplog.records)
    assert any("liubu_subtask" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_workflow_run_times_out_and_logs_request_context(monkeypatch, caplog):
    request = make_request("workflow_timeout")
    workflow = workflow_module.ProvinceWorkflow()

    class HangingGraph:
        async def ainvoke(self, graph_input, config=None):
            await asyncio.sleep(10)
            return {}

    async def fake_ensure():
        return HangingGraph()

    async def fake_close():
        return None

    monkeypatch.setenv("PLAN_REQUEST_TIMEOUT_SECONDS", "0.01")
    get_settings.cache_clear()
    monkeypatch.setattr(workflow, "_ensure_persistent_graph", fake_ensure)
    monkeypatch.setattr(workflow, "_close_checkpointer", fake_close)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(workflow.run(request), timeout=1.0)

    assert any("Workflow run timed out" in record.getMessage() for record in caplog.records)
    assert any("workflow_timeout" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_execute_plan_request_uses_logger_progress_reporter_by_default(monkeypatch):
    captured = {}
    request = make_request("default_logger")

    class FakePlanner:
        def __init__(self, progress_reporter):
            self.progress_reporter = progress_reporter
            self.sessions = {}

        async def plan_trip(self, request):
            captured["reporter_callable"] = callable(self.progress_reporter)
            self.progress_reporter("[00:00:00] [START] workflow | request_id=default_logger")
            return {"status": "HUMAN_INTERVENE", "request_id": request.request_id, "question": "Need input"}

    def fake_create_travel_system(progress_reporter=None):
        captured["progress_reporter"] = progress_reporter
        return FakePlanner(progress_reporter)

    monkeypatch.setattr(main, "create_travel_system", fake_create_travel_system)

    await main._execute_plan_request(request)

    assert captured["reporter_callable"] is True
    assert callable(captured["progress_reporter"])
