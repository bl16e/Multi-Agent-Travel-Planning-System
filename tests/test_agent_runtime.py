import pytest
import json
from types import SimpleNamespace

import utils.agent_runtime as agent_runtime
from utils.schemas import ItineraryDraftModel
from utils.settings import get_settings


class FailingChain:
    async def ainvoke(self, variables):
        raise RuntimeError("live provider unavailable")


class FakePrompt:
    def __or__(self, other):
        return FailingChain()


class FakePromptTemplate:
    @staticmethod
    def from_messages(messages):
        return FakePrompt()


class FakeLLM:
    def with_structured_output(self, output_model):
        return object()


class SchemaReturningLLM:
    def with_structured_output(self, output_model):
        return output_model


class FailingStructuredLLM:
    def with_structured_output(self, output_model):
        raise RuntimeError("structured output unavailable")


class WrappedItineraryChain:
    def __init__(self, output_model):
        self.output_model = output_model

    async def ainvoke(self, variables):
        return self.output_model.model_validate(
            {
                "itinerary_draft": {
                    "destination": "Tokyo",
                    "overview": "Wrapped live itinerary.",
                    "trip_style": "structured",
                    "daily_plan": [
                        {
                            "day_index": 1,
                            "date": "2026-05-01",
                            "city": "Tokyo",
                            "theme": "Culture",
                            "summary": "Visit named places.",
                            "activities": [
                                {
                                    "start_time": "09:00",
                                    "end_time": "10:30",
                                    "title": "Senso-ji Temple",
                                    "location_name": "Senso-ji Temple, Asakusa",
                                    "description": "Visit the temple.",
                                }
                            ],
                        }
                    ],
                }
            }
        )


class WrappedItineraryPrompt:
    def __or__(self, output_model):
        return WrappedItineraryChain(output_model)


class WrappedItineraryPromptTemplate:
    @staticmethod
    def from_messages(messages):
        return WrappedItineraryPrompt()


class UnsupportedOutputModel:
    pass


@pytest.mark.asyncio
async def test_direct_mcp_tool_calls_return_requested_tool_results(monkeypatch):
    calls = []

    class FakeTool:
        def __init__(self, name):
            self.name = name

        async def ainvoke(self, args):
            calls.append((self.name, args))
            return {"tool": self.name, "args": args, "result": "ok"}

    async def fake_load_mcp_tools(server_names):
        return [
            FakeTool("search_google_flights"),
            FakeTool("search_google_maps"),
            FakeTool("search_local_places"),
        ]

    monkeypatch.setattr(agent_runtime, "load_mcp_tools", fake_load_mcp_tools)

    result = await agent_runtime.run_direct_mcp_tool_calls(
        server_names=["serpapi"],
        tool_calls=[
            {"tool": "search_local_places", "args": {"query": "Senso-ji Tokyo", "location": "Tokyo"}},
            {"tool": "search_google_maps", "args": {"query": "Tokyo National Museum"}},
        ],
    )

    assert calls == [
        ("search_local_places", {"query": "Senso-ji Tokyo", "location": "Tokyo"}),
        ("search_google_maps", {"query": "Tokyo National Museum"}),
    ]
    assert "search_local_places" in result
    assert "search_google_maps" in result
    assert "search_google_flights" not in result


@pytest.mark.asyncio
async def test_direct_mcp_tool_calls_compact_large_serpapi_results(monkeypatch):
    class FakeTool:
        name = "search_local_places"

        async def ainvoke(self, args):
            return {
                "local_results": [
                    {
                        "title": "Sensō-ji",
                        "type": "Buddhist temple",
                        "address": "2 Chome-3-1 Asakusa",
                        "rating": 4.6,
                        "reviews": 96000,
                        "description": "must visit" * 1000,
                        "thumbnail": "https://example.test/image.jpg",
                        "place_id_search": "https://serpapi.test/search.json?engine=google_maps&place_id=very-long-id",
                        "gps_coordinates": {"latitude": 35.714765, "longitude": 139.796655},
                    }
                ],
                "raw_html_file": "x" * 5000,
            }

    async def fake_load_mcp_tools(server_names):
        return [FakeTool()]

    monkeypatch.setattr(agent_runtime, "load_mcp_tools", fake_load_mcp_tools)

    result = await agent_runtime.run_direct_mcp_tool_calls(
        server_names=["serpapi"],
        tool_calls=[{"tool": "search_local_places", "args": {"query": "Senso-ji Tokyo"}}],
    )

    assert "Sensō-ji" in result
    assert "2 Chome-3-1 Asakusa" in result
    assert "must visitmust visit" not in result
    assert "raw_html_file" not in result
    assert "place_id_search" not in result
    assert len(result) < 2000


@pytest.mark.asyncio
async def test_react_mcp_task_filters_tools_by_allowed_names(monkeypatch):
    captured = {}
    fake_tools = [
        SimpleNamespace(name="search_google_flights"),
        SimpleNamespace(name="search_google_hotels"),
        SimpleNamespace(name="search_google_maps"),
        SimpleNamespace(name="search_local_places"),
    ]

    class FakeAgent:
        async def ainvoke(self, payload):
            return {"messages": [SimpleNamespace(content="research complete")]}

    def fake_create_react_agent(*, model, tools, prompt):
        captured["tool_names"] = [tool.name for tool in tools]
        return FakeAgent()

    async def fake_load_mcp_tools(server_names):
        return fake_tools

    monkeypatch.setattr(agent_runtime, "build_qwen_chat", lambda: object())
    monkeypatch.setattr(agent_runtime, "load_mcp_tools", fake_load_mcp_tools)
    monkeypatch.setattr(agent_runtime, "create_react_agent", fake_create_react_agent)

    result = await agent_runtime.run_react_mcp_task(
        soul_path="provinces/zhongshu_itinerary/SOUL.md",
        server_names=["serpapi"],
        user_task="Find Tokyo places.",
        allowed_tool_names=["search_google_maps", "search_local_places"],
    )

    assert result == "research complete"
    assert captured["tool_names"] == ["search_google_maps", "search_local_places"]


@pytest.mark.asyncio
async def test_react_mcp_task_uses_configured_invoke_timeout_by_default(monkeypatch):
    captured_timeouts = []

    class FakeAgent:
        async def ainvoke(self, payload):
            return {"messages": [SimpleNamespace(content="research complete")]}

    def fake_create_react_agent(*, model, tools, prompt):
        return FakeAgent()

    async def fake_load_mcp_tools(server_names):
        return [SimpleNamespace(name="search_google_maps")]

    async def fake_wait_for(awaitable, timeout):
        captured_timeouts.append(timeout)
        return await awaitable

    monkeypatch.setenv("QWEN_TIMEOUT_SECONDS", "33")
    get_settings.cache_clear()
    monkeypatch.setattr(agent_runtime, "build_qwen_chat", lambda: object())
    monkeypatch.setattr(agent_runtime, "load_mcp_tools", fake_load_mcp_tools)
    monkeypatch.setattr(agent_runtime, "create_react_agent", fake_create_react_agent)
    monkeypatch.setattr(agent_runtime.asyncio, "wait_for", fake_wait_for)

    result = await agent_runtime.run_react_mcp_task(
        soul_path="provinces/zhongshu_itinerary/SOUL.md",
        server_names=["serpapi"],
        user_task="Find Tokyo places.",
    )

    assert result == "research complete"
    assert captured_timeouts[-1] == 33


@pytest.mark.asyncio
async def test_structured_synthesis_falls_back_when_live_llm_invocation_fails(monkeypatch):
    monkeypatch.setattr(agent_runtime, "build_qwen_chat", lambda: FakeLLM())
    monkeypatch.setattr(agent_runtime, "ChatPromptTemplate", FakePromptTemplate)

    result = await agent_runtime.run_structured_synthesis(
        soul_path="provinces/zhongshu_itinerary/SOUL.md",
        output_model=ItineraryDraftModel,
        user_prompt="ignored",
        variables={
            "destination": "Tokyo",
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "interests": "food, culture",
            "research_context": "live provider unavailable",
        },
    )

    assert result.destination == "Tokyo"
    assert len(result.daily_plan) == 2
    assert "Did not use real-time data" in result.planning_notes[0]


@pytest.mark.asyncio
async def test_structured_synthesis_uses_configured_timeout_by_default(monkeypatch):
    captured = {}

    async def fake_wait_for(awaitable, timeout):
        captured["timeout"] = timeout
        return await awaitable

    monkeypatch.setenv("QWEN_TIMEOUT_SECONDS", "34")
    get_settings.cache_clear()
    monkeypatch.setattr(agent_runtime, "build_qwen_chat", lambda: SchemaReturningLLM())
    monkeypatch.setattr(agent_runtime, "ChatPromptTemplate", WrappedItineraryPromptTemplate)
    monkeypatch.setattr(agent_runtime.asyncio, "wait_for", fake_wait_for)

    result = await agent_runtime.run_structured_synthesis(
        soul_path="provinces/zhongshu_itinerary/SOUL.md",
        output_model=ItineraryDraftModel,
        user_prompt="ignored",
        variables={"destination": "Tokyo", "start_date": "2026-05-01", "end_date": "2026-05-01"},
    )

    assert result.destination == "Tokyo"
    assert captured["timeout"] == 34


@pytest.mark.asyncio
async def test_structured_synthesis_uses_live_research_context_when_llm_times_out(monkeypatch):
    monkeypatch.setattr(agent_runtime, "build_qwen_chat", lambda: FakeLLM())
    monkeypatch.setattr(agent_runtime, "ChatPromptTemplate", FakePromptTemplate)

    research_context = json.dumps(
        [
            {
                "tool": "search_local_places",
                "status": "ok",
                "result": {
                    "local_results": [
                        {
                            "title": "Senso-ji Temple",
                            "type": "Buddhist temple",
                            "address": "2 Chome-3-1 Asakusa, Tokyo",
                            "rating": 4.6,
                            "website": "https://www.senso-ji.jp/",
                            "link": "https://www.google.com/maps/place/Senso-ji",
                        },
                        {
                            "title": "Tokyo National Museum",
                            "type": "Museum",
                            "address": "13-9 Uenokoen, Tokyo",
                            "rating": 4.5,
                            "website": "https://www.tnm.jp/",
                        },
                    ]
                },
            }
        ]
    )

    result = await agent_runtime.run_structured_synthesis(
        soul_path="provinces/zhongshu_itinerary/SOUL.md",
        output_model=ItineraryDraftModel,
        user_prompt="ignored",
        variables={
            "destination": "Tokyo",
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "interests": "food, culture",
            "research_context": research_context,
        },
    )

    serialized = result.model_dump_json()
    assert result.destination == "Tokyo"
    assert result.overview.startswith("Live MCP research was used")
    assert "Senso-ji Temple" in serialized
    assert "Tokyo National Museum" in serialized
    assert "Tokyo food route with named local checkpoints" not in serialized
    assert any("structured_llm_timeout" in note for note in result.planning_notes)


@pytest.mark.asyncio
async def test_structured_synthesis_accepts_wrapped_itinerary_output(monkeypatch):
    monkeypatch.setattr(agent_runtime, "build_qwen_chat", lambda: SchemaReturningLLM())
    monkeypatch.setattr(agent_runtime, "ChatPromptTemplate", WrappedItineraryPromptTemplate)

    result = await agent_runtime.run_structured_synthesis(
        soul_path="provinces/zhongshu_itinerary/SOUL.md",
        output_model=ItineraryDraftModel,
        user_prompt="ignored",
        variables={"destination": "Tokyo", "start_date": "2026-05-01", "end_date": "2026-05-01"},
    )

    assert result.overview == "Wrapped live itinerary."
    assert result.daily_plan[0].activities[0].title == "Senso-ji Temple"


@pytest.mark.asyncio
async def test_structured_synthesis_falls_back_when_live_structured_setup_fails(monkeypatch):
    monkeypatch.setattr(agent_runtime, "build_qwen_chat", lambda: FailingStructuredLLM())

    result = await agent_runtime.run_structured_synthesis(
        soul_path="provinces/zhongshu_itinerary/SOUL.md",
        output_model=ItineraryDraftModel,
        user_prompt="ignored",
        variables={
            "destination": "Kyoto",
            "start_date": "2026-05-01",
            "end_date": "2026-05-01",
            "interests": "temples",
            "research_context": "structured output unavailable",
        },
    )

    assert result.destination == "Kyoto"
    assert len(result.daily_plan) == 1


@pytest.mark.asyncio
async def test_structured_synthesis_preserves_original_exception_cause_when_no_fallback(monkeypatch):
    monkeypatch.setattr(agent_runtime, "build_qwen_chat", lambda: FakeLLM())
    monkeypatch.setattr(agent_runtime, "ChatPromptTemplate", FakePromptTemplate)

    with pytest.raises(RuntimeError) as excinfo:
        await agent_runtime.run_structured_synthesis(
            soul_path="provinces/zhongshu_itinerary/SOUL.md",
            output_model=UnsupportedOutputModel,
            user_prompt="ignored",
            variables={"destination": "Tokyo"},
        )

    assert "Structured synthesis failed" in str(excinfo.value)
    assert excinfo.value.__cause__ is not None
    assert str(excinfo.value.__cause__) == "live provider unavailable"
