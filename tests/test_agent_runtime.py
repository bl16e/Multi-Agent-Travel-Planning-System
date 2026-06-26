import pytest
import json

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


def test_handwritten_mcp_agent_helpers_are_removed():
    assert not hasattr(agent_runtime, "run_direct_mcp_tool_calls")
    assert not hasattr(agent_runtime, "run_react_mcp_task")
    assert not hasattr(agent_runtime, "create_react_agent")

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

