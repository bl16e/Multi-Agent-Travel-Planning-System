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


class LooseItineraryChain:
    def __init__(self, output_model):
        self.output_model = output_model

    async def ainvoke(self, variables):
        return self.output_model.model_validate(
            {
                "destination": "\u4e0a\u6d77",
                "overview": "Structured Shanghai itinerary.",
                "trip_style": "structured",
                "planning_notes": "Use named attractions and verify bookings.",
                "itinerary_draft": [
                    {
                        "date": "2026-10-24",
                        "activities": "\u4e0a\u5348\uff1a\u53c2\u89c2\u4e0a\u6d77\u535a\u7269\u9986\u3002\u4e0b\u5348\uff1a\u6f2b\u6b65\u5916\u6ee9\u3002",
                        "transport": "\u5730\u94c1\u548c\u6b65\u884c",
                        "booking_link": "https://www.shanghaimuseum.net/",
                    }
                ],
            }
        )


class LooseItineraryPrompt:
    def __or__(self, output_model):
        return LooseItineraryChain(output_model)


class LooseItineraryPromptTemplate:
    @staticmethod
    def from_messages(messages):
        return LooseItineraryPrompt()


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
    assert result.overview.startswith("\u57fa\u4e8e\u5b9e\u65f6\u5730\u70b9\u68c0\u7d22\u7ed3\u679c")
    assert "Senso-ji Temple" in serialized
    assert "Tokyo National Museum" in serialized
    assert "Tokyo food route with named local checkpoints" not in serialized
    assert "fallback" not in serialized.lower()


@pytest.mark.asyncio
async def test_structured_synthesis_uses_amap_poi_research_context_when_llm_times_out(monkeypatch):
    monkeypatch.setattr(agent_runtime, "build_qwen_chat", lambda: FakeLLM())
    monkeypatch.setattr(agent_runtime, "ChatPromptTemplate", FakePromptTemplate)

    research_context = json.dumps(
        [
            {
                "tool": "maps_text_search",
                "status": "ok",
                "result": json.dumps({
                    "pois": [
                        {
                            "name": "\u4e0a\u6d77\u535a\u7269\u9986",
                            "type": "\u79d1\u6559\u6587\u5316\u670d\u52a1;\u535a\u7269\u9986",
                            "address": "\u4e0a\u6d77\u5e02\u9ec4\u6d66\u533a\u4eba\u6c11\u5927\u9053201\u53f7",
                            "location": "121.475379,31.228017",
                            "website": "https://www.shanghaimuseum.net/",
                        },
                        {
                            "name": "\u5916\u6ee9",
                            "type": "\u98ce\u666f\u540d\u80dc;\u98ce\u666f\u540d\u80dc",
                            "address": "\u4e0a\u6d77\u5e02\u9ec4\u6d66\u533a\u4e2d\u5c71\u4e1c\u4e00\u8def",
                            "location": "121.490317,31.240638",
                        },
                        {
                            "name": "\u4e0a\u6d77\u6587\u5316\u5e7f\u573a",
                            "typecode": "110105",
                            "address": "\u8302\u540d\u5357\u8def178\u53f7",
                        },
                    ]
                }),
            }
        ]
    )

    result = await agent_runtime.run_structured_synthesis(
        soul_path="provinces/zhongshu_itinerary/SOUL.md",
        output_model=ItineraryDraftModel,
        user_prompt="ignored",
        variables={
            "destination": "\u4e0a\u6d77",
            "start_date": "2026-10-24",
            "end_date": "2026-10-25",
            "interests": "culture, food",
            "research_context": research_context,
        },
    )

    serialized = result.model_dump_json()
    assert result.destination == "\u4e0a\u6d77"
    assert "\u4e0a\u6d77\u535a\u7269\u9986" in serialized
    assert "\u5916\u6ee9" in serialized
    assert "\u4e0a\u6d77 culture route with named local checkpoints" not in serialized
    assert "Amap POI result" not in serialized
    assert "110105" not in serialized
    assert "fallback" not in serialized.lower()


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
async def test_structured_synthesis_normalizes_loose_qwen_itinerary_output(monkeypatch):
    monkeypatch.setattr(agent_runtime, "build_qwen_chat", lambda: SchemaReturningLLM())
    monkeypatch.setattr(agent_runtime, "ChatPromptTemplate", LooseItineraryPromptTemplate)

    result = await agent_runtime.run_structured_synthesis(
        soul_path="provinces/zhongshu_itinerary/SOUL.md",
        output_model=ItineraryDraftModel,
        user_prompt="ignored",
        variables={
            "destination": "\u4e0a\u6d77",
            "start_date": "2026-10-24",
            "end_date": "2026-10-24",
            "interests": "culture, food",
        },
    )

    assert result.destination == "\u4e0a\u6d77"
    assert result.daily_plan[0].day_index == 1
    assert result.daily_plan[0].city == "\u4e0a\u6d77"
    assert result.daily_plan[0].activities
    assert "\u4e0a\u6d77\u535a\u7269\u9986" in result.daily_plan[0].activities[0].title
    assert isinstance(result.planning_notes, list)


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

