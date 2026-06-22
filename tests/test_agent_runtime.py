import pytest

import utils.agent_runtime as agent_runtime
from utils.schemas import ItineraryDraftModel


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


class FailingStructuredLLM:
    def with_structured_output(self, output_model):
        raise RuntimeError("structured output unavailable")


class UnsupportedOutputModel:
    pass


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
