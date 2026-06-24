import pytest
import utils.agent_runtime as agent_runtime
from provinces.zhongshu_itinerary.graph import ZhongshuItineraryAgent
from utils.schemas import ItineraryDraftModel


@pytest.mark.asyncio
async def test_ingest_request():
    agent = ZhongshuItineraryAgent()
    state = {
        "user_request": {
            "profile": {
                "destination_preferences": ["Tokyo"],
                "origin_city": "Beijing",
                "start_date": "2026-05-01",
                "end_date": "2026-05-05",
                "total_budget": 5000,
                "currency": "USD",
                "interests": ["culture", "food"]
            }
        },
        "governance": {"rejection_count": 0}
    }
    result = await agent.ingest_request(state)
    assert "normalized_request" in result
    assert result["normalized_request"]["destination"] == "Tokyo"


@pytest.mark.asyncio
async def test_ingest_request_rejects_missing_destination_preferences():
    agent = ZhongshuItineraryAgent()
    state = {
        "user_request": {
            "profile": {
                "destination_preferences": [],
                "origin_city": "Beijing",
                "start_date": "2026-05-01",
                "end_date": "2026-05-05",
                "total_budget": 5000,
            }
        },
        "governance": {"rejection_count": 0},
    }

    with pytest.raises(ValueError, match="destination preference"):
        await agent.ingest_request(state)


@pytest.mark.asyncio
async def test_draft_itinerary_offline_output_uses_request_destination(monkeypatch):
    async def fake_research(**kwargs):
        return agent_runtime.FALLBACK_MESSAGE

    monkeypatch.setattr(agent_runtime, "build_qwen_chat", lambda: None)
    monkeypatch.setattr("provinces.zhongshu_itinerary.graph.run_direct_mcp_tool_calls", fake_research)

    agent = ZhongshuItineraryAgent()
    result = await agent.draft_itinerary(
        {
            "request_id": "offline_kyoto",
            "normalized_request": {
                "destination": "Kyoto",
                "origin_city": "Beijing",
                "start_date": "2026-05-01",
                "end_date": "2026-05-03",
                "total_budget": 3000,
                "currency": "USD",
                "adults": 2,
                "interests": ["culture"],
                "constraints": [],
                "revision_round": 0,
                "rejection_reasons": [],
                "revision_requests": [],
            },
        }
    )

    draft = result["draft"]
    assert draft["destination"] == "Kyoto"
    assert draft["daily_plan"][0]["date"] == "2026-05-01"
    assert draft["daily_plan"][0]["city"] == "Kyoto"
    assert "Kyoto" in draft["daily_plan"][0]["activities"][0]["title"]
    assert "Tokyo" not in str(draft)


@pytest.mark.asyncio
async def test_draft_itinerary_uses_direct_place_tool_calls(monkeypatch):
    seen = {}

    async def fake_direct_research(**kwargs):
        seen["server_names"] = kwargs.get("server_names")
        seen["tool_calls"] = kwargs.get("tool_calls")
        return "Senso-ji Temple; Tokyo National Museum; Tsukiji Outer Market; Shinjuku Gyoen."

    async def fake_synthesis(**kwargs):
        return ItineraryDraftModel.model_validate(
            {
                "destination": "Tokyo",
                "overview": "Concrete Tokyo plan.",
                "trip_style": "structured",
                "daily_plan": [
                    {
                        "day_index": 1,
                        "date": "2026-05-01",
                        "city": "Tokyo",
                        "theme": "Culture",
                        "summary": "Named Tokyo places.",
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
        )

    monkeypatch.setattr("provinces.zhongshu_itinerary.graph.run_direct_mcp_tool_calls", fake_direct_research)
    monkeypatch.setattr("provinces.zhongshu_itinerary.graph.run_structured_synthesis", fake_synthesis)

    agent = ZhongshuItineraryAgent()
    await agent.draft_itinerary(
        {
            "request_id": "tool_limit",
            "normalized_request": {
                "destination": "Tokyo",
                "origin_city": "Beijing",
                "start_date": "2026-05-01",
                "end_date": "2026-05-01",
                "total_budget": 1800,
                "currency": "USD",
                "adults": 1,
                "children": 0,
                "interests": ["culture", "food"],
                "constraints": ["include official links"],
                "user_message": "Use named places and transport durations.",
                "revision_round": 0,
                "rejection_reasons": [],
                "revision_requests": [],
            },
        }
    )

    assert seen["server_names"] == ["serpapi"]
    assert {call["tool"] for call in seen["tool_calls"]} <= {"search_google_maps", "search_local_places"}
    assert any(call["tool"] == "search_local_places" for call in seen["tool_calls"])


@pytest.mark.asyncio
async def test_draft_itinerary_does_not_hardcode_structured_synthesis_timeout(monkeypatch):
    seen = {}

    async def fake_direct_research(**kwargs):
        return "Senso-ji Temple; Tokyo National Museum."

    async def fake_synthesis(**kwargs):
        seen["timeout_seconds"] = kwargs.get("timeout_seconds")
        return ItineraryDraftModel.model_validate(
            {
                "destination": "Tokyo",
                "overview": "Concrete Tokyo plan.",
                "trip_style": "structured",
                "daily_plan": [
                    {
                        "day_index": 1,
                        "date": "2026-05-01",
                        "city": "Tokyo",
                        "theme": "Culture",
                        "summary": "Named Tokyo places.",
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
        )

    monkeypatch.setattr("provinces.zhongshu_itinerary.graph.run_direct_mcp_tool_calls", fake_direct_research)
    monkeypatch.setattr("provinces.zhongshu_itinerary.graph.run_structured_synthesis", fake_synthesis)

    agent = ZhongshuItineraryAgent()
    await agent.draft_itinerary(
        {
            "request_id": "bounded_timeout",
            "normalized_request": {
                "destination": "Tokyo",
                "origin_city": "Beijing",
                "start_date": "2026-05-01",
                "end_date": "2026-05-01",
                "total_budget": 1800,
                "currency": "USD",
                "adults": 1,
                "children": 0,
                "interests": ["culture"],
                "constraints": [],
                "user_message": "Use named places.",
                "revision_round": 0,
                "rejection_reasons": [],
                "revision_requests": [],
            },
        }
    )

    assert seen["timeout_seconds"] is None


@pytest.mark.asyncio
async def test_draft_itinerary_passes_readable_chinese_prompt_to_synthesis(monkeypatch):
    seen = {}

    async def fake_direct_research(**kwargs):
        return "Senso-ji Temple; Tokyo National Museum."

    async def fake_synthesis(**kwargs):
        seen["user_prompt"] = kwargs.get("user_prompt")
        return ItineraryDraftModel.model_validate(
            {
                "destination": "Tokyo",
                "overview": "Concrete Tokyo plan.",
                "trip_style": "structured",
                "daily_plan": [
                    {
                        "day_index": 1,
                        "date": "2026-05-01",
                        "city": "Tokyo",
                        "theme": "Culture",
                        "summary": "Named Tokyo places.",
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
        )

    monkeypatch.setattr("provinces.zhongshu_itinerary.graph.run_direct_mcp_tool_calls", fake_direct_research)
    monkeypatch.setattr("provinces.zhongshu_itinerary.graph.run_structured_synthesis", fake_synthesis)

    agent = ZhongshuItineraryAgent()
    await agent.draft_itinerary(
        {
            "request_id": "readable_prompt",
            "normalized_request": {
                "destination": "Tokyo",
                "origin_city": "Beijing",
                "start_date": "2026-05-01",
                "end_date": "2026-05-01",
                "total_budget": 1800,
                "currency": "USD",
                "adults": 1,
                "children": 0,
                "interests": ["culture"],
                "constraints": [],
                "user_message": "Use named places.",
                "revision_round": 0,
                "rejection_reasons": [],
                "revision_requests": [],
            },
        }
    )

    assert "请根据用户需求生成详细的旅行行程草案" in seen["user_prompt"]
    assert "璇锋" not in seen["user_prompt"]


@pytest.mark.asyncio
async def test_draft_itinerary_with_llm():
    """测试完整的行程生成流程（包含LLM调用）"""
    agent = ZhongshuItineraryAgent()
    state = {
        "request_id": "test_llm_001",
        "normalized_request": {
            "destination": "Tokyo",
            "origin_city": "Beijing",
            "start_date": "2026-05-01",
            "end_date": "2026-05-03",
            "total_budget": 3000,
            "currency": "USD",
            "adults": 2,
            "interests": ["culture"],
            "constraints": [],
            "revision_round": 0,
            "rejection_reasons": [],
            "revision_requests": []
        }
    }
    result = await agent.draft_itinerary(state)
    assert "draft" in result
    draft = result["draft"]
    assert draft["destination"] == "Tokyo"
    assert "daily_plan" in draft
    assert len(draft["daily_plan"]) > 0
