import pytest
from provinces.zhongshu_itinerary.graph import ZhongshuItineraryAgent


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

