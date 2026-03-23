import pytest
from provinces.menxia_review.graph import MenxiaReviewAgent


@pytest.mark.asyncio
async def test_ingest_draft():
    agent = MenxiaReviewAgent()
    state = {
        "draft": {
            "request_id": "test_001",
            "destination": "Tokyo",
            "itinerary_draft": {
                "destination": "Tokyo",
                "overview": "Test itinerary",
                "trip_style": "balanced",
                "daily_plan": []
            },
            "required_bureaus": ["WEATHER"],
            "bureau_tasks": [],
            "governance": {
                "producer": "ZHONGSHU",
                "revision_round": 0
            }
        }
    }
    result = await agent.ingest_draft(state)
    assert "parsed_draft" in result


@pytest.mark.asyncio
async def test_verdict_with_llm():
    """测试门下省LLM审核"""
    agent = MenxiaReviewAgent()
    state = {
        "request_id": "test_verdict_001",
        "parsed_draft": {
            "request_id": "test_verdict_001",
            "destination": "Tokyo",
            "itinerary_draft": {
                "destination": "Tokyo",
                "overview": "3日东京文化之旅",
                "trip_style": "balanced",
                "daily_plan": [
                    {
                        "day_index": 1,
                        "date": "2026-05-01",
                        "theme": "抵达与适应",
                        "summary": "抵达东京",
                        "activities": []
                    }
                ]
            },
            "required_bureaus": ["WEATHER", "BUDGET"],
            "bureau_tasks": [],
            "governance": {
                "producer": "ZHONGSHU",
                "revision_round": 0
            }
        },
        "user_request": {
            "profile": {
                "total_budget": 3000,
                "currency": "USD"
            }
        }
    }
    result = await agent.verdict(state)
    assert "verdict_payload" in result
    verdict = result["verdict_payload"]
    assert verdict["verdict"] in ["APPROVED", "REJECTED", "HUMAN_INTERVENE"]

