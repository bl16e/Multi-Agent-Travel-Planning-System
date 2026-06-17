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


@pytest.mark.asyncio
async def test_offline_verdict_rejects_empty_daily_plan():
    agent = MenxiaReviewAgent()
    result = await agent.verdict(
        {
            "request_id": "offline_reject_001",
            "parsed_draft": {
                "request_id": "offline_reject_001",
                "destination": "Tokyo",
                "itinerary_draft": {
                    "destination": "Tokyo",
                    "overview": "Missing detail",
                    "trip_style": "balanced",
                    "daily_plan": [],
                },
                "required_bureaus": ["WEATHER"],
                "bureau_tasks": [],
                "governance": {"producer": "ZHONGSHU", "revision_round": 0},
            },
            "user_request": {"profile": {"total_budget": 3000, "currency": "USD"}},
        }
    )

    verdict = result["verdict_payload"]
    assert verdict["verdict"] == "REJECTED"
    assert verdict["approved_bureaus"] == []


@pytest.mark.asyncio
async def test_offline_verdict_approves_only_requested_bureaus():
    agent = MenxiaReviewAgent()
    result = await agent.verdict(
        {
            "request_id": "offline_approve_001",
            "parsed_draft": {
                "request_id": "offline_approve_001",
                "destination": "Tokyo",
                "itinerary_draft": {
                    "destination": "Tokyo",
                    "overview": "Structured plan",
                    "trip_style": "balanced",
                    "daily_plan": [
                        {
                            "day_index": 1,
                            "date": "2026-05-01",
                            "city": "Tokyo",
                            "theme": "Arrival",
                            "summary": "Arrival and orientation",
                            "activities": [
                                {
                                    "start_time": "09:00",
                                    "end_time": "11:00",
                                    "title": "Orientation walk",
                                    "location_name": "Central Tokyo",
                                    "description": "Confirm transit and local geography.",
                                }
                            ],
                        }
                    ],
                },
                "required_bureaus": ["WEATHER", "BUDGET"],
                "bureau_tasks": [],
                "governance": {"producer": "ZHONGSHU", "revision_round": 0},
            },
            "user_request": {"profile": {"total_budget": 3000, "currency": "USD"}},
        }
    )

    verdict = result["verdict_payload"]
    assert verdict["verdict"] == "APPROVED"
    assert verdict["approved_bureaus"] == ["WEATHER", "BUDGET"]


@pytest.mark.asyncio
async def test_verdict_falls_back_when_live_review_raises_non_runtime_error(monkeypatch):
    async def fail_live_review(**kwargs):
        raise ValueError("schema mismatch")

    monkeypatch.setattr("provinces.menxia_review.graph.run_structured_synthesis", fail_live_review)
    agent = MenxiaReviewAgent()

    result = await agent.verdict(
        {
            "request_id": "offline_exception_001",
            "parsed_draft": {
                "request_id": "offline_exception_001",
                "destination": "Tokyo",
                "itinerary_draft": {
                    "destination": "Tokyo",
                    "overview": "Structured plan",
                    "trip_style": "balanced",
                    "daily_plan": [
                        {
                            "day_index": 1,
                            "date": "2026-05-01",
                            "city": "Tokyo",
                            "theme": "Arrival",
                            "summary": "Arrival and orientation",
                            "activities": [
                                {
                                    "start_time": "09:00",
                                    "end_time": "11:00",
                                    "title": "Orientation walk",
                                    "location_name": "Central Tokyo",
                                    "description": "Confirm transit and local geography.",
                                }
                            ],
                        }
                    ],
                },
                "required_bureaus": ["WEATHER"],
                "bureau_tasks": [],
                "governance": {"producer": "ZHONGSHU", "revision_round": 0},
            },
            "user_request": {"profile": {"total_budget": 3000, "currency": "USD"}},
        }
    )

    verdict = result["verdict_payload"]
    assert verdict["verdict"] == "APPROVED"
    assert verdict["approved_bureaus"] == ["WEATHER"]
