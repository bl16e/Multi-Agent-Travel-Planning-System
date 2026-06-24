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
                            "summary": "Arrival transfer followed by Senso-ji and Asakusa visit",
                            "activities": [
                                {
                                    "start_time": "09:00",
                                    "end_time": "11:00",
                                    "title": "Senso-ji temple visit",
                                    "location_name": "Senso-ji, Asakusa",
                                    "description": "Visit the named temple complex and Nakamise-dori with opening-hour verification pending.",
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
                            "summary": "Arrival transfer followed by Senso-ji and Asakusa visit",
                            "activities": [
                                {
                                    "start_time": "09:00",
                                    "end_time": "11:00",
                                    "title": "Senso-ji temple visit",
                                    "location_name": "Senso-ji, Asakusa",
                                    "description": "Visit the named temple complex and Nakamise-dori with opening-hour verification pending.",
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


@pytest.mark.asyncio
async def test_verdict_rejects_generic_placeholder_itinerary_items():
    agent = MenxiaReviewAgent()
    result = await agent.verdict(
        {
            "request_id": "placeholder_reject_001",
            "parsed_draft": {
                "request_id": "placeholder_reject_001",
                "destination": "Tokyo",
                "itinerary_draft": {
                    "destination": "Tokyo",
                    "overview": "Offline fallback itinerary generated because live LLM/MCP synthesis was unavailable.",
                    "trip_style": "balanced",
                    "daily_plan": [
                        {
                            "day_index": 1,
                            "date": "2026-05-01",
                            "city": "Tokyo",
                            "theme": "Arrival",
                            "summary": "Estimated offline itinerary block; not validated with real-time availability.",
                            "activities": [
                                {
                                    "start_time": "09:00",
                                    "end_time": "11:30",
                                    "title": "Tokyo orientation walk",
                                    "location_name": "Central Tokyo",
                                    "description": "Fallback activity generated without live search data.",
                                },
                                {
                                    "start_time": "14:00",
                                    "end_time": "16:30",
                                    "title": "Culture focused visit",
                                    "location_name": "Tokyo",
                                    "description": "Estimated attraction slot; confirm opening hours before booking.",
                                },
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
    assert verdict["verdict"] == "REJECTED"
    assert verdict["approved_bureaus"] == []
    assert any("placeholder" in issue.lower() for issue in verdict["blocking_issues"])
    assert verdict["data_source"] == "fallback_estimate"
    assert any("generic" in warning.lower() for warning in verdict["warnings"])


@pytest.mark.asyncio
async def test_verdict_approves_live_mcp_research_draft_for_liubu_completion(monkeypatch):
    called = {"live_review": False}

    async def fail_if_live_review_is_called(**kwargs):
        called["live_review"] = True
        raise AssertionError("Live MCP fallback drafts should be evaluated by deterministic dispatchability rules.")

    monkeypatch.setattr("provinces.menxia_review.graph.run_structured_synthesis", fail_if_live_review_is_called)
    agent = MenxiaReviewAgent()

    result = await agent.verdict(
        {
            "request_id": "live_mcp_dispatchable",
            "parsed_draft": {
                "request_id": "live_mcp_dispatchable",
                "destination": "Tokyo",
                "itinerary_draft": {
                    "destination": "Tokyo",
                    "overview": "Live MCP research was used to build this draft; structured LLM synthesis was unavailable, so the plan remains pending review.",
                    "trip_style": "live_research_fallback",
                    "planning_notes": [
                        "data_source=live_mcp_research; synthesis=fallback_from_compact_tool_results.",
                        "Review all opening hours, reservation requirements, and transport durations before final approval.",
                    ],
                    "pending_confirmations": [
                        "Confirm official opening hours and ticket availability for each listed place.",
                        "Confirm transit duration and routing between daily activities.",
                    ],
                    "risk_flags": ["Live place search does not guarantee booking availability."],
                    "daily_plan": [
                        {
                            "day_index": 1,
                            "date": "2026-05-01",
                            "city": "Tokyo",
                            "theme": "Tokyo culture",
                            "summary": "Uses live MCP place results for Tokyo; downstream bureaus must verify hours, bookings, weather, and transport.",
                            "activities": [
                                {
                                    "start_time": "09:00",
                                    "end_time": "11:00",
                                    "title": "Senso-ji Temple",
                                    "location_name": "2 Chome-3-1 Asakusa, Tokyo",
                                    "description": "Live MCP place result for Buddhist temple. Verify opening hours and ticket availability before booking.",
                                    "map_link": "https://www.google.com/maps/search/?api=1&query=Senso-ji+Temple",
                                    "status": "pending",
                                    "transport": {
                                        "from_location": "hotel",
                                        "to_location": "2 Chome-3-1 Asakusa, Tokyo",
                                        "mode": "transit",
                                        "duration_text": "Confirm live transit duration before final approval.",
                                        "status": "pending",
                                    },
                                }
                            ],
                        }
                    ],
                },
                "required_bureaus": ["WEATHER", "BUDGET", "ACCOMMODATION", "FLIGHT_TRANSPORT", "CALENDAR"],
                "bureau_tasks": [],
                "governance": {"producer": "ZHONGSHU", "revision_round": 0},
            },
            "user_request": {"profile": {"total_budget": 1800, "currency": "USD"}},
        }
    )

    verdict = result["verdict_payload"]
    assert verdict["verdict"] == "APPROVED"
    assert verdict["approved_bureaus"] == ["WEATHER", "BUDGET", "ACCOMMODATION", "FLIGHT_TRANSPORT", "CALENDAR"]
    assert called["live_review"] is False
    assert any("Liubu" in note for note in verdict["review_notes"])
    assert any("pending" in warning.lower() for warning in verdict["warnings"])
