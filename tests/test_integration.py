import pytest
import provinces.liubu.calendar.service as calendar_service
from provinces.liubu.calendar.service import CalendarBureau
from provinces.liubu.constrained.state import normalize_worker_input
from workflow import ProvinceWorkflow
from utils.schemas import PlanningRequest, TravelerProfile


@pytest.mark.asyncio
async def test_full_workflow_integration():
    """完整工作流集成测试（包含LLM调用）"""
    workflow = ProvinceWorkflow()
    destination = "\u4e0a\u6d77"
    request = PlanningRequest(
        request_id="test_integration_001",
        user_message="Plan a 3-day domestic Shanghai trip",
        profile=TravelerProfile(
            destination_preferences=[destination],
            origin_city="\u5317\u4eac",
            start_date="2026-05-01",
            end_date="2026-05-03",
            total_budget=3000,
            currency="USD",
            adults=2,
            interests=["culture"]
        )
    )
    result = await workflow.run(request)
    assert result["status"] in ["DONE", "HUMAN_INTERVENE", "REJECTED"]
    if result["status"] == "DONE":
        assert "final_package" in result
        package = result["final_package"]
        assert package["request_id"] == request.request_id
        assert package["destination"] == destination
        assert package["workflow_state"] == "DONE"
        assert package["itinerary"]["destination"] == destination
        assert package["review"]["request_id"] == request.request_id
        assert isinstance(package["booking_links"], list)
        assert isinstance(package["progress_events"], list)
    elif result["status"] == "HUMAN_INTERVENE":
        assert result["question"]
        assert result["context"].request_id == request.request_id
    else:
        payload = result["rejected_payload"]
        assert payload["status"] == "REJECTED"
        assert payload["request_id"] == request.request_id


@pytest.mark.asyncio
async def test_final_package_and_artifacts_expose_data_source_labels(tmp_path, monkeypatch):
    async def no_tools(server_names, allowed_names, *, agent=None):
        return []

    monkeypatch.setattr(calendar_service, "load_allowed_liubu_tools", no_tools)
    monkeypatch.setattr(calendar_service, "build_qwen_chat", lambda: None)

    request = PlanningRequest(
        request_id="artifact_labels",
        user_message="Plan Hangzhou",
        profile=TravelerProfile(
            destination_preferences=["\u676d\u5dde"],
            origin_city="Beijing",
            start_date="2026-05-01",
            end_date="2026-05-01",
            total_budget=3000,
            currency="USD",
        ),
    )
    draft_packet = {
        "request_id": request.request_id,
        "destination": "\u676d\u5dde",
        "itinerary_draft": {
            "destination": "\u676d\u5dde",
            "overview": "Named Hangzhou plan",
            "trip_style": "balanced",
            "daily_plan": [
                {
                    "day_index": 1,
                    "date": "2026-05-01",
                    "city": "\u676d\u5dde",
                    "theme": "Temples",
                    "summary": "Visit named temples.",
                    "activities": [
                        {
                            "start_time": "09:00",
                            "end_time": "10:30",
                            "title": "West Lake lakeside walk",
                            "location_name": "\u676d\u5dde\u897f\u6e56",
                            "description": "Visit the named West Lake scenic area.",
                            "estimated_cost": 20,
                        }
                    ],
                }
            ],
            "planning_notes": ["data_source=fallback_estimate"],
            "pending_confirmations": [],
            "risk_flags": [],
        },
        "required_bureaus": ["WEATHER", "BUDGET", "CALENDAR"],
        "bureau_tasks": [],
        "governance": {"producer": "ZHONGSHU", "revision_round": 0},
    }
    review_packet = {
        "request_id": request.request_id,
        "verdict": "APPROVED",
        "summary": "Offline review approved named content.",
        "blocking_issues": [],
        "revision_requests": [],
        "human_questions": [],
        "approved_bureaus": ["WEATHER", "BUDGET", "CALENDAR"],
        "governance": {
            "reviewer": "MENXIA",
            "source_producer": "ZHONGSHU",
            "next_hop": "SHANGSHU",
            "verdict_state": "APPROVED",
            "veto_enabled": True,
            "rejection_round": 0,
            "max_rejection_rounds": 2,
        },
        "review_notes": ["offline structural review"],
        "data_source": "fallback_estimate",
        "warnings": ["Did not use live review."],
    }
    calendar_result = await CalendarBureau(output_dir=tmp_path).run(
        {
            "worker_input": normalize_worker_input(
                {
                    "request_id": request.request_id,
                    "approved_draft": draft_packet,
                    "execution_plan": {"user_request": request.model_dump(mode="json")},
                },
                "CALENDAR",
            )
        }
    )
    execution_results = {
        "WEATHER": {
            "bureau": "WEATHER",
            "status": "fallback",
            "data_source": "fallback_estimate",
            "destination": "\u676d\u5dde",
            "forecast_days": [],
            "packing_list": [],
            "warnings": ["offline weather"],
            "summary": "Estimated weather",
        },
        "BUDGET": {
            "bureau": "BUDGET",
            "status": "fallback",
            "data_source": "fallback_estimate",
            "currency": "USD",
            "budget_breakdown": [],
            "total_estimated_cost": 0,
            "warnings": ["offline budget"],
        },
        "CALENDAR": calendar_result,
    }

    package = ProvinceWorkflow.build_final_package(
        request,
        {"dashboard_url": "http://testserver/dashboard/artifact_labels", "progress_events": []},
        draft_packet,
        review_packet,
        execution_results,
        tmp_path,
    )

    assert package.weather.data_source == "fallback_estimate"
    assert package.budget.data_source == "fallback_estimate"
    assert package.calendar_file is not None
    assert package.review.data_source == "fallback_estimate"
    markdown = package.markdown_file.read_text(encoding="utf-8")
    ics = package.calendar_file.read_text(encoding="utf-8")
    assert "## Data Sources" in markdown
    assert "WEATHER: fallback / fallback_estimate" in markdown
    assert "Menxia Review: fallback_estimate" in markdown
    assert "X-MA-DATA-SOURCE:" in ics
    assert "fallback_estimate" in ics


def test_markdown_labels_search_fallback_links_transparently(tmp_path):
    request = PlanningRequest(
        request_id="link_labels",
        user_message="Plan Shanghai",
        profile=TravelerProfile(
            destination_preferences=["上海"],
            origin_city="Beijing",
            start_date="2026-05-01",
            end_date="2026-05-01",
            total_budget=3000,
            currency="USD",
        ),
    )
    draft_packet = {
        "request_id": request.request_id,
        "destination": "上海",
        "itinerary_draft": {
            "destination": "上海",
            "overview": "Named Shanghai plan",
            "trip_style": "balanced",
            "daily_plan": [
                {
                    "day_index": 1,
                    "date": "2026-05-01",
                    "city": "上海",
                    "theme": "Museums",
                    "summary": "Visit named places.",
                    "activities": [
                        {
                            "start_time": "09:00",
                            "end_time": "10:30",
                            "title": "上海博物馆",
                            "location_name": "上海博物馆",
                            "description": "A confirmed POI with only a search fallback link.",
                            "estimated_cost": 0,
                            "map_link": "https://ditu.amap.com/search?query=%E4%B8%8A%E6%B5%B7%E5%8D%9A%E7%89%A9%E9%A6%86",
                            "search_url": "https://ditu.amap.com/search?query=%E4%B8%8A%E6%B5%B7%E5%8D%9A%E7%89%A9%E9%A6%86",
                            "link_confidence": "search_fallback",
                        },
                        {
                            "start_time": "11:00",
                            "end_time": "12:00",
                            "title": "Shanghai Museum",
                            "location_name": "上海博物馆",
                            "description": "A provider result with a real map link and official site.",
                            "estimated_cost": 0,
                            "map_link": "https://www.google.com/maps/place/Shanghai+Museum",
                            "booking_link": "https://www.shanghaimuseum.net/",
                            "link_confidence": "provider_result",
                        },
                    ],
                }
            ],
            "planning_notes": [],
            "pending_confirmations": [],
            "risk_flags": [],
        },
    }
    review_packet = {
        "request_id": request.request_id,
        "verdict": "APPROVED",
        "summary": "Approved link label fixture.",
        "blocking_issues": [],
        "revision_requests": [],
        "human_questions": [],
        "approved_bureaus": [],
        "governance": {
            "reviewer": "MENXIA",
            "source_producer": "ZHONGSHU",
            "next_hop": "SHANGSHU",
            "verdict_state": "APPROVED",
            "veto_enabled": True,
            "rejection_round": 0,
            "max_rejection_rounds": 2,
        },
        "data_source": "live",
    }

    package = ProvinceWorkflow.build_final_package(
        request,
        {"dashboard_url": "http://testserver/dashboard/link_labels", "progress_events": []},
        draft_packet,
        review_packet,
        {},
        tmp_path,
    )

    markdown = package.markdown_file.read_text(encoding="utf-8")
    assert "[Search](https://ditu.amap.com/search?query=%E4%B8%8A%E6%B5%B7%E5%8D%9A%E7%89%A9%E9%A6%86)" in markdown
    assert "[Map](https://ditu.amap.com/search?query=%E4%B8%8A%E6%B5%B7%E5%8D%9A%E7%89%A9%E9%A6%86)" not in markdown
    assert "[Map](https://www.google.com/maps/place/Shanghai+Museum)" in markdown
    assert "[Website](https://www.shanghaimuseum.net/)" in markdown
