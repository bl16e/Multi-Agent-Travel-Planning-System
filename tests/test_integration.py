import pytest
from provinces.liubu.calendar.service import CalendarBureau
from provinces.liubu.constrained.state import normalize_worker_input
from workflow import ProvinceWorkflow
from utils.schemas import PlanningRequest, TravelerProfile


@pytest.mark.asyncio
async def test_full_workflow_integration():
    """完整工作流集成测试（包含LLM调用）"""
    workflow = ProvinceWorkflow()
    request = PlanningRequest(
        request_id="test_integration_001",
        user_message="计划东京3日游",
        profile=TravelerProfile(
            destination_preferences=["Tokyo"],
            origin_city="Beijing",
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
        assert package["destination"] == "Tokyo"
        assert package["workflow_state"] == "DONE"
        assert package["itinerary"]["destination"] == "Tokyo"
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
async def test_final_package_and_artifacts_expose_data_source_labels(tmp_path):
    request = PlanningRequest(
        request_id="artifact_labels",
        user_message="Plan Kyoto",
        profile=TravelerProfile(
            destination_preferences=["Kyoto"],
            origin_city="Beijing",
            start_date="2026-05-01",
            end_date="2026-05-01",
            total_budget=3000,
            currency="USD",
        ),
    )
    draft_packet = {
        "request_id": request.request_id,
        "destination": "Kyoto",
        "itinerary_draft": {
            "destination": "Kyoto",
            "overview": "Named Kyoto plan",
            "trip_style": "balanced",
            "daily_plan": [
                {
                    "day_index": 1,
                    "date": "2026-05-01",
                    "city": "Kyoto",
                    "theme": "Temples",
                    "summary": "Visit named temples.",
                    "activities": [
                        {
                            "start_time": "09:00",
                            "end_time": "10:30",
                            "title": "Kiyomizu-dera temple visit",
                            "location_name": "Kiyomizu-dera",
                            "description": "Visit the named temple complex.",
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
            "destination": "Kyoto",
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
    assert "X-MA-DATA-SOURCE:fallback_estimate" in ics
