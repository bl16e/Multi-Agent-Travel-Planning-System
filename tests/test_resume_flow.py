from __future__ import annotations

import pytest

from main import BoundaryResumeConflict, HumanResumePayload, ThreeProvinceTravelSystem
from utils.schemas import BudgetExecutionResult, CalendarExecutionResult, PlanningRequest, TravelerProfile, WeatherExecutionResult
from utils.session_store import JsonSessionStore, StoredSession


def make_request(request_id="resume_trip", total_budget=None):
    return PlanningRequest(
        request_id=request_id,
        user_message="Plan Tokyo",
        profile=TravelerProfile(
            origin_city="Beijing",
            destination_preferences=["Tokyo"],
            start_date="2026-05-01",
            end_date="2026-05-03",
            total_budget=total_budget,
        ),
    )


def test_stored_session_round_trips_boundary_resume_metadata(tmp_path):
    store = JsonSessionStore(tmp_path)
    session = StoredSession(
        request_id="trip_resume",
        request={"request_id": "trip_resume", "user_message": "Plan Tokyo", "profile": {}},
        status="HUMAN_INTERVENE",
        resume_state={
            "mode": "boundary",
            "thread_id": "trip_resume",
            "interrupt_id": "abc",
            "question": "Need budget",
            "next": ["interrupt_preflight"],
            "created_at": "2026-06-24T00:00:00Z",
        },
        resume_mode="boundary",
    )

    store.save(session)

    loaded = store.load("trip_resume")
    assert loaded is not None
    assert loaded.resume_state["thread_id"] == "trip_resume"
    assert loaded.resume_state["question"] == "Need budget"
    assert loaded.resume_mode == "boundary"


def test_stored_session_defaults_resume_metadata_for_legacy_json(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        """
        {
          "schema_version": 1,
          "request_id": "legacy",
          "request": {"request_id": "legacy", "user_message": "Plan Tokyo", "profile": {}},
          "status": "HUMAN_INTERVENE"
        }
        """,
        encoding="utf-8",
    )

    loaded = JsonSessionStore(tmp_path).load("legacy")

    assert loaded is not None
    assert loaded.resume_state == {}
    assert loaded.resume_mode == "none"


def test_execution_result_models_expose_status_and_data_source_defaults(tmp_path):
    weather = WeatherExecutionResult(
        destination="Tokyo",
        forecast_days=[
            {
                "date": "2026-05-01",
                "condition": "Estimated",
                "min_temp_c": 18,
                "max_temp_c": 24,
                "precipitation_probability": 0.2,
                "activity_suitability": "Flexible",
            }
        ],
        packing_list=[],
        summary="Estimated forecast",
    )
    budget = BudgetExecutionResult(
        currency="USD",
        budget_breakdown=[],
        total_estimated_cost=0,
    )
    calendar = CalendarExecutionResult(
        calendar_file=tmp_path / "trip.ics",
        events_created=0,
        calendar_name="Trip",
    )

    for result in (weather, budget, calendar):
        assert result.status in {"ok", "fallback", "error"}
        assert result.data_source in {"live", "structured_llm", "fallback_estimate", "unavailable"}


@pytest.mark.asyncio
async def test_resume_trip_uses_langgraph_command_resume_across_fresh_system(tmp_path):
    store = JsonSessionStore(tmp_path)
    request = make_request(total_budget=None)
    first = ThreeProvinceTravelSystem(artifact_dir=tmp_path, session_store=store)

    interrupted = await first.plan_trip(request)

    assert interrupted["status"] == "HUMAN_INTERVENE"
    assert interrupted["resume_mode"] == "boundary"
    assert interrupted["resume_state"]["thread_id"] == request.request_id
    assert interrupted["resume_state"]["question"]

    second = ThreeProvinceTravelSystem(artifact_dir=tmp_path, session_store=store)
    resumed = await second.resume_trip(
        request.request_id,
        HumanResumePayload(profile_updates={"total_budget": 2500}),
    )

    payload = resumed.model_dump(mode="json") if hasattr(resumed, "model_dump") else resumed
    assert payload["request_id"] == request.request_id
    assert payload.get("status") in {"REJECTED", "HUMAN_INTERVENE"} or payload.get("workflow_state") == "DONE"


@pytest.mark.asyncio
async def test_resume_trip_rejects_legacy_session_without_boundary_checkpoint(tmp_path):
    store = JsonSessionStore(tmp_path)
    request = make_request(total_budget=None)
    store.save(
        StoredSession(
            request_id=request.request_id,
            request=request.model_dump(mode="json"),
            status="HUMAN_INTERVENE",
            result={"status": "HUMAN_INTERVENE", "question": "Need budget"},
            resume_mode="none",
        )
    )
    system = ThreeProvinceTravelSystem(artifact_dir=tmp_path, session_store=store)

    with pytest.raises(BoundaryResumeConflict, match="boundary checkpoint"):
        await system.resume_trip(
            request.request_id,
            HumanResumePayload(profile_updates={"total_budget": 2500}),
        )
