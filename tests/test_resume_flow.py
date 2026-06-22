from __future__ import annotations

import pytest

from main import HumanResumePayload, ThreeProvinceTravelSystem
from utils.schemas import BudgetExecutionResult, CalendarExecutionResult, PlanningRequest, TravelerProfile, WeatherExecutionResult
from utils.session_store import JsonSessionStore, StoredSession


class BoundaryWorkflow:
    def __init__(self):
        self.run_calls = []
        self.resume_calls = []
        self.orchestrator = type(
            "Orchestrator",
            (),
            {"build_dashboard_link": lambda self, context: f"http://testserver/dashboard/{context.request_id}"},
        )()

    async def run(self, request):
        self.run_calls.append(request)
        context = type(
            "Context",
            (),
            {
                "request_id": request.request_id,
                "current_state": type("State", (), {"value": "HUMAN_INTERVENE"})(),
                "pending_user_inputs": ["Need budget"],
                "progress_events": [{"stage": "preflight"}],
            },
        )()
        return {
            "status": "HUMAN_INTERVENE",
            "question": "Need budget",
            "context": context,
            "resume_mode": "boundary",
            "resume_state": {
                "mode": "boundary",
                "next_node": "zhongshu_itinerary",
                "question": "Need budget",
                "state": {"request": request.model_dump(mode="json")},
                "created_at": "2026-06-22T00:00:00+00:00",
            },
        }

    async def resume(self, resume_state, payload):
        self.resume_calls.append((resume_state, payload))
        return {
            "status": "HUMAN_INTERVENE",
            "request_id": resume_state["state"]["request"]["request_id"],
            "question": "Review updated budget",
            "resume_mode": "boundary",
            "resume_state": resume_state,
        }


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


def test_stored_session_round_trips_resume_metadata(tmp_path):
    store = JsonSessionStore(tmp_path)
    session = StoredSession(
        request_id="trip_resume",
        request={"request_id": "trip_resume", "user_message": "Plan Tokyo", "profile": {}},
        status="HUMAN_INTERVENE",
        resume_state={"mode": "boundary", "next_node": "menxia_review"},
        resume_mode="boundary",
    )

    store.save(session)

    loaded = store.load("trip_resume")
    assert loaded is not None
    assert loaded.resume_state == {"mode": "boundary", "next_node": "menxia_review"}
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
async def test_resume_trip_uses_boundary_resume_when_available(tmp_path):
    store = JsonSessionStore(tmp_path)
    first = ThreeProvinceTravelSystem(session_store=store)
    first.workflow = BoundaryWorkflow()

    initial = await first.plan_trip(make_request(total_budget=None))
    assert initial["resume_mode"] == "boundary"

    second = ThreeProvinceTravelSystem(session_store=store)
    second.workflow = BoundaryWorkflow()
    result = await second.resume_trip(
        "resume_trip",
        HumanResumePayload(profile_updates={"total_budget": 2500}),
    )

    assert result["resume_mode"] == "boundary"
    assert result["resume_state"]["mode"] == "boundary"
    assert second.workflow.resume_calls
    assert second.workflow.run_calls == []
    assert second.sessions["resume_trip"]["resume_mode"] == "boundary"


@pytest.mark.asyncio
async def test_resume_trip_labels_legacy_replay_when_boundary_missing(tmp_path):
    store = JsonSessionStore(tmp_path)
    legacy_request = make_request("legacy_resume", total_budget=None)
    store.save(
        StoredSession(
            request_id="legacy_resume",
            request=legacy_request.model_dump(mode="json"),
            status="HUMAN_INTERVENE",
            result={"status": "HUMAN_INTERVENE", "question": "Need budget"},
        )
    )
    system = ThreeProvinceTravelSystem(session_store=store)
    system.workflow = BoundaryWorkflow()

    result = await system.resume_trip(
        "legacy_resume",
        HumanResumePayload(profile_updates={"total_budget": 2500}),
    )

    assert result["resume_mode"] == "replay"
    assert result["replay_reason"] == "Stored session has no compatible boundary resume state."
    assert system.workflow.run_calls
    assert system.workflow.resume_calls == []
    assert system.sessions["legacy_resume"]["resume_mode"] == "replay"
    assert system.sessions["legacy_resume"]["resume_state"]["mode"] == "replay"
