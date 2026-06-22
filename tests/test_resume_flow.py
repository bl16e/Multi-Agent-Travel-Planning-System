from __future__ import annotations

from utils.schemas import BudgetExecutionResult, CalendarExecutionResult, WeatherExecutionResult
from utils.session_store import JsonSessionStore, StoredSession


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
