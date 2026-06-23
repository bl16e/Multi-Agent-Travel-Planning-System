import os
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

import main
from utils.llm_factory import build_qwen_chat
from utils.settings import get_settings


@pytest.mark.live
def test_live_full_planning_flow_through_plan_api(tmp_path, monkeypatch):
    if os.getenv("RUN_LIVE_TESTS") != "1":
        pytest.skip("RUN_LIVE_TESTS=1 is required for live E2E tests")
    if not os.getenv("QWEN_API_KEY"):
        pytest.skip("QWEN_API_KEY is required for live E2E tests")

    get_settings.cache_clear()
    build_qwen_chat.cache_clear()
    if build_qwen_chat() is None:
        pytest.skip("Qwen client is not configured")

    monkeypatch.setattr(main, "DEFAULT_ARTIFACT_DIR", tmp_path)
    start_date = date.today() + timedelta(days=120)
    end_date = start_date
    payload = {
        "request_id": "live_e2e_tokyo_plan",
        "user_message": "Plan a compact Tokyo trip with concrete places, transport notes, and calendar-ready timing.",
        "profile": {
            "origin_city": "Beijing",
            "origin_airport_code": "PEK",
            "destination_preferences": ["Tokyo"],
            "destination_airport_code": "HND",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "adults": 1,
            "children": 0,
            "budget_level": "mid_range",
            "total_budget": 1800,
            "currency": "USD",
            "interests": ["culture", "food"],
            "constraints": ["avoid generic placeholder activities"],
            "pace": "structured",
        },
    }

    client = TestClient(main.app, raise_server_exceptions=False)
    response = client.post("/plan", json=payload)

    assert response.status_code == 200, response.text
    data = response.json()
    final_status = data.get("status") or data.get("workflow_state")
    assert final_status in {"DONE", "HUMAN_INTERVENE", "REJECTED"}
    assert data.get("request_id") == payload["request_id"]

    if final_status == "DONE":
        assert data["destination"] == "Tokyo"
        assert data["workflow_state"] == "DONE"
        assert data["review"]["data_source"] in {"structured_llm", "live"}
        for bureau_name in ("weather", "budget"):
            assert data[bureau_name]["data_source"] in {"structured_llm", "fallback_estimate"}
        for bureau_name in ("accommodation", "flight_transport"):
            bureau_payload = data[bureau_name]
            assert bureau_payload["data_source"] in {"live", "structured_llm", "fallback_estimate"}
            assert "liubu_quality" in bureau_payload
            assert "liubu_evidence" in bureau_payload
            if bureau_payload["status"] == "ok":
                assert bureau_payload["liubu_quality"]["passed"] is True
            else:
                assert bureau_payload.get("warnings") or bureau_payload.get("transport_notes")
        if data.get("calendar_file"):
            assert data["calendar_file"].endswith("_trip_calendar.ics")
        assert data["itinerary"]["daily_plan"]
        assert data["progress_events"]
    elif final_status == "REJECTED":
        assert data["review"]["data_source"] == "structured_llm"
        assert data["review"]["summary"]
    else:
        assert data["question"]
