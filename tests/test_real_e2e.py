from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import main
from main import HumanResumePayload
from utils.llm_factory import build_qwen_chat
from utils.schemas import PlanningRequest, TravelerProfile
from utils.session_cache import InMemorySessionCache
from utils.session_store import JsonSessionStore
from utils.settings import get_settings
from workflow import ProvinceWorkflow


def make_e2e_request(request_id: str, *, total_budget: int | None = None) -> PlanningRequest:
    return PlanningRequest(
        request_id=request_id,
        user_message=(
            "Plan a concrete Tokyo trip with named places, transport notes, "
            "budget labels, and calendar-ready timing."
        ),
        profile=TravelerProfile(
            origin_city="Beijing",
            origin_airport_code="PEK",
            destination_preferences=["Tokyo"],
            destination_airport_code="HND",
            start_date="2026-10-10",
            end_date="2026-10-11",
            total_budget=total_budget,
            currency="USD",
            adults=2,
            interests=["culture", "food"],
            constraints=["avoid generic placeholder activities"],
            pace="structured",
        ),
    )


def configure_offline_e2e(monkeypatch: pytest.MonkeyPatch, tmp_path):
    output_dir = tmp_path / "artifacts"
    session_dir = tmp_path / "sessions"
    checkpoint_db = tmp_path / "checkpoints" / "langgraph.sqlite"
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("SESSION_STORE_DIR", str(session_dir))
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_DB", str(checkpoint_db))
    monkeypatch.setenv("QWEN_API_KEY", "")
    monkeypatch.setenv("AMAP_API_KEY", "")
    monkeypatch.setenv("SERPAPI_API_KEY", "")
    monkeypatch.setenv("PLAN_REQUEST_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("MCP_TOOLING_TIMEOUT_SECONDS", "0.05")
    monkeypatch.setenv("LIUBU_TOOL_TIMEOUT_SECONDS", "0.05")
    monkeypatch.setenv("QWEN_TIMEOUT_SECONDS", "1")

    get_settings.cache_clear()
    build_qwen_chat.cache_clear()

    try:
        from utils import mcp_client

        mcp_client._CLIENT_CACHE.clear()
        mcp_client._TOOLS_CACHE.clear()
    except Exception:
        pass

    return output_dir, session_dir, checkpoint_db


@pytest.mark.asyncio
async def test_local_workflow_e2e_resumes_and_writes_artifacts(tmp_path, monkeypatch):
    output_dir, _, checkpoint_db = configure_offline_e2e(monkeypatch, tmp_path)
    progress_lines: list[str] = []
    workflow = ProvinceWorkflow(artifact_dir=output_dir, progress_reporter=progress_lines.append)
    request = make_e2e_request("local_real_e2e", total_budget=None)

    interrupted = await workflow.run(request)

    assert interrupted["status"] == "HUMAN_INTERVENE"
    assert interrupted["resume_mode"] == "boundary"
    assert interrupted["resume_state"]["thread_id"] == request.request_id
    assert interrupted["resume_state"]["next"]
    assert checkpoint_db.exists()

    resumed = await workflow.resume(
        interrupted["resume_state"],
        {"profile_updates": {"total_budget": 2600}},
    )

    assert resumed["status"] == "DONE"
    package = resumed["final_package"]
    assert package["request_id"] == request.request_id
    assert package["workflow_state"] == "DONE"
    assert package["destination"] == "Tokyo"
    assert package["review"]["data_source"] == "fallback_estimate"
    assert package["weather"]["data_source"] == "fallback_estimate"
    assert package["budget"]["data_source"] == "fallback_estimate"
    assert package["calendar_file"].endswith("_trip_calendar.ics")
    assert package["itinerary"]["daily_plan"]

    markdown_path = output_dir / f"{request.request_id}_travel_plan.md"
    calendar_path = output_dir / f"{request.request_id}_trip_calendar.ics"
    assert markdown_path.exists()
    assert calendar_path.exists()
    assert "## Data Sources" in markdown_path.read_text(encoding="utf-8")
    assert "X-MA-DATA-SOURCE:fallback_estimate" in calendar_path.read_text(encoding="utf-8")
    assert any("shangshu_preflight" in line for line in progress_lines)
    assert any("shangshu_assemble" in line for line in progress_lines)


def test_api_e2e_plan_resume_dashboard_and_download(tmp_path, monkeypatch):
    output_dir, session_dir, _ = configure_offline_e2e(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "DEFAULT_ARTIFACT_DIR", output_dir)
    monkeypatch.setattr(main, "_shared_sessions", InMemorySessionCache(max_entries=10, ttl_seconds=3600))
    monkeypatch.setattr(main, "session_lookup", main.SessionLookup(JsonSessionStore(session_dir)))
    request = make_e2e_request("api_real_e2e", total_budget=None)

    client = TestClient(main.app, raise_server_exceptions=False)
    plan_response = client.post("/plan", json=request.model_dump(mode="json"))

    assert plan_response.status_code == 200, plan_response.text
    interrupted = plan_response.json()
    assert interrupted["status"] == "HUMAN_INTERVENE"
    assert interrupted["resume_mode"] == "boundary"
    assert interrupted["resume_state"]["thread_id"] == request.request_id

    waiting_dashboard = client.get(f"/dashboard/{request.request_id}")
    assert waiting_dashboard.status_code == 200
    assert waiting_dashboard.json()["resume_state"]["question"]

    resume_response = client.post(
        f"/resume/{request.request_id}",
        json=HumanResumePayload(profile_updates={"total_budget": 2600}).model_dump(mode="json"),
    )

    assert resume_response.status_code == 200, resume_response.text
    final_package = resume_response.json()
    assert final_package["request_id"] == request.request_id
    assert final_package["workflow_state"] == "DONE"
    assert final_package["resume_mode"] == "none"
    assert final_package["resume_state"] == {}
    assert final_package["weather"]["data_source"] == "fallback_estimate"
    assert final_package["budget"]["data_source"] == "fallback_estimate"
    assert final_package["calendar_file"].endswith("_trip_calendar.ics")

    done_dashboard = client.get(f"/dashboard/{request.request_id}")
    assert done_dashboard.status_code == 200
    assert done_dashboard.json()["status"] == "DONE"
    assert done_dashboard.json()["has_package"] is True

    markdown_response = client.get(f"/download/{request.request_id}")
    assert markdown_response.status_code == 200
    assert markdown_response.headers["content-type"].startswith("text/markdown")
    assert "## Data Sources" in markdown_response.text
    assert "WEATHER: fallback / fallback_estimate" in markdown_response.text
