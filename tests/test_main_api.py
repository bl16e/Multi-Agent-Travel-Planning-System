from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import main
from main import HumanResumePayload, ThreeProvinceTravelSystem
from utils.schemas import PlanningRequest, TravelerProfile
from utils.session_store import JsonSessionStore


class FakeWorkflow:
    def __init__(self):
        self.orchestrator = SimpleNamespace(
            build_dashboard_link=lambda context: f"http://testserver/dashboard/{context.request_id}"
        )
        self.requests = []

    async def run(self, request):
        self.requests.append(request)
        context = SimpleNamespace(
            request_id=request.request_id,
            current_state=SimpleNamespace(value="HUMAN_INTERVENE"),
            pending_user_inputs=["Need budget"],
            progress_events=[{"stage": "test"}],
        )
        return {
            "status": "HUMAN_INTERVENE",
            "question": "Need budget",
            "context": context,
        }


def make_request(request_id="persisted_trip", total_budget=None):
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


@pytest.mark.asyncio
async def test_resume_trip_uses_json_session_store(tmp_path):
    store = JsonSessionStore(tmp_path)
    first = ThreeProvinceTravelSystem(session_store=store)
    first.workflow = FakeWorkflow()

    await first.plan_trip(make_request(total_budget=None))

    second = ThreeProvinceTravelSystem(session_store=store)
    second.workflow = FakeWorkflow()
    result = await second.resume_trip(
        "persisted_trip",
        HumanResumePayload(profile_updates={"total_budget": 2500}),
    )

    assert result["status"] == "HUMAN_INTERVENE"
    assert second.workflow.requests[0].profile.total_budget == 2500


def test_download_artifact_rejects_unsafe_request_id_alias(tmp_path, monkeypatch):
    artifact = tmp_path / "bad_secret_travel_plan.md"
    artifact.write_text("# safe", encoding="utf-8")
    outside = tmp_path.parent / "secret_travel_plan.md"
    outside.write_text("# outside", encoding="utf-8")
    monkeypatch.setattr(main, "DEFAULT_ARTIFACT_DIR", Path(tmp_path))

    client = TestClient(main.app)
    response = client.get("/download/bad:secret")

    assert response.status_code == 404


def test_plan_stream_emits_progress_result_and_done(monkeypatch):
    class StreamingFakeSystem:
        def __init__(self, artifact_dir=None, progress_reporter=None):
            self.progress_reporter = progress_reporter
            self.sessions = {}

        async def plan_trip(self, request):
            if self.progress_reporter:
                self.progress_reporter("fake progress")
            self.sessions[request.request_id] = {"request": request, "status": "HUMAN_INTERVENE"}
            return {"status": "HUMAN_INTERVENE", "request_id": request.request_id, "question": "Need budget"}

    monkeypatch.setattr(main, "ThreeProvinceTravelSystem", StreamingFakeSystem)
    client = TestClient(main.app)

    response = client.post("/plan/stream", json=make_request("stream_trip").model_dump(mode="json"))

    assert response.status_code == 200
    body = response.text
    assert "event: progress" in body
    assert "event: result" in body
    assert "event: done" in body


def test_planning_request_rejects_unsafe_request_id():
    with pytest.raises(ValidationError):
        make_request("bad/request")


def test_planning_request_rejects_end_date_before_start_date():
    with pytest.raises(ValidationError):
        PlanningRequest(
            request_id="bad_dates",
            user_message="Plan impossible dates",
            profile=TravelerProfile(
                origin_city="Beijing",
                destination_preferences=["Tokyo"],
                start_date="2026-05-05",
                end_date="2026-05-01",
                total_budget=1000,
            ),
        )


def test_plan_api_rejects_unsafe_request_id():
    client = TestClient(main.app)
    payload = make_request("safe_request").model_dump(mode="json")
    payload["request_id"] = "bad/request"

    response = client.post("/plan", json=payload)

    assert response.status_code == 422


def test_plan_api_returns_structured_500_for_internal_error(monkeypatch):
    async def raise_internal_error(request):
        raise RuntimeError("workflow exploded")

    monkeypatch.setattr(main.system, "plan_trip", raise_internal_error)
    client = TestClient(main.app, raise_server_exceptions=False)

    response = client.post("/plan", json=make_request("internal_error").model_dump(mode="json"))

    assert response.status_code == 500
    assert response.json()["detail"] == "Planning workflow failed"


def test_dashboard_returns_conflict_for_corrupt_session(tmp_path, monkeypatch):
    store = JsonSessionStore(tmp_path)
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(main, "system", ThreeProvinceTravelSystem(session_store=store))

    client = TestClient(main.app, raise_server_exceptions=False)
    response = client.get("/dashboard/broken")

    assert response.status_code == 409
    assert response.json()["detail"] == "Stored session is corrupt"
