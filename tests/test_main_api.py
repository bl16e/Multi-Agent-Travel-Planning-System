from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import main
from main import ThreeProvinceTravelSystem
from utils.schemas import PlanningRequest, TravelerProfile
from utils.session_cache import InMemorySessionCache
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
            "resume_mode": "boundary",
            "resume_state": {
                "mode": "boundary",
                "thread_id": request.request_id,
                "interrupt_id": "fake",
                "question": "Need budget",
                "next": ["interrupt_preflight"],
                "created_at": "2026-06-24T00:00:00Z",
            },
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


def test_plan_stream_returns_sse_events(monkeypatch):
    class StreamingFakeSystem:
        sessions = {}

        async def plan_trip(self, request):
            raise AssertionError("streaming endpoint must use planner stream")

        async def stream_plan_trip(self, request):
            yield {"event": "progress", "data": {"message": "started", "request_id": request.request_id}}
            yield {
                "event": "result",
                "data": {
                    "status": "HUMAN_INTERVENE",
                    "request_id": request.request_id,
                    "question": "Need budget",
                    "resume_mode": "boundary",
                    "resume_state": {"mode": "boundary", "thread_id": request.request_id},
                },
            }

    monkeypatch.setattr(main, "create_travel_system", lambda progress_reporter=None: StreamingFakeSystem())
    client = TestClient(main.app, raise_server_exceptions=False)

    response = client.post("/plan/stream", json=make_request("stream_trip").model_dump(mode="json"))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: progress" in response.text
    assert "event: result" in response.text
    assert "event: done" in response.text


def test_resume_stream_returns_sse_events(monkeypatch):
    class StreamingFakeSystem:
        sessions = {}

        async def resume_trip(self, request_id, payload):
            raise AssertionError("streaming endpoint must use planner stream")

        async def stream_resume_trip(self, request_id, payload):
            yield {"event": "progress", "data": {"message": "resumed", "request_id": request_id}}
            yield {"event": "result", "data": {"status": "DONE", "request_id": request_id, "resume_mode": "none", "resume_state": {}}}

    monkeypatch.setattr(main, "_load_resume_session", lambda request_id: {"request": make_request(request_id), "status": "HUMAN_INTERVENE", "resume_mode": "boundary", "resume_state": {"thread_id": request_id}})
    monkeypatch.setattr(main, "create_travel_system", lambda progress_reporter=None: StreamingFakeSystem())
    client = TestClient(main.app, raise_server_exceptions=False)

    response = client.post("/resume/stream_trip/stream", json={"profile_updates": {"total_budget": 2500}})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: progress" in response.text
    assert "event: result" in response.text
    assert "event: done" in response.text


@pytest.mark.asyncio
async def test_resume_trip_uses_json_session_store_boundary_metadata(tmp_path):
    store = JsonSessionStore(tmp_path)
    first = ThreeProvinceTravelSystem(session_store=store)
    first.workflow = FakeWorkflow()

    result = await first.plan_trip(make_request(total_budget=None))

    second = ThreeProvinceTravelSystem(session_store=store)
    assert result["resume_mode"] == "boundary"
    loaded = second._load_session("persisted_trip")
    assert loaded["resume_mode"] == "boundary"
    assert loaded["resume_state"]["thread_id"] == "persisted_trip"


def test_download_artifact_rejects_unsafe_request_id_alias(tmp_path, monkeypatch):
    artifact = tmp_path / "bad_secret_travel_plan.md"
    artifact.write_text("# safe", encoding="utf-8")
    outside = tmp_path.parent / "secret_travel_plan.md"
    outside.write_text("# outside", encoding="utf-8")
    monkeypatch.setattr(main, "DEFAULT_ARTIFACT_DIR", Path(tmp_path))

    client = TestClient(main.app)
    response = client.get("/download/bad:secret")

    assert response.status_code == 404


def test_download_markdown_artifact_preserves_data_source_labels(tmp_path, monkeypatch):
    artifact = tmp_path / "labeled_trip_travel_plan.md"
    artifact.write_text("# Trip\n\n## Data Sources\n- WEATHER: fallback / fallback_estimate", encoding="utf-8")
    monkeypatch.setattr(main, "DEFAULT_ARTIFACT_DIR", Path(tmp_path))

    client = TestClient(main.app)
    response = client.get("/download/labeled_trip")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "## Data Sources" in response.text
    assert "fallback_estimate" in response.text


def test_download_calendar_artifact_preserves_data_source_labels(tmp_path, monkeypatch):
    artifact = tmp_path / "calendar_labeled_trip_trip_calendar.ics"
    artifact.write_text(
        "BEGIN:VCALENDAR\r\nVERSION:2.0\r\nX-MA-DATA-SOURCE:fallback_estimate\r\nEND:VCALENDAR\r\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(main, "DEFAULT_ARTIFACT_DIR", Path(tmp_path))

    client = TestClient(main.app)
    response = client.get("/download/calendar_labeled_trip")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/calendar")
    assert "X-MA-DATA-SOURCE:fallback_estimate" in response.text


def test_plan_api_uses_fresh_planner_instance_per_request(monkeypatch):
    created = []

    class IsolatedFakeSystem:
        def __init__(self):
            self.instance_id = len(created)
            self.sessions = {}
            created.append(self)

        async def plan_trip(self, request):
            return {
                "status": "HUMAN_INTERVENE",
                "request_id": request.request_id,
                "question": f"instance-{self.instance_id}",
            }

    monkeypatch.setattr(main, "create_travel_system", lambda progress_reporter=None: IsolatedFakeSystem())
    client = TestClient(main.app)

    first = client.post("/plan", json=make_request("isolated_one").model_dump(mode="json"))
    second = client.post("/plan", json=make_request("isolated_two").model_dump(mode="json"))

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["question"] == "instance-0"
    assert second.json()["question"] == "instance-1"
    assert len(created) == 2


def test_concurrent_plan_requests_keep_sessions_isolated(monkeypatch):
    cache = InMemorySessionCache(max_entries=10, ttl_seconds=3600)

    class ConcurrentFakeSystem:
        def __init__(self):
            self.sessions = {}

        async def plan_trip(self, request):
            result = {
                "status": "HUMAN_INTERVENE",
                "request_id": request.request_id,
                "question": f"Need budget for {request.profile.destination_preferences[0]}",
            }
            self.sessions[request.request_id] = {
                "request": request,
                "status": result["status"],
                "result": result,
                "context_snapshot": {"progress_events": [{"request_id": request.request_id}]},
            }
            return result

    monkeypatch.setattr(main, "_shared_sessions", cache)
    monkeypatch.setattr(main, "create_travel_system", lambda progress_reporter=None: ConcurrentFakeSystem())

    tokyo = make_request("concurrent_tokyo").model_dump(mode="json")
    seoul = make_request("concurrent_seoul").model_dump(mode="json")
    seoul["profile"]["destination_preferences"] = ["Seoul"]

    def post_plan(payload):
        with TestClient(main.app, raise_server_exceptions=False) as client:
            return client.post("/plan", json=payload)

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(post_plan, [tokyo, seoul]))

    assert {response.status_code for response in responses} == {200}
    bodies = {response.json()["request_id"]: response.json() for response in responses}
    assert bodies["concurrent_tokyo"]["question"] == "Need budget for Tokyo"
    assert bodies["concurrent_seoul"]["question"] == "Need budget for Seoul"
    assert cache.get("concurrent_tokyo")["request"].profile.destination_preferences == ["Tokyo"]
    assert cache.get("concurrent_seoul")["request"].profile.destination_preferences == ["Seoul"]


def test_api_created_sessions_obey_shared_cache_eviction(monkeypatch):
    cache = InMemorySessionCache(max_entries=1, ttl_seconds=3600)

    class CachedFakeSystem:
        def __init__(self):
            self.sessions = {}

        async def plan_trip(self, request):
            result = {"status": "HUMAN_INTERVENE", "request_id": request.request_id, "question": "Need budget"}
            self.sessions[request.request_id] = {"request": request, "status": result["status"], "result": result}
            return result

    monkeypatch.setattr(main, "_shared_sessions", cache)
    monkeypatch.setattr(main, "create_travel_system", lambda progress_reporter=None: CachedFakeSystem())
    client = TestClient(main.app, raise_server_exceptions=False)

    first = client.post("/plan", json=make_request("cache_evict_first").model_dump(mode="json"))
    second = client.post("/plan", json=make_request("cache_evict_second").model_dump(mode="json"))

    assert first.status_code == 200
    assert second.status_code == 200
    assert cache.get("cache_evict_first") is None
    assert cache.get("cache_evict_second")["result"]["request_id"] == "cache_evict_second"


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
    class FailingSystem:
        sessions = {}

        async def plan_trip(self, request):
            raise RuntimeError("workflow exploded")

    monkeypatch.setattr(main, "create_travel_system", lambda progress_reporter=None: FailingSystem())
    client = TestClient(main.app, raise_server_exceptions=False)

    response = client.post("/plan", json=make_request("internal_error").model_dump(mode="json"))

    assert response.status_code == 500
    assert response.json()["detail"] == {"error": "planning_failed", "message": "Planning workflow failed"}


def test_plan_api_invalid_date_range_stops_before_workflow_execution(monkeypatch):
    created = []

    class ShouldNotRunSystem:
        def __init__(self):
            created.append(self)

        async def plan_trip(self, request):
            raise AssertionError("workflow should not run for invalid request dates")

    monkeypatch.setattr(main, "create_travel_system", lambda progress_reporter=None: ShouldNotRunSystem())
    payload = make_request("bad_dates_api", total_budget=1000).model_dump(mode="json")
    payload["profile"]["start_date"] = "2026-05-05"
    payload["profile"]["end_date"] = "2026-05-01"

    client = TestClient(main.app, raise_server_exceptions=False)
    response = client.post("/plan", json=payload)

    assert response.status_code == 422
    assert created == []


def test_dashboard_returns_conflict_for_corrupt_session(tmp_path, monkeypatch):
    store = JsonSessionStore(tmp_path)
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(main, "session_lookup", main.SessionLookup(session_store=store))

    client = TestClient(main.app, raise_server_exceptions=False)
    response = client.get("/dashboard/broken")

    assert response.status_code == 409
    assert response.json()["detail"] == "Stored session is corrupt"


def test_dashboard_exposes_boundary_resume_metadata(tmp_path, monkeypatch):
    store = JsonSessionStore(tmp_path)
    request = make_request("dashboard_resume", total_budget=None)
    store.save(
        main.StoredSession(
            request_id="dashboard_resume",
            request=request.model_dump(mode="json"),
            status="HUMAN_INTERVENE",
            context_snapshot={
                "current_state": "HUMAN_INTERVENE",
                "pending_user_inputs": ["Need budget"],
                "progress_events": [{"stage": "preflight"}],
            },
            resume_mode="boundary",
            resume_state={"mode": "boundary", "thread_id": "dashboard_resume", "question": "Need budget"},
            result={"status": "HUMAN_INTERVENE", "question": "Need budget"},
        )
    )
    monkeypatch.setattr(main, "session_lookup", main.SessionLookup(session_store=store))

    client = TestClient(main.app, raise_server_exceptions=False)
    response = client.get("/dashboard/dashboard_resume")

    assert response.status_code == 200
    data = response.json()
    assert data["resume_mode"] == "boundary"
    assert data["resume_state"]["thread_id"] == "dashboard_resume"


def test_resume_api_returns_conflict_for_legacy_session_without_checkpoint(monkeypatch):
    request = make_request("resume_contract", total_budget=None)
    main._shared_sessions.set(
        "resume_contract",
        {"request": request, "status": "HUMAN_INTERVENE", "resume_mode": "none", "resume_state": {}},
    )

    client = TestClient(main.app, raise_server_exceptions=False)
    response = client.post("/resume/resume_contract", json={"profile_updates": {"total_budget": 2500}})

    assert response.status_code == 409
    assert "boundary checkpoint" in response.json()["detail"]


def test_resume_read_cleans_shared_cache_before_session_lookup(monkeypatch):
    class CleanupSpyCache:
        def __init__(self):
            self.cleanup_calls = 0

        def cleanup_expired(self):
            self.cleanup_calls += 1

        def get(self, request_id):
            return None

    class MissingLookup:
        def _load_session(self, request_id):
            return None

    cache = CleanupSpyCache()
    monkeypatch.setattr(main, "_shared_sessions", cache)
    monkeypatch.setattr(main, "session_lookup", MissingLookup())
    client = TestClient(main.app, raise_server_exceptions=False)

    response = client.post("/resume/cache_cleanup_resume", json={"profile_updates": {"total_budget": 2500}})

    assert response.status_code == 404
    assert cache.cleanup_calls == 1


def test_dashboard_cleans_shared_cache_before_session_lookup(monkeypatch):
    class CleanupSpyCache:
        def __init__(self):
            self.cleanup_calls = 0

        def cleanup_expired(self):
            self.cleanup_calls += 1

        def get(self, request_id):
            return None

    class DashboardLookup:
        def dashboard_snapshot(self, request_id):
            return {"request_id": request_id, "status": "HUMAN_INTERVENE"}

    cache = CleanupSpyCache()
    monkeypatch.setattr(main, "_shared_sessions", cache)
    monkeypatch.setattr(main, "session_lookup", DashboardLookup())
    client = TestClient(main.app, raise_server_exceptions=False)

    response = client.get("/dashboard/cache_cleanup_dashboard")

    assert response.status_code == 200
    assert response.json()["request_id"] == "cache_cleanup_dashboard"
    assert cache.cleanup_calls == 1
