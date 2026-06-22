import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import main
from main import HumanResumePayload, ThreeProvinceTravelSystem
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


def parse_sse_events(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in body.strip().split("\n\n"):
        event_name = None
        event_data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ")
            if line.startswith("data: "):
                event_data = json.loads(line.removeprefix("data: "))
        if event_name is not None and event_data is not None:
            events.append((event_name, event_data))
    return events


def make_resume_state(request: PlanningRequest) -> dict:
    return {
        "mode": "boundary",
        "next_node": "zhongshu_itinerary",
        "question": "Need budget",
        "state": {"request": request.model_dump(mode="json")},
        "created_at": "2026-06-22T00:00:00+00:00",
    }


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


def test_plan_api_uses_fresh_planner_instance_per_request(monkeypatch):
    created = []

    class IsolatedFakeSystem:
        def __init__(self, artifact_dir=None, progress_reporter=None):
            self.instance_id = len(created)
            created.append(self)

        async def plan_trip(self, request):
            return {
                "status": "HUMAN_INTERVENE",
                "request_id": request.request_id,
                "question": f"instance-{self.instance_id}",
            }

    monkeypatch.setattr(main, "ThreeProvinceTravelSystem", IsolatedFakeSystem)
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
        def __init__(self, artifact_dir=None, progress_reporter=None):
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
                "context_snapshot": {
                    "progress_events": [{"request_id": request.request_id}],
                    "pending_user_inputs": [result["question"]],
                },
            }
            return result

    monkeypatch.setattr(main, "_shared_sessions", cache)
    monkeypatch.setattr(main, "ThreeProvinceTravelSystem", ConcurrentFakeSystem)

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


def test_streaming_plan_sessions_are_recorded_by_request_id(monkeypatch):
    cache = InMemorySessionCache(max_entries=10, ttl_seconds=3600)

    class StreamingSessionFakeSystem:
        def __init__(self, artifact_dir=None, progress_reporter=None):
            self.progress_reporter = progress_reporter
            self.sessions = {}

        async def plan_trip(self, request):
            if self.progress_reporter:
                self.progress_reporter(f"{request.request_id}: streaming progress")
            result = {
                "status": "HUMAN_INTERVENE",
                "request_id": request.request_id,
                "question": f"Need budget for {request.request_id}",
            }
            self.sessions[request.request_id] = {
                "request": request,
                "status": result["status"],
                "result": result,
                "context_snapshot": {
                    "progress_events": [{"request_id": request.request_id}],
                },
            }
            return result

    monkeypatch.setattr(main, "_shared_sessions", cache)
    monkeypatch.setattr(main, "ThreeProvinceTravelSystem", StreamingSessionFakeSystem)
    client = TestClient(main.app, raise_server_exceptions=False)

    first = client.post("/plan/stream", json=make_request("stream_iso_a").model_dump(mode="json"))
    second = client.post("/plan/stream", json=make_request("stream_iso_b").model_dump(mode="json"))

    assert first.status_code == 200
    assert second.status_code == 200
    first_events = parse_sse_events(first.text)
    second_events = parse_sse_events(second.text)
    assert next(data for name, data in first_events if name == "result")["request_id"] == "stream_iso_a"
    assert next(data for name, data in second_events if name == "result")["request_id"] == "stream_iso_b"
    assert all(
        "stream_iso_b" not in data["line"]
        for name, data in first_events
        if name == "progress"
    )
    assert all(
        "stream_iso_a" not in data["line"]
        for name, data in second_events
        if name == "progress"
    )
    assert cache.get("stream_iso_a")["result"]["request_id"] == "stream_iso_a"
    assert cache.get("stream_iso_b")["result"]["request_id"] == "stream_iso_b"


def test_api_created_sessions_obey_shared_cache_eviction(monkeypatch):
    cache = InMemorySessionCache(max_entries=1, ttl_seconds=3600)

    class CachedFakeSystem:
        def __init__(self, artifact_dir=None, progress_reporter=None):
            self.sessions = {}

        async def plan_trip(self, request):
            result = {
                "status": "HUMAN_INTERVENE",
                "request_id": request.request_id,
                "question": "Need budget",
            }
            self.sessions[request.request_id] = {
                "request": request,
                "status": result["status"],
                "result": result,
            }
            return result

    monkeypatch.setattr(main, "_shared_sessions", cache)
    monkeypatch.setattr(main, "ThreeProvinceTravelSystem", CachedFakeSystem)
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
        def __init__(self, artifact_dir=None, progress_reporter=None):
            pass

        async def plan_trip(self, request):
            raise RuntimeError("workflow exploded")

    monkeypatch.setattr(main, "ThreeProvinceTravelSystem", FailingSystem)
    client = TestClient(main.app, raise_server_exceptions=False)

    response = client.post("/plan", json=make_request("internal_error").model_dump(mode="json"))

    assert response.status_code == 500
    assert response.json()["detail"] == {
        "error": "planning_failed",
        "message": "Planning workflow failed",
    }


def test_plan_api_invalid_date_range_stops_before_workflow_execution(monkeypatch):
    created = []

    class ShouldNotRunSystem:
        def __init__(self, artifact_dir=None, progress_reporter=None):
            created.append(self)

        async def plan_trip(self, request):
            raise AssertionError("workflow should not run for invalid request dates")

    monkeypatch.setattr(main, "ThreeProvinceTravelSystem", ShouldNotRunSystem)
    payload = make_request("bad_dates_api", total_budget=1000).model_dump(mode="json")
    payload["profile"]["start_date"] = "2026-05-05"
    payload["profile"]["end_date"] = "2026-05-01"

    client = TestClient(main.app, raise_server_exceptions=False)
    response = client.post("/plan", json=payload)

    assert response.status_code == 422
    assert created == []


def test_plan_stream_matches_non_streaming_for_success_human_and_error(monkeypatch):
    class OutcomeSystem:
        def __init__(self, artifact_dir=None, progress_reporter=None):
            self.progress_reporter = progress_reporter
            self.sessions = {}

        async def plan_trip(self, request):
            if self.progress_reporter:
                self.progress_reporter(f"running {request.request_id}")
            if request.request_id == "equiv_error":
                raise RuntimeError("workflow exploded")
            result = {
                "status": "DONE" if request.request_id == "equiv_done" else "HUMAN_INTERVENE",
                "request_id": request.request_id,
            }
            if result["status"] == "HUMAN_INTERVENE":
                result["question"] = "Need budget"
            self.sessions[request.request_id] = {"request": request, "status": result["status"], "result": result}
            return result

    monkeypatch.setattr(main, "ThreeProvinceTravelSystem", OutcomeSystem)
    client = TestClient(main.app, raise_server_exceptions=False)

    for request_id in ("equiv_done", "equiv_human"):
        payload = make_request(request_id, total_budget=1000).model_dump(mode="json")
        normal = client.post("/plan", json=payload)
        streamed = client.post("/plan/stream", json=payload)
        stream_events = parse_sse_events(streamed.text)
        result_event = next(data for name, data in stream_events if name == "result")

        assert normal.status_code == 200
        assert streamed.status_code == 200
        assert result_event["status"] == normal.json()["status"]
        assert result_event["request_id"] == normal.json()["request_id"]

    error_payload = make_request("equiv_error", total_budget=1000).model_dump(mode="json")
    normal_error = client.post("/plan", json=error_payload)
    streamed_error = client.post("/plan/stream", json=error_payload)
    error_event = next(data for name, data in parse_sse_events(streamed_error.text) if name == "error")

    assert normal_error.status_code == 500
    assert streamed_error.status_code == 200
    assert normal_error.json()["detail"]["error"] == "planning_failed"
    assert error_event["error"] == "planning_failed"
    assert error_event["message"] == "Planning workflow failed"


def test_dashboard_returns_conflict_for_corrupt_session(tmp_path, monkeypatch):
    store = JsonSessionStore(tmp_path)
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(main, "system", ThreeProvinceTravelSystem(session_store=store))

    client = TestClient(main.app, raise_server_exceptions=False)
    response = client.get("/dashboard/broken")

    assert response.status_code == 409
    assert response.json()["detail"] == "Stored session is corrupt"


def test_dashboard_exposes_resume_metadata(tmp_path, monkeypatch):
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
            resume_state=make_resume_state(request),
            result={"status": "HUMAN_INTERVENE", "question": "Need budget"},
        )
    )
    monkeypatch.setattr(main, "system", ThreeProvinceTravelSystem(session_store=store))

    client = TestClient(main.app, raise_server_exceptions=False)
    response = client.get("/dashboard/dashboard_resume")

    assert response.status_code == 200
    data = response.json()
    assert data["resume_mode"] == "boundary"
    assert data["resume_state"]["mode"] == "boundary"
    assert data["resume_state"]["next_node"] == "zhongshu_itinerary"


def test_resume_api_response_includes_contract_resume_fields(monkeypatch):
    request = make_request("resume_contract", total_budget=None)
    main._shared_sessions.set(
        "resume_contract",
        {
            "request": request,
            "status": "HUMAN_INTERVENE",
            "resume_mode": "boundary",
            "resume_state": make_resume_state(request),
        },
    )

    class ResumeFakeSystem:
        def __init__(self, artifact_dir=None, progress_reporter=None):
            self.sessions = {}

        async def plan_trip(self, *args, **kwargs):
            raise AssertionError("resume endpoint should use boundary resume, not plan replay")

        async def resume_trip(self, request_id, payload):
            result = {
                "status": "HUMAN_INTERVENE",
                "request_id": request_id,
                "question": "Need budget",
                "resume_mode": "boundary",
                "resume_state": make_resume_state(request),
            }
            self.sessions[request_id] = {
                "request": request,
                "status": "HUMAN_INTERVENE",
                "result": result,
                "resume_mode": "boundary",
                "resume_state": result["resume_state"],
            }
            return result

    monkeypatch.setattr(main, "ThreeProvinceTravelSystem", ResumeFakeSystem)
    client = TestClient(main.app, raise_server_exceptions=False)

    response = client.post(
        "/resume/resume_contract",
        json={"profile_updates": {"total_budget": 2500}},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HUMAN_INTERVENE"
    assert data["resume_mode"] == "boundary"
    assert data["resume_state"]["mode"] == "boundary"


def test_resume_read_cleans_shared_cache_before_session_lookup(monkeypatch):
    class CleanupSpyCache:
        def __init__(self):
            self.cleanup_calls = 0

        def cleanup_expired(self):
            self.cleanup_calls += 1

        def get(self, request_id):
            return None

    class MissingSessionSystem:
        def _load_session(self, request_id):
            return None

    cache = CleanupSpyCache()
    monkeypatch.setattr(main, "_shared_sessions", cache)
    monkeypatch.setattr(main, "system", MissingSessionSystem())
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

    class DashboardSystem:
        def dashboard_snapshot(self, request_id):
            return {"request_id": request_id, "status": "HUMAN_INTERVENE"}

    cache = CleanupSpyCache()
    monkeypatch.setattr(main, "_shared_sessions", cache)
    monkeypatch.setattr(main, "system", DashboardSystem())
    client = TestClient(main.app, raise_server_exceptions=False)

    response = client.get("/dashboard/cache_cleanup_dashboard")

    assert response.status_code == 200
    assert response.json()["request_id"] == "cache_cleanup_dashboard"
    assert cache.cleanup_calls == 1


def test_resume_stream_emits_labeled_resume_events(monkeypatch):
    request = make_request("stream_resume_contract", total_budget=None)
    main._shared_sessions.set(
        "stream_resume_contract",
        {
            "request": request,
            "status": "HUMAN_INTERVENE",
            "resume_mode": "boundary",
            "resume_state": make_resume_state(request),
        },
    )

    class ResumeStreamingFakeSystem:
        def __init__(self, artifact_dir=None, progress_reporter=None):
            self.progress_reporter = progress_reporter
            self.sessions = {}

        async def plan_trip(self, *args, **kwargs):
            raise AssertionError("streaming resume should use resume_trip")

        async def resume_trip(self, request_id, payload):
            if self.progress_reporter:
                self.progress_reporter(f"resuming {request_id} with boundary")
            result = {
                "status": "HUMAN_INTERVENE",
                "request_id": request_id,
                "question": "Need budget",
                "resume_mode": "boundary",
                "resume_state": make_resume_state(request),
            }
            self.sessions[request_id] = {
                "request": request,
                "status": "HUMAN_INTERVENE",
                "result": result,
                "resume_mode": "boundary",
                "resume_state": result["resume_state"],
            }
            return result

    monkeypatch.setattr(main, "ThreeProvinceTravelSystem", ResumeStreamingFakeSystem)
    client = TestClient(main.app, raise_server_exceptions=False)

    response = client.post(
        "/resume/stream_resume_contract/stream",
        json={"profile_updates": {"total_budget": 2500}},
    )

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    result = next(data for name, data in events if name == "result")
    progress = [data["line"] for name, data in events if name == "progress"]
    assert result["resume_mode"] == "boundary"
    assert result["resume_state"]["mode"] == "boundary"
    assert any("boundary" in line for line in progress)
