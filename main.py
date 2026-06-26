from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from utils.schemas import FinalTravelPackageModel, PlanningRequest
from utils.path_safety import validate_request_id
from utils.session_cache import InMemorySessionCache
from utils.session_store import CorruptSessionError, JsonSessionStore, SessionStore, StoredSession
from utils.settings import get_settings
from workflow import ProvinceWorkflow

logger = logging.getLogger(__name__)


class HumanResumePayload(BaseModel):
    profile_updates: dict[str, Any] = Field(default_factory=dict)
    user_message: str | None = None


class BoundaryResumeConflict(RuntimeError):
    pass


class ThreeProvinceTravelSystem:
    def __init__(
        self,
        artifact_dir: str | Path | None = None,
        progress_reporter: Callable[[str], None] | None = None,
        session_store: SessionStore | None = None,
    ) -> None:
        self.workflow = ProvinceWorkflow(artifact_dir=artifact_dir, progress_reporter=progress_reporter)
        self.sessions: dict[str, dict[str, Any]] = {}
        self.session_store = session_store or JsonSessionStore(get_settings().session_store_dir)

    async def plan_trip(self, request: PlanningRequest, human_resume: HumanResumePayload | None = None) -> FinalTravelPackageModel | dict[str, Any]:
        effective_request = self._merge_request(request, human_resume)
        result = await self.workflow.run(effective_request)
        return self._record_workflow_result(effective_request, result)

    async def stream_plan_trip(self, request: PlanningRequest, human_resume: HumanResumePayload | None = None):
        effective_request = self._merge_request(request, human_resume)
        async for event in self.workflow.stream_run(effective_request):
            if event.get("event") == "result":
                yield {"event": "result", "data": _result_to_jsonable(self._record_workflow_result(effective_request, event["data"]))}
            else:
                yield event

    async def resume_trip(self, request_id: str, payload: HumanResumePayload) -> FinalTravelPackageModel | dict[str, Any]:
        validate_request_id(request_id)
        session = self.sessions.get(request_id) or self._load_session(request_id)
        if not session:
            raise KeyError(request_id)
        if session.get("resume_mode") != "boundary" or not session.get("resume_state", {}).get("thread_id"):
            raise BoundaryResumeConflict("Stored session has no boundary checkpoint metadata; resume cannot continue.")
        result = await self.workflow.resume(session["resume_state"], payload)
        effective_request = PlanningRequest.model_validate(result.get("request") or session["request"].model_dump(mode="python"))
        return self._record_workflow_result(effective_request, result, include_resume_metadata=True)

    async def stream_resume_trip(self, request_id: str, payload: HumanResumePayload):
        validate_request_id(request_id)
        session = self.sessions.get(request_id) or self._load_session(request_id)
        if not session:
            raise KeyError(request_id)
        if session.get("resume_mode") != "boundary" or not session.get("resume_state", {}).get("thread_id"):
            raise BoundaryResumeConflict("Stored session has no boundary checkpoint metadata; resume cannot continue.")
        async for event in self.workflow.stream_resume(session["resume_state"], payload):
            if event.get("event") == "result":
                effective_request = PlanningRequest.model_validate(event["data"].get("request") or session["request"].model_dump(mode="python"))
                yield {"event": "result", "data": _result_to_jsonable(self._record_workflow_result(effective_request, event["data"], include_resume_metadata=True))}
            else:
                yield event

    def dashboard_snapshot(self, request_id: str) -> dict[str, Any]:
        request_id = validate_request_id(request_id)
        session = self.sessions.get(request_id) or self._load_session(request_id)
        if not session:
            raise KeyError(request_id)
        context = session.get("context")
        context_snapshot = session.get("context_snapshot") or self._context_snapshot(context)
        resume_state = session.get("resume_state") or {}
        return {
            "request_id": request_id,
            "status": session.get("status"),
            "current_state": context_snapshot.get("current_state"),
            "pending_user_inputs": context_snapshot.get("pending_user_inputs", []),
            "progress_events": context_snapshot.get("progress_events", []),
            "has_package": "package" in session and session.get("package") is not None,
            "resume_mode": session.get("resume_mode", "none"),
            "resume_state": resume_state,

        }

    def _merge_request(self, request: PlanningRequest, human_resume: HumanResumePayload | None) -> PlanningRequest:
        if human_resume is None:
            return request
        data = request.model_dump(mode="python")
        profile = dict(data["profile"])
        profile.update(human_resume.profile_updates)
        data["profile"] = profile
        if human_resume.user_message:
            data["user_message"] = human_resume.user_message
        return PlanningRequest.model_validate(data)

    def _persist_session(self, request_id: str, session: dict[str, Any]) -> None:
        package = session.get("package")
        self.session_store.save(
            StoredSession(
                request_id=request_id,
                request=session["request"].model_dump(mode="json"),
                status=session.get("status", "UNKNOWN"),
                context_snapshot=self._context_snapshot(session.get("context")),
                result=self._serializable_result(session.get("result", {})),
                package=package.model_dump(mode="json") if hasattr(package, "model_dump") else package,
                resume_state=session.get("resume_state") or {},
                resume_mode=session.get("resume_mode", "none"),
            )
        )

    def _load_session(self, request_id: str) -> dict[str, Any] | None:
        stored = self.session_store.load(request_id)
        if stored is None:
            return None
        session: dict[str, Any] = {
            "request": PlanningRequest.model_validate(stored.request),
            "status": stored.status,
            "context_snapshot": stored.context_snapshot,
            "result": stored.result,
            "resume_state": stored.resume_state,
            "resume_mode": stored.resume_mode,
        }
        if stored.package is not None:
            session["package"] = FinalTravelPackageModel.model_validate(stored.package)
        self.sessions[request_id] = session
        return session

    def _context_snapshot(self, context: Any | None) -> dict[str, Any]:
        if context is None:
            return {}
        current_state = getattr(getattr(context, "current_state", None), "value", None)
        return {
            "current_state": current_state,
            "pending_user_inputs": list(getattr(context, "pending_user_inputs", []) or []),
            "progress_events": list(getattr(context, "progress_events", []) or []),
        }

    def _serializable_result(self, result: dict[str, Any]) -> dict[str, Any]:
        serialized = dict(result)
        serialized.pop("context", None)
        return serialized

    def _record_workflow_result(
        self,
        request: PlanningRequest,
        result: dict[str, Any],
        *,
        resume_mode: str | None = None,
        resume_state: dict[str, Any] | None = None,

        include_resume_metadata: bool = False,
    ) -> FinalTravelPackageModel | dict[str, Any]:
        effective_resume_mode = resume_mode or result.get("resume_mode") or "none"
        effective_resume_state = resume_state if resume_state is not None else result.get("resume_state", {})
        session = {
            "request": request,
            "context": result.get("context"),
            "status": result.get("status", "UNKNOWN"),
            "result": result,
            "resume_mode": effective_resume_mode,
            "resume_state": effective_resume_state or {},
        }
        self.sessions[request.request_id] = session
        if result.get("status") == "DONE":
            package = FinalTravelPackageModel.model_validate(result["final_package"])
            session["package"] = package
            self._persist_session(request.request_id, session)
            if include_resume_metadata:
                response = package.model_dump(mode="json")
                response.update(self._resume_response_fields(effective_resume_mode, effective_resume_state))
                return response
            return package
        self._persist_session(request.request_id, session)
        if result.get("status") == "REJECTED":
            response = dict(result.get("rejected_payload", result))
            if include_resume_metadata:
                response.update(self._resume_response_fields(effective_resume_mode, effective_resume_state))
            return response
        context = result.get("context")
        response = {
            "status": result.get("status", "HUMAN_INTERVENE"),
            "request_id": request.request_id,
            "question": result.get("question"),
            "dashboard_url": self.workflow.orchestrator.build_dashboard_link(context) if context else None,
            "progress_events": context.progress_events if context else result.get("progress_events", []),
        }
        if result.get("resume_mode") or include_resume_metadata:
            response.update(self._resume_response_fields(effective_resume_mode, effective_resume_state))
        return response

    def _resume_response_fields(
        self,
        resume_mode: str,
        resume_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "resume_mode": resume_mode,
            "resume_state": resume_state or {},
        }
        return fields


DEFAULT_ARTIFACT_DIR = Path(get_settings().output_dir)

# Shared session store across all per-request ThreeProvinceTravelSystem instances
_shared_sessions = InMemorySessionCache(
    max_entries=get_settings().session_cache_max_entries,
    ttl_seconds=get_settings().session_cache_ttl_seconds,
)


def console_progress_reporter(line: str) -> None:
    print(line, flush=True)


def logger_progress_reporter(line: str) -> None:
    logger.info("workflow_progress | %s", line)


app = FastAPI(title="Three Provinces Six Bureaus Travel Planner", version="2.1.0")

_HERE = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=_HERE / "static"), name="static")
templates = Jinja2Templates(directory=_HERE / "templates")

class SessionLookup:
    def __init__(self, session_store: SessionStore | None = None) -> None:
        self.session_store = session_store or JsonSessionStore(get_settings().session_store_dir)

    def _load_session(self, request_id: str) -> dict[str, Any] | None:
        stored = self.session_store.load(request_id)
        if stored is None:
            return None
        session: dict[str, Any] = {
            "request": PlanningRequest.model_validate(stored.request),
            "status": stored.status,
            "context_snapshot": stored.context_snapshot,
            "result": stored.result,
            "resume_state": stored.resume_state,
            "resume_mode": stored.resume_mode,
        }
        if stored.package is not None:
            session["package"] = FinalTravelPackageModel.model_validate(stored.package)
        return session

    def dashboard_snapshot(self, request_id: str) -> dict[str, Any]:
        request_id = validate_request_id(request_id)
        session = _shared_sessions.get(request_id) or self._load_session(request_id)
        if not session:
            raise KeyError(request_id)
        context_snapshot = session.get("context_snapshot") or {}
        resume_state = session.get("resume_state") or {}
        return {
            "request_id": request_id,
            "status": session.get("status"),
            "current_state": context_snapshot.get("current_state"),
            "pending_user_inputs": context_snapshot.get("pending_user_inputs", []),
            "progress_events": context_snapshot.get("progress_events", []),
            "has_package": "package" in session and session.get("package") is not None,
            "resume_mode": session.get("resume_mode", "none"),
            "resume_state": resume_state,

        }


session_lookup = SessionLookup()


def create_travel_system(progress_reporter: Callable[[str], None] | None = None) -> ThreeProvinceTravelSystem:
    return ThreeProvinceTravelSystem(
        artifact_dir=DEFAULT_ARTIFACT_DIR,
        progress_reporter=progress_reporter,
    )


def _planning_failure_detail() -> dict[str, str]:
    return {"error": "planning_failed", "message": "Planning workflow failed"}


def _result_to_jsonable(result: FinalTravelPackageModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(result, FinalTravelPackageModel):
        return result.model_dump(mode="json")
    return result


def _cache_planner_sessions(planner: Any) -> None:
    for request_id, session in getattr(planner, "sessions", {}).items():
        _shared_sessions.set(request_id, session)


def _cleanup_shared_sessions() -> None:
    cleanup = getattr(_shared_sessions, "cleanup_expired", None)
    if callable(cleanup):
        cleanup()


async def _execute_plan_request(
    request: PlanningRequest,
    *,
    progress_reporter: Callable[[str], None] | None = None,
    human_resume: HumanResumePayload | None = None,
) -> FinalTravelPackageModel | dict[str, Any]:
    planner = create_travel_system(progress_reporter=progress_reporter or logger_progress_reporter)
    if human_resume is None:
        result = await planner.plan_trip(request)
    else:
        result = await planner.plan_trip(request, human_resume=human_resume)
    _cache_planner_sessions(planner)
    return result


async def _execute_resume_request(
    request_id: str,
    payload: HumanResumePayload,
    *,
    progress_reporter: Callable[[str], None] | None = None,
) -> FinalTravelPackageModel | dict[str, Any]:
    session = _load_resume_session(request_id)
    if not session:
        raise KeyError(f"Unknown request_id: {request_id}")
    planner = create_travel_system(progress_reporter=progress_reporter or logger_progress_reporter)
    safe_request_id = validate_request_id(request_id)
    planner.sessions[safe_request_id] = session
    result = await planner.resume_trip(safe_request_id, payload)
    _cache_planner_sessions(planner)
    return result


def _load_resume_session(request_id: str) -> dict[str, Any] | None:
    safe_request_id = validate_request_id(request_id)
    _cleanup_shared_sessions()
    return _shared_sessions.get(safe_request_id) or session_lookup._load_session(safe_request_id)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "three-provinces-six-bureaus"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/plan/stream")
async def plan_stream(request: PlanningRequest):
    return StreamingResponse(
        _sse_plan_events(request),
        media_type="text/event-stream",
    )


@app.post("/resume/{request_id}/stream")
async def resume_stream(request_id: str, payload: HumanResumePayload):
    return StreamingResponse(
        _sse_resume_events(request_id, payload),
        media_type="text/event-stream",
    )


@app.get("/download/{request_id:path}")
async def download_artifact(request_id: str):
    try:
        safe_id = validate_request_id(request_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Artifact not found") from exc
    md_path = DEFAULT_ARTIFACT_DIR / f"{safe_id}_travel_plan.md"
    if md_path.exists():
        return FileResponse(md_path, filename=md_path.name, media_type="text/markdown")
    ics_path = DEFAULT_ARTIFACT_DIR / f"{safe_id}_trip_calendar.ics"
    if ics_path.exists():
        return FileResponse(ics_path, filename=ics_path.name, media_type="text/calendar")
    raise HTTPException(status_code=404, detail="Artifact not found")


@app.post("/plan")
async def plan_trip(request: PlanningRequest) -> Any:
    try:
        return await _execute_plan_request(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_planning_failure_detail()) from exc


@app.post("/resume/{request_id}")
async def resume_trip(request_id: str, payload: HumanResumePayload) -> Any:
    try:
        return await _execute_resume_request(request_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown request_id: {request_id}") from exc
    except CorruptSessionError as exc:
        raise HTTPException(status_code=409, detail="Stored session is corrupt") from exc
    except BoundaryResumeConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/dashboard/{request_id}")
async def dashboard(request_id: str) -> dict[str, Any]:
    try:
        _cleanup_shared_sessions()
        return session_lookup.dashboard_snapshot(request_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown request_id: {request_id}") from exc
    except CorruptSessionError as exc:
        raise HTTPException(status_code=409, detail="Stored session is corrupt") from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown request_id: {request_id}") from exc


def _sse_format(event: str, payload: dict[str, Any]) -> str:
    import json

    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


async def _sse_plan_events(request: PlanningRequest):
    planner = create_travel_system(progress_reporter=logger_progress_reporter)
    try:
        yield _sse_format("progress", {"message": "planning_started", "request_id": request.request_id})
        async for event in planner.stream_plan_trip(request):
            yield _sse_format(event["event"], event.get("data", {}))
        _cache_planner_sessions(planner)
    except Exception as exc:
        yield _sse_format("error", {"error": type(exc).__name__, "message": str(exc)})
    finally:
        yield _sse_format("done", {"request_id": request.request_id})


async def _sse_resume_events(request_id: str, payload: HumanResumePayload):
    safe_request_id = validate_request_id(request_id)
    try:
        session = _load_resume_session(safe_request_id)
        if not session:
            raise KeyError(f"Unknown request_id: {safe_request_id}")
        planner = create_travel_system(progress_reporter=logger_progress_reporter)
        planner.sessions[safe_request_id] = session
        yield _sse_format("progress", {"message": "resume_started", "request_id": safe_request_id})
        async for event in planner.stream_resume_trip(safe_request_id, payload):
            yield _sse_format(event["event"], event.get("data", {}))
        _cache_planner_sessions(planner)
    except Exception as exc:
        yield _sse_format("error", {"error": type(exc).__name__, "message": str(exc)})
    finally:
        yield _sse_format("done", {"request_id": safe_request_id})
