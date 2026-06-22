from __future__ import annotations

import asyncio
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from utils.schemas import FinalTravelPackageModel, PlanningRequest
from utils.markdown_formatter import format_package_to_markdown
from utils.path_safety import validate_request_id
from utils.session_cache import InMemorySessionCache
from utils.session_store import CorruptSessionError, JsonSessionStore, SessionStore, StoredSession
from utils.settings import get_settings
from workflow import ProvinceWorkflow


class HumanResumePayload(BaseModel):
    profile_updates: dict[str, Any] = Field(default_factory=dict)
    user_message: str | None = None


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

    async def resume_trip(self, request_id: str, payload: HumanResumePayload) -> FinalTravelPackageModel | dict[str, Any]:
        request_id = validate_request_id(request_id)
        session = self.sessions.get(request_id) or self._load_session(request_id)
        if not session:
            raise KeyError(f"Unknown request_id: {request_id}")
        request: PlanningRequest = session["request"]
        resume_state = session.get("resume_state") or {}
        if resume_state.get("mode") == "boundary":
            try:
                result = await self.workflow.resume(resume_state, payload)
                effective_request = self._merge_request(request, payload)
                return self._record_workflow_result(
                    effective_request,
                    result,
                    resume_mode="boundary",
                    resume_state=resume_state,
                    include_resume_metadata=True,
                )
            except (KeyError, TypeError, ValueError):
                pass
        replay_reason = "Stored session has no compatible boundary resume state."
        result = await self.workflow.run(self._merge_request(request, payload))
        replay_state = {
            "mode": "replay",
            "next_node": None,
            "state": {},
            "question": session.get("result", {}).get("question"),
            "created_at": datetime.now().astimezone().isoformat(),
            "replay_reason": replay_reason,
        }
        return self._record_workflow_result(
            self._merge_request(request, payload),
            result,
            resume_mode="replay",
            resume_state=replay_state,
            replay_reason=replay_reason,
            include_resume_metadata=True,
        )

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
            "replay_reason": resume_state.get("replay_reason") or session.get("replay_reason"),
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
        replay_reason: str | None = None,
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
        if replay_reason:
            session["replay_reason"] = replay_reason
        self.sessions[request.request_id] = session
        if result.get("status") == "DONE":
            package = FinalTravelPackageModel.model_validate(result["final_package"])
            session["package"] = package
            self._persist_session(request.request_id, session)
            if include_resume_metadata:
                response = package.model_dump(mode="json")
                response.update(self._resume_response_fields(effective_resume_mode, effective_resume_state, replay_reason))
                return response
            return package
        self._persist_session(request.request_id, session)
        if result.get("status") == "REJECTED":
            response = dict(result.get("rejected_payload", result))
            if include_resume_metadata:
                response.update(self._resume_response_fields(effective_resume_mode, effective_resume_state, replay_reason))
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
            response.update(self._resume_response_fields(effective_resume_mode, effective_resume_state, replay_reason))
        return response

    def _resume_response_fields(
        self,
        resume_mode: str,
        resume_state: dict[str, Any] | None,
        replay_reason: str | None,
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "resume_mode": resume_mode,
            "resume_state": resume_state or {},
        }
        if replay_reason:
            fields["replay_reason"] = replay_reason
        return fields


DEFAULT_ARTIFACT_DIR = Path(get_settings().output_dir)

# Shared session store across all per-request ThreeProvinceTravelSystem instances
_shared_sessions = InMemorySessionCache(
    max_entries=get_settings().session_cache_max_entries,
    ttl_seconds=get_settings().session_cache_ttl_seconds,
)


def console_progress_reporter(line: str) -> None:
    print(line, flush=True)


app = FastAPI(title="Three Provinces Six Bureaus Travel Planner", version="2.1.0")

_HERE = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=_HERE / "static"), name="static")
templates = Jinja2Templates(directory=_HERE / "templates")

system = ThreeProvinceTravelSystem(artifact_dir=DEFAULT_ARTIFACT_DIR)


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
    planner = create_travel_system(progress_reporter=progress_reporter)
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
    planner = create_travel_system(progress_reporter=progress_reporter)
    safe_request_id = validate_request_id(request_id)
    planner.sessions[safe_request_id] = session
    result = await planner.resume_trip(safe_request_id, payload)
    _cache_planner_sessions(planner)
    return result


def _load_resume_session(request_id: str) -> dict[str, Any] | None:
    safe_request_id = validate_request_id(request_id)
    _cleanup_shared_sessions()
    return _shared_sessions.get(safe_request_id) or system._load_session(safe_request_id)


def _sse_event(name: str, data: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def _stream_execution(
    request: PlanningRequest,
    *,
    human_resume: HumanResumePayload | None = None,
) -> StreamingResponse:
    queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()

    def progress_reporter(line: str) -> None:
        queue.put_nowait(("progress", {"line": line}))

    async def run_workflow() -> None:
        try:
            result = await _execute_plan_request(
                request,
                progress_reporter=progress_reporter,
                human_resume=human_resume,
            )
            queue.put_nowait(("result", _result_to_jsonable(result)))
        except Exception:
            queue.put_nowait(("error", _planning_failure_detail()))
        finally:
            queue.put_nowait(None)

    async def event_generator():
        task = asyncio.create_task(run_workflow())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    yield _sse_event("done", {})
                    break
                name, data = event
                yield _sse_event(name, data)
        except asyncio.CancelledError:
            task.cancel()
            raise

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _stream_resume_execution(request_id: str, payload: HumanResumePayload) -> StreamingResponse:
    queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()

    def progress_reporter(line: str) -> None:
        queue.put_nowait(("progress", {"line": line}))

    async def run_workflow() -> None:
        try:
            result = await _execute_resume_request(
                request_id,
                payload,
                progress_reporter=progress_reporter,
            )
            queue.put_nowait(("result", _result_to_jsonable(result)))
        except Exception:
            queue.put_nowait(("error", _planning_failure_detail()))
        finally:
            queue.put_nowait(None)

    async def event_generator():
        task = asyncio.create_task(run_workflow())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    yield _sse_event("done", {})
                    break
                name, data = event
                yield _sse_event(name, data)
        except asyncio.CancelledError:
            task.cancel()
            raise

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "three-provinces-six-bureaus"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/plan/stream")
async def plan_stream(request: PlanningRequest):
    return _stream_execution(request)


@app.post("/resume/{request_id}/stream")
async def resume_stream(request_id: str, payload: HumanResumePayload):
    try:
        session = _load_resume_session(request_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown request_id: {request_id}") from exc
    except CorruptSessionError as exc:
        raise HTTPException(status_code=409, detail="Stored session is corrupt") from exc
    if not session:
        raise HTTPException(status_code=404, detail=f"Unknown request_id: {request_id}")

    return _stream_resume_execution(request_id, payload)


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
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/dashboard/{request_id}")
async def dashboard(request_id: str) -> dict[str, Any]:
    try:
        _cleanup_shared_sessions()
        return system.dashboard_snapshot(request_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown request_id: {request_id}") from exc
    except CorruptSessionError as exc:
        raise HTTPException(status_code=409, detail="Stored session is corrupt") from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown request_id: {request_id}") from exc


async def demo() -> None:
    demo_system = ThreeProvinceTravelSystem(
        artifact_dir=DEFAULT_ARTIFACT_DIR,
        progress_reporter=console_progress_reporter,
    )
    request = PlanningRequest.model_validate(
        {
            "request_id": "demo_tokyo_three_provinces", 
            "user_message": "Plan a highly structured Tokyo trip with clear logistics, review gates, and calendar output.", 
            "profile": 
            {
                "origin_city": "Beijing", 
                "origin_airport_code": "", 
                "destination_preferences": ["Tokyo"], 
                "destination_airport_code": "HND", 
                "start_date": date(2026, 4, 18), 
                "end_date": date(2026, 4, 21), 
                "adults": 3, 
                "budget_level": "mid_range", 
                "total_budget": 2000, 
                "currency": "USD", 
                "interests": ["food", "culture", "city walks"], 
                "constraints": ["prefer predictable transfers", "need calendar-ready schedule"], 
                "pace": "structured"
            }
        })
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [INFO] demo | 开始运行 main.py 示例", flush=True)
    result = await demo_system.plan_trip(request)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [INFO] demo | 最终输出如下", flush=True)
    if isinstance(result, FinalTravelPackageModel):
        markdown_output = format_package_to_markdown(result.model_dump(mode="python"))
        Path("travel_plan.md").write_text(markdown_output, encoding="utf-8")
        print(f"\n[INFO] Markdown 文件已保存至: travel_plan.md", flush=True)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    asyncio.run(demo())
