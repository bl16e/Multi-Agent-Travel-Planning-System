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
from workflow import ProvinceWorkflow


class HumanResumePayload(BaseModel):
    profile_updates: dict[str, Any] = Field(default_factory=dict)
    user_message: str | None = None


class ThreeProvinceTravelSystem:
    def __init__(
        self,
        artifact_dir: str | Path | None = None,
        progress_reporter: Callable[[str], None] | None = None,
    ) -> None:
        self.workflow = ProvinceWorkflow(artifact_dir=artifact_dir, progress_reporter=progress_reporter)
        self.sessions: dict[str, dict[str, Any]] = {}

    async def plan_trip(self, request: PlanningRequest, human_resume: HumanResumePayload | None = None) -> FinalTravelPackageModel | dict[str, Any]:
        effective_request = self._merge_request(request, human_resume)
        result = await self.workflow.run(effective_request)
        self.sessions[effective_request.request_id] = {"request": effective_request, "context": result.get("context"), "status": result.get("status", "UNKNOWN"), "result": result}
        if result.get("status") == "DONE":
            package = FinalTravelPackageModel.model_validate(result["final_package"])
            self.sessions[effective_request.request_id]["package"] = package
            return package
        if result.get("status") == "REJECTED":
            return result.get("rejected_payload", result)
        return {"status": result.get("status", "HUMAN_INTERVENE"), "request_id": effective_request.request_id, "question": result.get("question"), "dashboard_url": self.workflow.orchestrator.build_dashboard_link(result["context"]), "progress_events": result["context"].progress_events if result.get("context") else []}

    async def resume_trip(self, request_id: str, payload: HumanResumePayload) -> FinalTravelPackageModel | dict[str, Any]:
        session = self.sessions.get(request_id)
        if not session:
            raise KeyError(f"Unknown request_id: {request_id}")
        request: PlanningRequest = session["request"]
        return await self.plan_trip(request, human_resume=payload)

    def dashboard_snapshot(self, request_id: str) -> dict[str, Any]:
        session = self.sessions.get(request_id)
        if not session:
            raise KeyError(request_id)
        context = session.get("context")
        return {"request_id": request_id, "status": session.get("status"), "current_state": context.current_state.value if context else None, "pending_user_inputs": context.pending_user_inputs if context else [], "progress_events": context.progress_events if context else [], "has_package": "package" in session}

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


DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().with_name("artifacts")

# Shared session store across all per-request ThreeProvinceTravelSystem instances
_shared_sessions: dict[str, dict[str, Any]] = {}


def console_progress_reporter(line: str) -> None:
    print(line, flush=True)


app = FastAPI(title="Three Provinces Six Bureaus Travel Planner", version="2.1.0")

_HERE = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=_HERE / "static"), name="static")
templates = Jinja2Templates(directory=_HERE / "templates")

system = ThreeProvinceTravelSystem(artifact_dir=DEFAULT_ARTIFACT_DIR)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "three-provinces-six-bureaus"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/plan/stream")
async def plan_stream(request: PlanningRequest):
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def progress_reporter(line: str) -> None:
        queue.put_nowait(line)

    async def run_workflow():
        stream_system = ThreeProvinceTravelSystem(
            artifact_dir=DEFAULT_ARTIFACT_DIR,
            progress_reporter=progress_reporter,
        )
        try:
            result = await stream_system.plan_trip(request)
            # Persist session to shared store for resume
            _shared_sessions.update(stream_system.sessions)
            if isinstance(result, FinalTravelPackageModel):
                data = result.model_dump(mode="json")
                queue.put_nowait(f"__RESULT__:{json.dumps(data, ensure_ascii=False, default=str)}")
            else:
                queue.put_nowait(f"__RESULT__:{json.dumps(result, ensure_ascii=False, default=str)}")
        except Exception as e:
            queue.put_nowait(f"__ERROR__:{str(e)}")
        finally:
            queue.put_nowait(None)

    async def event_generator():
        task = asyncio.create_task(run_workflow())
        try:
            while True:
                msg = await queue.get()
                if msg is None:
                    yield "event: done\ndata: {}\n\n"
                    break
                if msg.startswith("__RESULT__:"):
                    yield f"event: result\ndata: {msg[11:]}\n\n"
                elif msg.startswith("__ERROR__:"):
                    yield f"event: error\ndata: {json.dumps({'error': msg[10:]})}\n\n"
                else:
                    yield f"event: progress\ndata: {json.dumps({'line': msg}, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            task.cancel()
            raise

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/resume/{request_id}/stream")
async def resume_stream(request_id: str, payload: HumanResumePayload):
    session = _shared_sessions.get(request_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Unknown request_id: {request_id}")

    original_request: PlanningRequest = session["request"]
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def progress_reporter(line: str) -> None:
        queue.put_nowait(line)

    async def run_workflow():
        stream_system = ThreeProvinceTravelSystem(
            artifact_dir=DEFAULT_ARTIFACT_DIR,
            progress_reporter=progress_reporter,
        )
        try:
            result = await stream_system.plan_trip(original_request, human_resume=payload)
            _shared_sessions.update(stream_system.sessions)
            if isinstance(result, FinalTravelPackageModel):
                data = result.model_dump(mode="json")
                queue.put_nowait(f"__RESULT__:{json.dumps(data, ensure_ascii=False, default=str)}")
            else:
                queue.put_nowait(f"__RESULT__:{json.dumps(result, ensure_ascii=False, default=str)}")
        except Exception as e:
            queue.put_nowait(f"__ERROR__:{str(e)}")
        finally:
            queue.put_nowait(None)

    async def event_generator():
        task = asyncio.create_task(run_workflow())
        try:
            while True:
                msg = await queue.get()
                if msg is None:
                    yield "event: done\ndata: {}\n\n"
                    break
                if msg.startswith("__RESULT__:"):
                    yield f"event: result\ndata: {msg[11:]}\n\n"
                elif msg.startswith("__ERROR__:"):
                    yield f"event: error\ndata: {json.dumps({'error': msg[10:]})}\n\n"
                else:
                    yield f"event: progress\ndata: {json.dumps({'line': msg}, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            task.cancel()
            raise

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/download/{request_id}")
async def download_artifact(request_id: str):
    safe_id = request_id.replace("/", "").replace("\\", "").replace("..", "")
    md_path = DEFAULT_ARTIFACT_DIR / f"{safe_id}_travel_plan.md"
    if md_path.exists():
        return FileResponse(md_path, filename=md_path.name, media_type="text/markdown")
    ics_path = DEFAULT_ARTIFACT_DIR / f"{safe_id}_trip_calendar.ics"
    if ics_path.exists():
        return FileResponse(ics_path, filename=ics_path.name, media_type="text/calendar")
    raise HTTPException(status_code=404, detail="Artifact not found")


@app.post("/plan")
async def plan_trip(request: PlanningRequest) -> Any:
    return await system.plan_trip(request)


@app.post("/resume/{request_id}")
async def resume_trip(request_id: str, payload: HumanResumePayload) -> Any:
    try:
        return await system.resume_trip(request_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/dashboard/{request_id}")
async def dashboard(request_id: str) -> dict[str, Any]:
    try:
        return system.dashboard_snapshot(request_id)
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
