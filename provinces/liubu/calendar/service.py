from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from provinces.liubu.constrained.state import LiubuWorkerInput
from provinces.liubu.official_tooling import EvidenceToolNode, bind_tools_if_available, invoke_bound_tool_model, load_allowed_liubu_tools, run_tool_node_collect_evidence
from utils.agent_runtime import escape_prompt_template_text, soul_path_for
from utils.icalendar_utils import build_ics_calendar
from utils.path_safety import sanitize_request_id
from utils.schemas import CalendarEventListModel, CalendarEventModel, CalendarExecutionResult
from utils.llm_factory import build_qwen_chat
from utils.settings import get_settings

logger = logging.getLogger(__name__)
STRUCTURED_SYNTHESIS_TIMEOUT_SECONDS: float | None = None
CALENDAR_ALLOWED_TOOLS = {"search_google_maps", "search_google_maps_directions", "search_local_places"}
CALENDAR_TOOL_SERVERS = ["serpapi"]


class CalendarState(TypedDict, total=False):
    worker_input: LiubuWorkerInput | dict[str, Any]
    messages: Annotated[list[Any], add_messages]
    destination: str
    daily_plan: list[dict[str, Any]]
    research_notes: str
    tool_evidence: list[dict[str, Any]]
    tool_step_count: int
    events: list[dict[str, Any]]
    calendar_status: str
    calendar_data_source: str
    calendar_warnings: list[str]
    result: dict[str, Any]


class CalendarBureau:
    def __init__(self, output_dir: str = "artifacts") -> None:
        self.output_dir = Path(output_dir)
        self.soul_path = soul_path_for(__file__)
        self.tool_node = EvidenceToolNode([], collector=run_tool_node_collect_evidence)
        self.bound_tool_model = None
        self._tooling_ready = False
        self.graph = self._build_graph()

    async def run(self, subtask: dict[str, Any]) -> dict[str, Any]:
        await self.ensure_live_tooling()
        result = await self.graph.ainvoke(dict(subtask))
        return result["result"]

    async def ensure_live_tooling(self) -> None:
        if self._tooling_ready:
            return
        tools = await load_allowed_liubu_tools(CALENDAR_TOOL_SERVERS, CALENDAR_ALLOWED_TOOLS)
        self.tool_node = EvidenceToolNode(tools, collector=run_tool_node_collect_evidence)
        self.bound_tool_model = bind_tools_if_available(build_qwen_chat(), tools)
        self.graph = self._build_graph()
        self._tooling_ready = True

    def _build_graph(self):
        graph = StateGraph(CalendarState)
        graph.add_node("agent", self.agent)
        graph.add_node("tools", self.tool_node)
        graph.add_node("quality_gate", self.quality_gate)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", self._route_after_agent, {"tools": "tools", "quality_gate": "quality_gate"})
        graph.add_edge("tools", "agent")
        graph.add_edge("quality_gate", END)
        return graph.compile()

    async def agent(self, state: CalendarState) -> dict[str, Any]:
        worker_input = self._worker_input(state)
        if self._daily_plan_has_live_links(worker_input.daily_plan):
            event_result = await self.build_events(
                {
                    "daily_plan": worker_input.daily_plan,
                    "research_notes": "Approved live itinerary activities available.",
                    "force_itinerary_events": True,
                }
            )
            return {
                "destination": worker_input.destination,
                "daily_plan": worker_input.daily_plan,
                "research_notes": "Approved live itinerary activities available.",
                "tool_evidence": [],
                **event_result,
            }
        if self.bound_tool_model is not None and int(state.get("tool_step_count") or 0) == 0:
            tool_message = await invoke_bound_tool_model(
                self.bound_tool_model,
                [HumanMessage(content=f"Search live place and route context for calendar events in {worker_input.destination}.")],
                bureau=worker_input.bureau,
                request_id=worker_input.request_id,
                destination=worker_input.destination,
            )
            if getattr(tool_message, "tool_calls", None):
                return {"worker_input": worker_input, "messages": [tool_message]}
        evidence = list(state.get("tool_evidence", []) or [])
        live_notes = [str(item.get("result")) for item in evidence if item.get("status") == "ok"]
        research_notes = (
            f"No live calendar ToolNode evidence was available for {worker_input.destination}; "
            "using structured synthesis or itinerary-derived fallback events."
        )
        if live_notes:
            research_notes = "\n".join(live_notes)
        event_result = await self.build_events(
            {
                "daily_plan": worker_input.daily_plan,
                "research_notes": research_notes,
            }
        )
        if live_notes and event_result.get("calendar_status") != "error":
            event_result["calendar_status"] = "ok"
            event_result["calendar_data_source"] = "live"
        return {
            "destination": worker_input.destination,
            "daily_plan": worker_input.daily_plan,
            "research_notes": research_notes,
            "tool_evidence": evidence,
            **event_result,
        }

    async def tools(self, state: CalendarState) -> dict[str, Any]:
        return {
            "tool_evidence": list(state.get("tool_evidence", [])),
            "tool_step_count": int(state.get("tool_step_count") or 0) + 1,
        }

    async def quality_gate(self, state: CalendarState) -> dict[str, Any]:
        return await self.write_calendar(state)

    async def build_events(self, state: CalendarState) -> dict[str, Any]:
        if state.get("force_itinerary_events"):
            return self._events_from_daily_plan(state, live=True, initial_warnings=[])
        llm = build_qwen_chat()
        if llm:
            try:
                structured = llm.with_structured_output(CalendarEventListModel)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", escape_prompt_template_text(Path(self.soul_path).read_text(encoding="utf-8"))),
                    ("user", "Daily plan: {daily_plan}\nResearch notes: {research_notes}\nReturn valid JSON structured object with an events list."),
                ])
                event_list = await asyncio.wait_for(
                    (prompt | structured).ainvoke({"daily_plan": str(state.get("daily_plan", [])), "research_notes": state.get("research_notes", "")}),
                    timeout=STRUCTURED_SYNTHESIS_TIMEOUT_SECONDS or get_settings().qwen_timeout_seconds,
                )
                events = [item.model_dump(mode="json") for item in event_list.events]
                expected_events = sum(len(day.get("activities", [])) for day in state.get("daily_plan", []))
                if len(events) >= expected_events:
                    return {"events": events, "calendar_status": "ok", "calendar_data_source": "structured_llm", "calendar_warnings": []}
                raise ValueError(
                    f"Calendar structured synthesis returned {len(events)} events for {expected_events} itinerary activities."
                )
            except asyncio.TimeoutError:
                logger.warning("Calendar structured synthesis timed out; using fallback events.")
                failure_warning = "Calendar structured synthesis timed out."
            except Exception as exc:
                logger.warning("Calendar structured synthesis failed; using fallback events: %s", exc)
                failure_warning = f"Calendar structured synthesis failed: {exc}"
        else:
            failure_warning = "Calendar structured synthesis unavailable; using itinerary-derived fallback events."
        return self._events_from_daily_plan(state, live=False, initial_warnings=[failure_warning])

    def _events_from_daily_plan(
        self,
        state: CalendarState,
        *,
        live: bool,
        initial_warnings: list[str],
    ) -> dict[str, Any]:
        events: list[dict[str, Any]] = []
        warnings: list[str] = list(initial_warnings)
        daily_plan = state.get("daily_plan", [])
        expected_events = 0
        has_live_activity_link = False
        for day in daily_plan:
            activities = day.get("activities", []) if isinstance(day, dict) else []
            expected_events += len(activities)
            for activity in activities:
                if activity.get("map_link"):
                    has_live_activity_link = True
                try:
                    events.append(CalendarEventModel(title=activity.get("title", "Activity"), start_at=self._combine_datetime(day.get("date"), activity.get("start_time", "09:00")), end_at=self._combine_datetime(day.get("date"), activity.get("end_time", "10:00")), location=activity.get("location_name", ""), description=activity.get("description", ""), url=activity.get("booking_link") or activity.get("map_link")).model_dump(mode="json"))
                except ValueError as exc:
                    warnings.append(str(exc))
        if expected_events and len(events) == expected_events and has_live_activity_link:
            return {
                "events": events,
                "calendar_status": "ok",
                "calendar_data_source": "live",
                "calendar_warnings": [] if live else ["Calendar events generated from approved live itinerary activities."],
            }
        status = "error" if any("missing itinerary date" in item.lower() for item in warnings) else "fallback"
        data_source = "unavailable" if status == "error" else "fallback_estimate"
        return {"events": events if status != "error" else [], "calendar_status": status, "calendar_data_source": data_source, "calendar_warnings": warnings}

    async def write_calendar(self, state: CalendarState) -> dict[str, Any]:
        worker_input = self._worker_input(state)
        output_path = self.output_dir / f"{sanitize_request_id(worker_input.request_id)}_trip_calendar.ics"
        events = [CalendarEventModel.model_validate(item) for item in state.get("events", [])]
        status = state.get("calendar_status") or ("fallback" if events else "error")
        data_source = state.get("calendar_data_source") or ("fallback_estimate" if events else "unavailable")
        warnings = list(state.get("calendar_warnings") or [])
        build_ics_calendar(f"{state['destination']} Travel Plan", events, output_path, data_source=data_source, status=status, warnings=warnings)
        return {"result": CalendarExecutionResult(calendar_file=output_path, events_created=len(events), calendar_name=f"{state['destination']} Travel Plan", status=status, data_source=data_source, warnings=warnings, liubu_evidence=list(state.get("tool_evidence", []))).model_dump(mode="json")}

    def _combine_datetime(self, day: str | None, clock: str) -> datetime:
        if not day:
            raise ValueError("Missing itinerary date for calendar event; refusing to use current time.")
        day_value = datetime.fromisoformat(f"{day}T00:00:00").replace(tzinfo=timezone.utc)
        hour, minute = [int(part) for part in clock.split(":", 1)]
        return day_value.replace(hour=hour, minute=minute)

    def _route_after_agent(self, state: CalendarState) -> str:
        messages = state.get("messages") or []
        last_message = messages[-1] if messages else None
        return "tools" if getattr(last_message, "tool_calls", None) else "quality_gate"

    def _worker_input(self, state: CalendarState) -> LiubuWorkerInput:
        return LiubuWorkerInput.model_validate(state["worker_input"])

    def _daily_plan_has_live_links(self, daily_plan: list[dict[str, Any]]) -> bool:
        return any(
            activity.get("map_link") or activity.get("booking_link")
            for day in daily_plan
            for activity in day.get("activities", [])
        )
