from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from utils.agent_runtime import run_react_mcp_task, soul_path_for
from utils.icalendar_utils import build_ics_calendar
from utils.schemas import CalendarEventModel, CalendarExecutionResult
from utils.llm_factory import build_qwen_chat


class CalendarState(TypedDict, total=False):
    payload: dict[str, Any]
    destination: str
    daily_plan: list[dict[str, Any]]
    research_notes: str
    events: list[dict[str, Any]]
    result: dict[str, Any]


class CalendarBureau:
    def __init__(self, output_dir: str = "artifacts") -> None:
        self.output_dir = Path(output_dir)
        self.soul_path = soul_path_for(__file__)
        self.graph = self._build_graph()

    async def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = await self.graph.ainvoke({"payload": payload})
        return result["result"]

    def _build_graph(self):
        graph = StateGraph(CalendarState)
        graph.add_node("ingest", self.ingest)
        graph.add_node("research_calendar", self.research_calendar)
        graph.add_node("build_events", self.build_events)
        graph.add_node("write_calendar", self.write_calendar)
        graph.set_entry_point("ingest")
        graph.add_edge("ingest", "research_calendar")
        graph.add_edge("research_calendar", "build_events")
        graph.add_edge("build_events", "write_calendar")
        graph.add_edge("write_calendar", END)
        return graph.compile()

    async def ingest(self, state: CalendarState) -> dict[str, Any]:
        payload = state["payload"]
        approved_draft = payload.get("approved_draft", {})
        draft = approved_draft.get("itinerary_draft", {})
        return {"destination": approved_draft.get("destination") or draft.get("destination") or "Trip", "daily_plan": draft.get("daily_plan", [])}

    async def research_calendar(self, state: CalendarState) -> dict[str, Any]:
        notes = await run_react_mcp_task(
            soul_path=self.soul_path,
            server_names=["amap"],
            user_task=(
                f"Normalize itinerary locations for calendar generation in {state['destination']}. "
                f"Daily plan: {state.get('daily_plan', [])}. "
                "Use MCP tools when available and return notes about location naming, ambiguity, and reminders."
            ),
        )
        return {"research_notes": notes}

    async def build_events(self, state: CalendarState) -> dict[str, Any]:
        llm = build_qwen_chat()
        if llm:
            try:
                structured = llm.with_structured_output(list[CalendarEventModel])
                prompt = ChatPromptTemplate.from_messages([
                    ("system", Path(self.soul_path).read_text(encoding="utf-8")),
                    ("user", "Daily plan: {daily_plan}\nResearch notes: {research_notes}\nReturn a structured list of calendar events."),
                ])
                events = await (prompt | structured).ainvoke({"daily_plan": str(state.get("daily_plan", [])), "research_notes": state.get("research_notes", "")})
                return {"events": [item.model_dump(mode="json") for item in events]}
            except Exception:
                pass
        events: list[dict[str, Any]] = []
        for day in state.get("daily_plan", []):
            for activity in day.get("activities", []):
                events.append(CalendarEventModel(title=activity.get("title", "Activity"), start_at=self._combine_datetime(day.get("date"), activity.get("start_time", "09:00")), end_at=self._combine_datetime(day.get("date"), activity.get("end_time", "10:00")), location=activity.get("location_name", ""), description=activity.get("description", ""), url=activity.get("booking_link") or activity.get("map_link")).model_dump(mode="json"))
        return {"events": events}

    async def write_calendar(self, state: CalendarState) -> dict[str, Any]:
        payload = state["payload"]
        output_path = self.output_dir / f"{payload.get('request_id', 'trip')}_trip_calendar.ics"
        events = [CalendarEventModel.model_validate(item) for item in state.get("events", [])]
        build_ics_calendar(f"{state['destination']} Travel Plan", events, output_path)
        return {"result": CalendarExecutionResult(calendar_file=output_path, events_created=len(events), calendar_name=f"{state['destination']} Travel Plan").model_dump(mode="json")}

    def _combine_datetime(self, day: str | None, clock: str) -> datetime:
        day_value = datetime.fromisoformat(f"{day}T00:00:00") if day else datetime.utcnow()
        hour, minute = [int(part) for part in clock.split(":", 1)]
        return day_value.replace(hour=hour, minute=minute)
