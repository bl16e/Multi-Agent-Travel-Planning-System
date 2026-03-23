from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Annotated, Callable, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from provinces.liubu.accommodation.service import AccommodationBureau
from provinces.liubu.budget.service import BudgetBureau
from provinces.liubu.calendar.service import CalendarBureau
from provinces.liubu.flight_transport.service import FlightTransportBureau
from provinces.liubu.weather.service import WeatherBureau
from provinces.menxia_review.graph import MenxiaReviewAgent
from provinces.shangshu_orchestrator.orchestrator import ShangshuOrchestrator, ShangshuWorkflowContext
from provinces.zhongshu_itinerary.graph import ZhongshuItineraryAgent
from utils.permission_matrix import AgentRole
from utils.schemas import (
    AccommodationExecutionResult,
    BudgetExecutionResult,
    CalendarExecutionResult,
    FinalTravelPackageModel,
    FlightTransportExecutionResult,
    MenxiaReviewPacketModel,
    PlanningRequest,
    ProgressEvent,
    WeatherExecutionResult,
)


ProgressReporter = Callable[[str], None]


def merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class SystemState(TypedDict, total=False):
    request: dict[str, Any]
    context: ShangshuWorkflowContext
    status: str
    question: str
    draft_packet: dict[str, Any]
    review_packet: dict[str, Any]
    zhongshu_task_payload: dict[str, Any]
    liubu_tasks: list[dict[str, Any]]
    execution_results: Annotated[dict[str, Any], merge_dicts]
    final_package: dict[str, Any]
    rejected_payload: dict[str, Any]


class ProvinceWorkflow:
    def __init__(
        self,
        artifact_dir: str | Path | None = None,
        progress_reporter: ProgressReporter | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else Path(__file__).resolve().with_name("artifacts")
        self.progress_reporter = progress_reporter
        self.orchestrator = ShangshuOrchestrator()
        self.zhongshu = ZhongshuItineraryAgent()
        self.menxia = MenxiaReviewAgent()
        self.weather = WeatherBureau()
        self.calendar = CalendarBureau(output_dir=self.artifact_dir)
        self.budget = BudgetBureau()
        self.accommodation = AccommodationBureau()
        self.flight_transport = FlightTransportBureau()
        self.graph = self._build_graph()

    def set_progress_reporter(self, progress_reporter: ProgressReporter | None) -> None:
        self.progress_reporter = progress_reporter

    def _build_graph(self):
        graph = StateGraph(SystemState)
        graph.add_node("shangshu_preflight", self._node_preflight)
        graph.add_node("zhongshu_itinerary", self._node_zhongshu)
        graph.add_node("menxia_review", self._node_menxia)
        graph.add_node("shangshu_review_gate", self._node_review_gate)
        graph.add_node("shangshu_dispatch_liubu", self._node_dispatch_liubu)
        graph.add_node("liubu_weather", self._node_liubu_weather)
        graph.add_node("liubu_budget", self._node_liubu_budget)
        graph.add_node("liubu_accommodation", self._node_liubu_accommodation)
        graph.add_node("liubu_flight_transport", self._node_liubu_flight_transport)
        graph.add_node("liubu_calendar", self._node_liubu_calendar)
        graph.add_node("shangshu_assemble", self._node_assemble)
        graph.add_node("finish_human_intervene", self._node_finish_human)
        graph.add_node("finish_rejected", self._node_finish_rejected)

        graph.set_entry_point("shangshu_preflight")
        graph.add_conditional_edges("shangshu_preflight", self._route_after_preflight, {"zhongshu_itinerary": "zhongshu_itinerary", "finish_human_intervene": "finish_human_intervene"})
        graph.add_conditional_edges("zhongshu_itinerary", self._route_after_zhongshu, {"menxia_review": "menxia_review", "finish_rejected": "finish_rejected"})
        graph.add_edge("menxia_review", "shangshu_review_gate")
        graph.add_conditional_edges(
            "shangshu_review_gate", 
            self._route_after_review, {
                "shangshu_dispatch_liubu": "shangshu_dispatch_liubu", 
                "finish_human_intervene": "finish_human_intervene", 
                "retry_zhongshu": "zhongshu_itinerary",
                "finish_rejected": "finish_rejected"
                })
        graph.add_conditional_edges("shangshu_dispatch_liubu", self._route_to_liubu)
        graph.add_edge("liubu_weather", "shangshu_assemble")
        graph.add_edge("liubu_budget", "shangshu_assemble")
        graph.add_edge("liubu_accommodation", "shangshu_assemble")
        graph.add_edge("liubu_flight_transport", "shangshu_assemble")
        graph.add_edge("liubu_calendar", "shangshu_assemble")
        graph.add_edge("shangshu_assemble", END)
        graph.add_edge("finish_human_intervene", END)
        graph.add_edge("finish_rejected", END)
        return graph.compile()

    async def run(self, request: PlanningRequest) -> dict[str, Any]:
        self._emit_progress("workflow", "start", f"start request {request.request_id}", request.model_dump(mode="json"))
        result = await self.graph.ainvoke({"request": request.model_dump(mode="json")})
        self._emit_progress("workflow", "done", f"workflow finished with status={result.get('status', 'UNKNOWN')}", result)
        return result

    async def _node_preflight(self, state: SystemState) -> dict[str, Any]:
        self._emit_progress("shangshu_preflight", "start", "run preflight checks")
        request = PlanningRequest.model_validate(state["request"])
        context = self.orchestrator.bootstrap(request.request_id, request.model_dump(mode="json"))
        question = self._preflight_question(request)
        if question:
            context.pending_user_inputs.append(question)
            result = {"context": context, "status": "HUMAN_INTERVENE", "question": question}
            self._emit_progress("shangshu_preflight", "done", "preflight requires user input", result)
            return result
        result = {"context": context, "status": "RUNNING"}
        self._emit_progress("shangshu_preflight", "done", "preflight passed", result)
        return result

    def _route_after_preflight(self, state: SystemState) -> str:
        return "finish_human_intervene" if state.get("status") == "HUMAN_INTERVENE" else "zhongshu_itinerary"

    def _route_after_zhongshu(self, state: SystemState) -> str:
        return "finish_rejected" if state.get("status") == "ZHONGSHU_FAILED" else "menxia_review"

    async def _node_zhongshu(self, state: SystemState) -> dict[str, Any]:
        self._emit_progress("zhongshu_itinerary", "start", "generate itinerary draft")
        context = state["context"]
        payload = state.get("zhongshu_task_payload")
        if payload is None:
            dispatch = self.orchestrator.dispatch_to_zhongshu(context)
            payload = dispatch.tasks[0].payload

        try:
            result = await self.zhongshu.graph.ainvoke(payload)
            update = {"context": context, "draft_packet": result["finalized_packet"]}
            self._emit_progress("zhongshu_itinerary", "done", "draft generated", update)
            return update
        except Exception as e:
            self._emit_progress("zhongshu_itinerary", "error", f"draft generation failed: {str(e)}")
            return {"context": context, "status": "ZHONGSHU_FAILED", "error": str(e)}

    async def _node_menxia(self, state: SystemState) -> dict[str, Any]:
        self._emit_progress("menxia_review", "start", "review itinerary draft")
        context = state["context"]
        draft_packet = state["draft_packet"]
        review_dispatch = self.orchestrator.submit_draft_to_review(context, draft_packet)
        request = state["request"]
        result = await self.menxia.graph.ainvoke({"request_id": request["request_id"], "draft": review_dispatch.tasks[0].payload["draft"], "user_request": request})
        update = {"context": context, "review_packet": result["verdict_payload"]}
        self._emit_progress("menxia_review", "done", "review completed", update)
        return update

    async def _node_review_gate(self, state: SystemState) -> dict[str, Any]:
        self._emit_progress("shangshu_review_gate", "start", "process review verdict")
        context = state["context"]
        review_packet = state["review_packet"]
        follow_up = self.orchestrator.apply_review_verdict(context, review_packet)
        verdict = review_packet["verdict"]
        if verdict == "HUMAN_INTERVENE":
            question = (review_packet.get("human_questions") or ["Review requires user input."])[0]
            result = {"context": context, "status": "HUMAN_INTERVENE", "question": question}
            self._emit_progress("shangshu_review_gate", "done", "review requested user input", result)
            return result
        if verdict == "REJECTED":
            follow_up_payload = follow_up.tasks[0].payload if follow_up else None
            retry_allowed = bool((follow_up_payload or {}).get("governance", {}).get("retry_allowed"))
            result = {
                "context": context,
                "status": "RETRY_ZHONGSHU" if retry_allowed else "REJECTED",
                "zhongshu_task_payload": follow_up_payload if retry_allowed else None,
                "rejected_payload": {
                    "status": "REJECTED",
                    "request_id": context.request_id,
                    "review": review_packet,
                    "dashboard_url": self.orchestrator.build_dashboard_link(context),
                    "follow_up_task": follow_up_payload,
                    "rejection_count": context.rejection_count,
                    "max_rejection_rounds": context.max_rejection_rounds,
                    "retry_allowed": retry_allowed,
                },
            }
            self._emit_progress("shangshu_review_gate", "done", "draft rejected", result)
            return result
        result = {"context": context, "status": "APPROVED"}
        self._emit_progress("shangshu_review_gate", "done", "draft approved", result)
        return result

    def _route_after_review(self, state: SystemState) -> str:
        if state.get("status") == "HUMAN_INTERVENE":
            return "finish_human_intervene"
        if state.get("status") == "RETRY_ZHONGSHU":
            return "retry_zhongshu"
        if state.get("status") == "REJECTED":
            return "finish_rejected"
        return "shangshu_dispatch_liubu"

    async def _node_dispatch_liubu(self, state: SystemState) -> dict[str, Any]:
        self._emit_progress("shangshu_dispatch_liubu", "start", "dispatch tasks to bureaus")
        context = state["context"]
        draft_packet = state["draft_packet"]
        review_packet = state["review_packet"]
        execution_plan = {"required_bureaus": review_packet.get("approved_bureaus") or draft_packet.get("required_bureaus") or [], "user_request": state["request"]}
        dispatch = self.orchestrator.dispatch_liubu_execution(context, execution_plan)
        tasks = [{"node": self.orchestrator._graph_node_name_for(task.target), "payload": task.payload, "target": task.target.value} for task in dispatch.tasks]
        result = {"context": context, "liubu_tasks": tasks}
        self._emit_progress("shangshu_dispatch_liubu", "done", "bureaus dispatched", result)
        return result

    def _route_to_liubu(self, state: SystemState) -> list[Send]:
        return [Send(task["node"], {"payload": task["payload"], "context": state["context"]}) for task in state.get("liubu_tasks", [])]

    async def _node_liubu_weather(self, state: SystemState) -> dict[str, Any]:
        self._emit_progress("liubu_weather", "start", "weather bureau running")
        result = await self.weather.run(state["payload"])
        self.orchestrator.register_execution_result(state["context"], AgentRole.WEATHER, result)
        update = {"execution_results": {"WEATHER": result}}
        self._emit_progress("liubu_weather", "done", "weather bureau returned", update)
        return update

    async def _node_liubu_budget(self, state: SystemState) -> dict[str, Any]:
        self._emit_progress("liubu_budget", "start", "budget bureau running")
        result = await self.budget.run(state["payload"])
        self.orchestrator.register_execution_result(state["context"], AgentRole.BUDGET, result)
        update = {"execution_results": {"BUDGET": result}}
        self._emit_progress("liubu_budget", "done", "budget bureau returned", update)
        return update

    async def _node_liubu_accommodation(self, state: SystemState) -> dict[str, Any]:
        self._emit_progress("liubu_accommodation", "start", "accommodation bureau running")
        result = await self.accommodation.run(state["payload"])
        self.orchestrator.register_execution_result(state["context"], AgentRole.ACCOMMODATION, result)
        update = {"execution_results": {"ACCOMMODATION": result}}
        self._emit_progress("liubu_accommodation", "done", "accommodation bureau returned", update)
        return update

    async def _node_liubu_flight_transport(self, state: SystemState) -> dict[str, Any]:
        self._emit_progress("liubu_flight_transport", "start", "flight transport bureau running")
        result = await self.flight_transport.run(state["payload"])
        self.orchestrator.register_execution_result(state["context"], AgentRole.FLIGHT_TRANSPORT, result)
        update = {"execution_results": {"FLIGHT_TRANSPORT": result}}
        self._emit_progress("liubu_flight_transport", "done", "flight transport bureau returned", update)
        return update

    async def _node_liubu_calendar(self, state: SystemState) -> dict[str, Any]:
        self._emit_progress("liubu_calendar", "start", "calendar bureau running")
        result = await self.calendar.run(state["payload"])
        self.orchestrator.register_execution_result(state["context"], AgentRole.CALENDAR, result)
        update = {"execution_results": {"CALENDAR": result}}
        self._emit_progress("liubu_calendar", "done", "calendar bureau returned", update)
        return update

    async def _node_assemble(self, state: SystemState) -> dict[str, Any]:
        self._emit_progress("shangshu_assemble", "start", "assemble final package")
        context = state["context"]
        assembled = self.orchestrator.assemble_outputs(context)
        package = self.build_final_package(PlanningRequest.model_validate(state["request"]), assembled, state["draft_packet"], state["review_packet"], state.get("execution_results", {}), self.artifact_dir)
        result = {"context": context, "status": "DONE", "final_package": package.model_dump(mode="json")}
        self._emit_progress("shangshu_assemble", "done", "final package assembled", result)
        return result

    async def _node_finish_human(self, state: SystemState) -> dict[str, Any]:
        result = {"status": "HUMAN_INTERVENE"}
        self._emit_progress("finish_human_intervene", "done", "workflow waiting for human input", result)
        return result

    async def _node_finish_rejected(self, state: SystemState) -> dict[str, Any]:
        error_msg = state.get("error", "未知错误")
        rejected_payload = state.get("rejected_payload", {})
        if not rejected_payload:
            ctx = state.get("context")
            rid = ctx.request_id if isinstance(ctx, ShangshuWorkflowContext) else (ctx.get("request_id") if isinstance(ctx, dict) else "unknown")
            rejected_payload = {
                "status": "REJECTED",
                "request_id": rid,
                "error": error_msg,
                "reason": "中书省生成失败或达到最大拒绝轮次"
            }
        result = {"status": "REJECTED", "rejected_payload": rejected_payload}
        self._emit_progress("finish_rejected", "done", f"workflow finished as rejected: {error_msg}", result)
        return result

    def _preflight_question(self, request: PlanningRequest) -> str | None:
        if not request.profile.destination_preferences:
            return "Please provide at least one destination preference."
        if not request.profile.start_date or not request.profile.end_date:
            return "Please provide trip start and end dates."
        if request.profile.total_budget is None:
            return "Please confirm the total trip budget cap before review."
        return None

    def _emit_progress(self, step: str, phase: str, message: str, payload: Any | None = None) -> None:
        if self.progress_reporter is None:
            return
        timestamp = datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")
        line = f"[{timestamp}] [{phase.upper()}] {step} | {message}"
        summary = self._summarize_payload(payload)
        if summary:
            line = f"{line} | output={summary}"
        self.progress_reporter(line)

    def _summarize_payload(self, payload: Any | None) -> str:
        if payload is None:
            return ""
        if isinstance(payload, ShangshuWorkflowContext):
            return self._to_json({"request_id": payload.request_id, "state": payload.current_state.value})
        if isinstance(payload, dict):
            summary: dict[str, Any] = {}
            for key in ("status", "question", "request_id", "destination"):
                if key in payload:
                    summary[key] = payload[key]
            if "draft_packet" in payload:
                draft = payload["draft_packet"] or {}
                summary["draft"] = {
                    "destination": draft.get("destination"),
                    "required_bureaus": draft.get("required_bureaus", []),
                    "days": len((draft.get("itinerary_draft") or {}).get("daily_plan", [])),
                }
            if "review_packet" in payload:
                review = payload["review_packet"] or {}
                review_governance = review.get("governance") or {}
                summary["review"] = {
                    "verdict": review.get("verdict"),
                    "summary": review.get("summary"),
                    "rejection_round": review_governance.get("rejection_round"),
                    "max_rejection_rounds": review_governance.get("max_rejection_rounds"),
                }
            if "liubu_tasks" in payload:
                summary["liubu_tasks"] = [task.get("target") for task in payload["liubu_tasks"]]
            if "execution_results" in payload:
                summary["execution_results"] = {
                    key: self._summarize_execution_result(value)
                    for key, value in payload["execution_results"].items()
                }
            if "final_package" in payload:
                final_package = payload["final_package"] or {}
                summary["final_package"] = {
                    "destination": final_package.get("destination"),
                    "workflow_state": final_package.get("workflow_state"),
                    "booking_links": len(final_package.get("booking_links", [])),
                }
            if "rejected_payload" in payload:
                rejected = payload["rejected_payload"] or {}
                review = rejected.get("review") or {}
                summary["rejected"] = {
                    "verdict": review.get("verdict"),
                    "summary": review.get("summary"),
                    "rejection_count": rejected.get("rejection_count"),
                    "max_rejection_rounds": rejected.get("max_rejection_rounds"),
                    "retry_allowed": rejected.get("retry_allowed"),
                }
            if not summary:
                summary = payload
            return self._to_json(summary)
        return self._to_json(payload)

    def _summarize_execution_result(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {"value": str(payload)}
        summary: dict[str, Any] = {"bureau": payload.get("bureau")}
        if "forecast_days" in payload:
            summary["forecast_days"] = len(payload.get("forecast_days", []))
        if "budget_breakdown" in payload:
            summary["budget_items"] = len(payload.get("budget_breakdown", []))
            summary["total_estimated_cost"] = payload.get("total_estimated_cost")
        if "hotel_options" in payload:
            summary["hotel_options"] = len(payload.get("hotel_options", []))
        if "flight_options" in payload:
            summary["flight_options"] = len(payload.get("flight_options", []))
        if "events_created" in payload:
            summary["events_created"] = payload.get("events_created")
        if "warnings" in payload:
            summary["warnings"] = payload.get("warnings", [])[:2]
        return summary

    def _to_json(self, payload: Any) -> str:
        raw = json.dumps(payload, ensure_ascii=False, default=str)
        return raw if len(raw) <= 320 else raw[:317] + "..."

    @staticmethod
    def build_final_package(request: PlanningRequest, assembled: dict[str, Any], draft_packet: dict[str, Any], review_packet: dict[str, Any], execution_results: dict[str, Any], artifact_dir: Path) -> FinalTravelPackageModel:
        markdown_path = build_markdown(request, draft_packet, review_packet, execution_results, assembled["dashboard_url"], artifact_dir)
        weather = WeatherExecutionResult.model_validate(execution_results.get("WEATHER")) if execution_results.get("WEATHER") else None
        budget = BudgetExecutionResult.model_validate(execution_results.get("BUDGET")) if execution_results.get("BUDGET") else None
        accommodation = AccommodationExecutionResult.model_validate(execution_results.get("ACCOMMODATION")) if execution_results.get("ACCOMMODATION") else None
        flight_transport = FlightTransportExecutionResult.model_validate(execution_results.get("FLIGHT_TRANSPORT")) if execution_results.get("FLIGHT_TRANSPORT") else None
        calendar = CalendarExecutionResult.model_validate(execution_results.get("CALENDAR")) if execution_results.get("CALENDAR") else None
        booking_links = collect_booking_links(draft_packet, execution_results)
        progress_events = [ProgressEvent.model_validate(item) for item in assembled.get("progress_events", [])]
        return FinalTravelPackageModel(request_id=request.request_id, destination=draft_packet["destination"], markdown_file=markdown_path, calendar_file=Path(calendar.calendar_file) if calendar else None, dashboard_url=assembled["dashboard_url"], itinerary=draft_packet["itinerary_draft"], review=MenxiaReviewPacketModel.model_validate(review_packet), weather=weather, budget=budget, accommodation=accommodation, flight_transport=flight_transport, booking_links=booking_links, packing_list=weather.packing_list if weather else [], progress_events=progress_events, generated_at=datetime.now(timezone.utc))


def collect_booking_links(draft_packet: dict[str, Any], execution_results: dict[str, Any]) -> list[str]:
    links: list[str] = []
    for option in execution_results.get("FLIGHT_TRANSPORT", {}).get("booking_links", []):
        links.append(str(option))
    for option in execution_results.get("ACCOMMODATION", {}).get("booking_links", []):
        links.append(str(option))
    for day in draft_packet["itinerary_draft"].get("daily_plan", []):
        for activity in day.get("activities", []):
            if activity.get("booking_link"):
                links.append(str(activity["booking_link"]))
    return list(dict.fromkeys(links))


def build_markdown(request: PlanningRequest, draft_packet: dict[str, Any], review_packet: dict[str, Any], execution_results: dict[str, Any], dashboard_url: str, artifact_dir: Path) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"{request.request_id}_travel_plan.md"
    itinerary = draft_packet["itinerary_draft"]
    weather = execution_results.get("WEATHER", {})
    budget = execution_results.get("BUDGET", {})
    accommodation = execution_results.get("ACCOMMODATION", {})
    flight = execution_results.get("FLIGHT_TRANSPORT", {})
    calendar = execution_results.get("CALENDAR", {})
    lines: list[str] = [f"# {draft_packet['destination']} Travel Plan", "", "## Overview", f"- Request ID: {request.request_id}", f"- Review Verdict: {review_packet['verdict']}", f"- Dashboard: {dashboard_url}", f"- Calendar File: {calendar.get('calendar_file', '')}", "", "## Daily Itinerary"]
    for day in itinerary["daily_plan"]:
        lines.append(f"### Day {day['day_index']} - {day['date']} - {day['theme']}")
        lines.append(day["summary"])
        for activity in day["activities"]:
            line = f"- {activity['start_time']}-{activity['end_time']} {activity['title']} | {activity['location_name']}"
            if activity.get("map_link"):
                line += f" | [Map]({activity['map_link']})"
            if activity.get("booking_link"):
                line += f" | [Booking]({activity['booking_link']})"
            lines.append(line)
            lines.append(f"  - {activity['description']}")
        lines.append("")
    lines.extend(["## Budget", "| Category | Item | Estimated Cost | Currency | Notes |", "|---|---|---:|---|---|"])
    for item in budget.get("budget_breakdown", []):
        lines.append(f"| {item['category']} | {item['item']} | {item['estimated_cost']:.2f} | {item['currency']} | {item.get('notes', '')} |")
    if budget:
        lines.append(f"| total | Total estimated spend | {budget.get('total_estimated_cost', 0):.2f} | {budget.get('currency', request.profile.currency)} | {'; '.join(budget.get('warnings', []))} |")
    lines.extend(["", "## Booking Links"])
    for link in collect_booking_links(draft_packet, execution_results):
        lines.append(f"- {link}")
    lines.extend(["", "## Accommodation"])
    for hotel in accommodation.get("hotel_options", []):
        lines.append(f"- {hotel['name']} | {hotel.get('nightly_rate') or hotel.get('total_rate') or 0} {hotel['currency']} | {hotel.get('address') or ''} | {hotel.get('booking_link') or ''}")
    lines.extend(["", "## Flight & Transport"])
    for option in flight.get("flight_options", []):
        lines.append(f"- {option['airline']} | {option['price']} {option['currency']} | {option['departure_airport']} -> {option['arrival_airport']} | {option.get('booking_link') or ''}")
    lines.extend(["", "## Weather and Packing", weather.get("summary", "")])
    for day in weather.get("forecast_days", []):
        lines.append(f"- {day['date']}: {day['condition']} {day['min_temp_c']}-{day['max_temp_c']}C")
    lines.append("Packing list:")
    for item in weather.get("packing_list", []):
        lines.append(f"- {item}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
