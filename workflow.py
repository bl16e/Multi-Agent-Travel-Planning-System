from __future__ import annotations

import copy
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Annotated, Awaitable, Callable, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command, Send, interrupt
from langchain_core.messages import BaseMessage

from provinces.liubu.constrained.state import normalize_worker_input
from provinces.liubu.accommodation.service import AccommodationBureau
from provinces.liubu.budget.service import BudgetBureau
from provinces.liubu.calendar.service import CalendarBureau
from provinces.liubu.flight_transport.service import FlightTransportBureau
from provinces.liubu.weather.service import WeatherBureau
from provinces.menxia_review.graph import MenxiaReviewAgent
from provinces.shangshu_orchestrator.orchestrator import ShangshuOrchestrator, ShangshuWorkflowContext
from provinces.zhongshu_itinerary.graph import ZhongshuItineraryAgent
from utils.path_safety import sanitize_request_id
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
    AgentRole,
    WorkflowState,
)
from utils.settings import get_settings

logger = logging.getLogger(__name__)

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
    resume_state: dict[str, Any]
    resume_mode: str
    rewrite_reason: str


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
        self._checkpoint_cm = None
        self._checkpointer = None
        self._persistent_graph = None

    def set_progress_reporter(self, progress_reporter: ProgressReporter | None) -> None:
        self.progress_reporter = progress_reporter

    def _build_graph(self, entry_point: str = "shangshu_preflight", checkpointer: Any | None = None):
        graph = StateGraph(SystemState)
        graph.add_node("shangshu_preflight", self._node_preflight)
        graph.add_node("interrupt_preflight", self._node_interrupt_preflight)
        graph.add_node("zhongshu_itinerary", self._node_zhongshu)
        graph.add_node("menxia_review", self._node_menxia)
        graph.add_node("shangshu_review_gate", self._node_review_gate)
        graph.add_node("interrupt_review", self._node_interrupt_review)
        graph.add_node("shangshu_dispatch_liubu", self._node_dispatch_liubu)
        graph.add_node("liubu_weather", self._node_liubu_weather)
        graph.add_node("liubu_budget", self._node_liubu_budget)
        graph.add_node("liubu_accommodation", self._node_liubu_accommodation)
        graph.add_node("liubu_flight_transport", self._node_liubu_flight_transport)
        graph.add_node("liubu_calendar", self._node_liubu_calendar)
        graph.add_node("shangshu_assemble", self._node_assemble)
        graph.add_node("finish_human_intervene", self._node_finish_human)
        graph.add_node("finish_rejected", self._node_finish_rejected)

        graph.set_entry_point(entry_point)
        graph.add_conditional_edges("shangshu_preflight", self._route_after_preflight, {"zhongshu_itinerary": "zhongshu_itinerary", "interrupt_preflight": "interrupt_preflight"})
        graph.add_edge("interrupt_preflight", "zhongshu_itinerary")
        graph.add_conditional_edges("zhongshu_itinerary", self._route_after_zhongshu, {"menxia_review": "menxia_review", "finish_rejected": "finish_rejected"})
        graph.add_edge("menxia_review", "shangshu_review_gate")
        graph.add_conditional_edges(
            "shangshu_review_gate", 
            self._route_after_review, {
                "shangshu_dispatch_liubu": "shangshu_dispatch_liubu",
                "interrupt_review": "interrupt_review",
                "retry_zhongshu": "zhongshu_itinerary",
                "finish_rejected": "finish_rejected"
                })
        graph.add_edge("interrupt_review", "zhongshu_itinerary")
        graph.add_conditional_edges("shangshu_dispatch_liubu", self._route_to_liubu)
        graph.add_edge("liubu_weather", "shangshu_assemble")
        graph.add_edge("liubu_budget", "shangshu_assemble")
        graph.add_edge("liubu_accommodation", "shangshu_assemble")
        graph.add_edge("liubu_flight_transport", "shangshu_assemble")
        graph.add_edge("liubu_calendar", "shangshu_assemble")
        graph.add_edge("shangshu_assemble", END)
        graph.add_edge("finish_human_intervene", END)
        graph.add_edge("finish_rejected", END)
        return graph.compile(checkpointer=checkpointer)

    async def _ensure_persistent_graph(self):
        if self._persistent_graph is not None:
            return self._persistent_graph
        checkpoint_db = Path(get_settings().langgraph_checkpoint_db)
        checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint_cm = AsyncSqliteSaver.from_conn_string(str(checkpoint_db))
        self._checkpointer = await self._checkpoint_cm.__aenter__()
        self._persistent_graph = self._build_graph(checkpointer=self._checkpointer)
        return self._persistent_graph

    async def _close_checkpointer(self) -> None:
        if self._checkpoint_cm is not None:
            await self._checkpoint_cm.__aexit__(None, None, None)
        self._checkpoint_cm = None
        self._checkpointer = None
        self._persistent_graph = None

    async def run(self, request: PlanningRequest) -> dict[str, Any]:
        self._emit_progress("workflow", "start", f"start request {request.request_id}", request.model_dump(mode="json"))
        try:
            graph = await self._ensure_persistent_graph()
            config = self._thread_config(request.request_id)
            result = await self._await_graph_result(
                graph.ainvoke({"request": request.model_dump(mode="json")}, config=config),
                request_id=request.request_id,
                operation="Workflow run",
            )
            result = await self._normalize_interrupt_result(result, config)
            self._emit_progress("workflow", "done", f"workflow finished with status={result.get('status', 'UNKNOWN')}", result)
            return result
        finally:
            await self._close_checkpointer()

    async def stream_run(self, request: PlanningRequest):
        self._emit_progress("workflow", "start", f"start request {request.request_id}", request.model_dump(mode="json"))
        config = self._thread_config(request.request_id)
        try:
            graph = await self._ensure_persistent_graph()
            async for event in self._stream_graph_events(
                graph,
                {"request": request.model_dump(mode="json")},
                config,
            ):
                yield event
        finally:
            await self._close_checkpointer()

    async def resume(self, resume_state: dict[str, Any], human_resume: Any) -> dict[str, Any]:
        thread_id = str(resume_state.get("thread_id") or "")
        if not thread_id:
            raise ValueError("Cannot resume without boundary checkpoint thread_id.")
        try:
            graph = await self._ensure_persistent_graph()
            config = self._thread_config(thread_id)
            resume_payload = human_resume.model_dump(mode="python") if hasattr(human_resume, "model_dump") else dict(human_resume or {})
            result = await self._await_graph_result(
                graph.ainvoke(Command(resume=resume_payload), config=config),
                request_id=thread_id,
                operation="Workflow resume",
            )
            return await self._normalize_interrupt_result(result, config)
        finally:
            await self._close_checkpointer()

    async def stream_resume(self, resume_state: dict[str, Any], human_resume: Any):
        thread_id = str(resume_state.get("thread_id") or "")
        if not thread_id:
            raise ValueError("Cannot resume without boundary checkpoint thread_id.")
        config = self._thread_config(thread_id)
        resume_payload = human_resume.model_dump(mode="python") if hasattr(human_resume, "model_dump") else dict(human_resume or {})
        try:
            graph = await self._ensure_persistent_graph()
            async for event in self._stream_graph_events(graph, Command(resume=resume_payload), config):
                yield event
        finally:
            await self._close_checkpointer()

    async def _stream_graph_events(self, graph: Any, graph_input: Any, config: dict[str, Any]):
        last_state: dict[str, Any] = {}
        stream = graph.astream(graph_input, config=config, stream_mode=["updates", "messages"]).__aiter__()
        while True:
            try:
                chunk = await self._await_graph_result(
                    stream.__anext__(),
                    request_id=str(config.get("configurable", {}).get("thread_id", "")),
                    operation="Workflow stream",
                )
            except StopAsyncIteration:
                break
            mode, data = chunk if isinstance(chunk, tuple) and len(chunk) == 2 else ("updates", chunk)
            if mode == "messages":
                message = data[0] if isinstance(data, tuple) else data
                content = getattr(message, "content", None)
                if content:
                    yield {"event": "message", "data": {"content": content}}
                continue
            if mode != "updates" or not isinstance(data, dict):
                continue
            if "__interrupt__" in data:
                last_state["__interrupt__"] = data["__interrupt__"]
                yield {"event": "progress", "data": {"node": "__interrupt__", "status": "HUMAN_INTERVENE"}}
                continue
            for node_name, update in data.items():
                if isinstance(update, dict):
                    last_state.update(update)
                    payload = {"node": node_name}
                    if "status" in update:
                        payload["status"] = update["status"]
                    if "question" in update:
                        payload["question"] = update["question"]
                    yield {"event": "progress", "data": payload}
        snapshot = await graph.aget_state(config)
        result = dict(getattr(snapshot, "values", None) or last_state)
        if last_state.get("__interrupt__"):
            result["__interrupt__"] = last_state["__interrupt__"]
        result = await self._normalize_interrupt_result(result, config)
        self._emit_progress("workflow", "done", f"workflow finished with status={result.get('status', 'UNKNOWN')}", result)
        yield {"event": "result", "data": result}

    async def _node_preflight(self, state: SystemState) -> dict[str, Any]:
        self._emit_progress("shangshu_preflight", "start", "run preflight checks")
        request = PlanningRequest.model_validate(state["request"])
        context = self.orchestrator.bootstrap(request.request_id, request.model_dump(mode="json"))
        question = self._preflight_question(request)
        if question:
            context.pending_user_inputs.append(question)
            result = {
                "context": context,
                "status": "HUMAN_INTERVENE",
                "question": question,
                "resume_mode": "boundary",
                "resume_state": self._boundary_resume_state(context.request_id, question, ["interrupt_preflight"]),
            }
            self._emit_progress("shangshu_preflight", "done", "preflight requires user input", result)
            return result
        result = {"context": context, "status": "RUNNING"}
        self._emit_progress("shangshu_preflight", "done", "preflight passed", result)
        return result

    def _route_after_preflight(self, state: SystemState) -> str:
        return "interrupt_preflight" if state.get("status") == "HUMAN_INTERVENE" else "zhongshu_itinerary"

    async def _node_interrupt_preflight(self, state: SystemState) -> dict[str, Any]:
        context = state["context"]
        question = state.get("question") or "Please provide the missing trip details."
        answer = interrupt({"question": question, "request_id": context.request_id, "mode": "boundary"})
        request = self._apply_resume_payload(state["request"], answer)
        context.user_request = request
        context.pending_user_inputs = [item for item in context.pending_user_inputs if item != question]
        return {
            "request": request,
            "context": context,
            "status": "RUNNING",
            "question": "",
            "resume_mode": "none",
            "resume_state": {},
        }

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
            result = {
                "context": context,
                "status": "HUMAN_INTERVENE",
                "question": question,
                "resume_mode": "boundary",
                "resume_state": self._boundary_resume_state(context.request_id, question, ["interrupt_review"]),
            }
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
            return "interrupt_review"
        if state.get("status") == "RETRY_ZHONGSHU":
            return "retry_zhongshu"
        if state.get("status") == "REJECTED":
            return "finish_rejected"
        return "shangshu_dispatch_liubu"

    async def _node_interrupt_review(self, state: SystemState) -> dict[str, Any]:
        context = state["context"]
        question = state.get("question") or "Review requires user input."
        answer = interrupt({"question": question, "request_id": context.request_id, "mode": "boundary"})
        request = self._apply_resume_payload(state["request"], answer)
        context.user_request = request
        context.pending_user_inputs = [item for item in context.pending_user_inputs if item != question]
        follow_up_payload = {
            "request_id": context.request_id,
            "draft": state.get("draft_packet", {}),
            "review_feedback": state.get("review_packet", {}),
            "user_request": request,
            "human_resume": answer,
            "governance": {
                "source_state": WorkflowState.HUMAN_INTERVENE.value,
                "human_intervened": True,
                "retry_allowed": True,
            },
        }
        return {
            "request": request,
            "context": context,
            "status": "RETRY_ZHONGSHU",
            "zhongshu_task_payload": follow_up_payload,
            "question": "",
            "resume_mode": "none",
            "resume_state": {},
        }

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
        tasks = state.get("liubu_tasks") or []
        if not tasks:
            raise ValueError("No Liubu bureau tasks to dispatch")
        return [
            Send(task["node"], {"payload": task["payload"], "context": self._clone_liubu_context(state["context"])})
            for task in tasks
        ]

    def _clone_liubu_context(self, context: ShangshuWorkflowContext) -> ShangshuWorkflowContext:
        return copy.deepcopy(context)

    async def _node_liubu_weather(self, state: SystemState) -> dict[str, Any]:
        return await self._run_liubu_node(
            state,
            node_name="liubu_weather",
            role=AgentRole.WEATHER,
            running_message="weather bureau running",
            returned_message="weather bureau returned",
            run_bureau=self.weather.run,
            fallback_result=self._fallback_weather_result,
        )

    async def _node_liubu_budget(self, state: SystemState) -> dict[str, Any]:
        return await self._run_liubu_node(
            state,
            node_name="liubu_budget",
            role=AgentRole.BUDGET,
            running_message="budget bureau running",
            returned_message="budget bureau returned",
            run_bureau=self.budget.run,
            fallback_result=self._fallback_budget_result,
        )

    async def _node_liubu_accommodation(self, state: SystemState) -> dict[str, Any]:
        return await self._run_liubu_node(
            state,
            node_name="liubu_accommodation",
            role=AgentRole.ACCOMMODATION,
            running_message="accommodation bureau running",
            returned_message="accommodation bureau returned",
            run_bureau=self.accommodation.run,
            fallback_result=self._fallback_accommodation_result,
        )

    async def _node_liubu_flight_transport(self, state: SystemState) -> dict[str, Any]:
        return await self._run_liubu_node(
            state,
            node_name="liubu_flight_transport",
            role=AgentRole.FLIGHT_TRANSPORT,
            running_message="flight transport bureau running",
            returned_message="flight transport bureau returned",
            run_bureau=self.flight_transport.run,
            fallback_result=self._fallback_flight_transport_result,
        )

    async def _node_liubu_calendar(self, state: SystemState) -> dict[str, Any]:
        return await self._run_liubu_node(
            state,
            node_name="liubu_calendar",
            role=AgentRole.CALENDAR,
            running_message="calendar bureau running",
            returned_message="calendar bureau returned",
            run_bureau=self.calendar.run,
            fallback_result=self._fallback_calendar_result,
        )

    async def _run_liubu_node(
        self,
        state: SystemState,
        *,
        node_name: str,
        role: AgentRole,
        running_message: str,
        returned_message: str,
        run_bureau: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
        fallback_result: Callable[[dict[str, Any], Exception], dict[str, Any]],
    ) -> dict[str, Any]:
        self._emit_progress(node_name, "start", running_message)
        try:
            result = await run_bureau(self._liubu_subtask(state["payload"], role))
            phase = "done"
            message = returned_message
        except Exception as exc:
            result = fallback_result(state["payload"], exc)
            phase = "error"
            message = f"{role.value} bureau failed; using fallback result: {exc}"
        self.orchestrator.register_execution_result(state["context"], role, result)
        update = {"execution_results": {role.value: result}}
        self._emit_progress(node_name, phase, message, update)
        return update

    def _liubu_subtask(self, payload: dict[str, Any], role: AgentRole) -> dict[str, Any]:
        return {"worker_input": normalize_worker_input(payload, role.value)}

    def _fallback_weather_result(self, payload: dict[str, Any], exc: Exception) -> dict[str, Any]:
        return {
            "bureau": "WEATHER",
            "destination": self._payload_destination(payload),
            "forecast_days": [],
            "packing_list": [],
            "warnings": [f"Weather bureau failed: {exc}"],
            "summary": "Weather bureau failed; no live weather guidance was generated.",
        }

    def _fallback_budget_result(self, payload: dict[str, Any], exc: Exception) -> dict[str, Any]:
        return {
            "bureau": "BUDGET",
            "currency": self._payload_profile(payload).get("currency", "USD"),
            "budget_breakdown": [],
            "total_estimated_cost": 0,
            "warnings": [f"Budget bureau failed: {exc}"],
        }

    def _fallback_accommodation_result(self, payload: dict[str, Any], exc: Exception) -> dict[str, Any]:
        return {
            "bureau": "ACCOMMODATION",
            "destination": self._payload_destination(payload),
            "hotel_options": [],
            "booking_links": [],
            "search_notes": [f"Accommodation bureau failed: {exc}"],
        }

    def _fallback_flight_transport_result(self, payload: dict[str, Any], exc: Exception) -> dict[str, Any]:
        profile = self._payload_profile(payload)
        return {
            "bureau": "FLIGHT_TRANSPORT",
            "origin": profile.get("origin_city") or "Unknown origin",
            "destination": self._payload_destination(payload),
            "flight_options": [],
            "transport_notes": [f"Flight transport bureau failed: {exc}"],
            "booking_links": [],
        }

    def _fallback_calendar_result(self, payload: dict[str, Any], exc: Exception) -> dict[str, Any]:
        request_id = sanitize_request_id(str(payload.get("request_id") or "trip"))
        return {
            "bureau": "CALENDAR",
            "calendar_file": self.artifact_dir / f"{request_id}_trip_calendar.ics",
            "events_created": 0,
            "calendar_name": f"{self._payload_destination(payload)} Travel Plan",
            "warnings": [f"Calendar bureau failed: {exc}"],
        }

    def _payload_destination(self, payload: dict[str, Any]) -> str:
        approved_draft = payload.get("approved_draft", {})
        draft = approved_draft.get("itinerary_draft", {})
        return approved_draft.get("destination") or draft.get("destination") or "Unknown Destination"

    def _payload_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        execution_plan = payload.get("execution_plan", {})
        user_request = execution_plan.get("user_request", {})
        return user_request.get("profile", {})

    async def _node_assemble(self, state: SystemState) -> dict[str, Any]:
        self._emit_progress("shangshu_assemble", "start", "assemble final package")
        context = state["context"]
        fallback_sources = self._fallback_delivery_sources(
            state.get("draft_packet", {}),
            state.get("review_packet", {}),
            state.get("execution_results", {}),
        )
        if fallback_sources:
            result = {
                "context": context,
                "status": "REJECTED",
                "rejected_payload": {
                    "status": "REJECTED",
                    "request_id": context.request_id,
                    "reason": "fallback_outputs_not_deliverable",
                    "summary": "Final package requires non-fallback review and bureau evidence before delivery.",
                    "fallback_sources": fallback_sources,
                    "review": state.get("review_packet", {}),
                    "dashboard_url": self.orchestrator.build_dashboard_link(context),
                },
            }
            self._emit_progress("shangshu_assemble", "done", "fallback outputs blocked before final delivery", result)
            return result
        assembled = self.orchestrator.assemble_outputs(context)
        package = self.build_final_package(PlanningRequest.model_validate(state["request"]), assembled, state["draft_packet"], state["review_packet"], state.get("execution_results", {}), self.artifact_dir)
        result = {"context": context, "status": "DONE", "final_package": package.model_dump(mode="json")}
        self._emit_progress("shangshu_assemble", "done", "final package assembled", result)
        return result

    def _fallback_delivery_sources(
        self,
        draft_packet: dict[str, Any],
        review_packet: dict[str, Any],
        execution_results: dict[str, Any],
    ) -> list[dict[str, str]]:
        sources: list[dict[str, str]] = []
        if self._contains_fallback_content(draft_packet):
            sources.append(
                {
                    "component": "ZHONGSHU",
                    "status": "approved",
                    "data_source": "live",
                    "reason": "fallback_content",
                }
            )
        review_source = str(review_packet.get("data_source") or "")
        if review_source in {"fallback_estimate", "unavailable"}:
            sources.append(
                {
                    "component": "MENXIA",
                    "status": str(review_packet.get("verdict") or "unknown"),
                    "data_source": review_source,
                }
            )
        for bureau_name, result in sorted((execution_results or {}).items()):
            if not isinstance(result, dict):
                continue
            status = str(result.get("status") or "unknown")
            data_source = str(result.get("data_source") or "unavailable")
            if status in {"fallback", "error"} or data_source in {"fallback_estimate", "unavailable"}:
                sources.append(
                    {
                        "component": str(bureau_name),
                        "status": status,
                        "data_source": data_source,
                    }
                )
                continue
            if self._contains_fallback_content(result):
                sources.append(
                    {
                        "component": str(bureau_name),
                        "status": status,
                        "data_source": data_source,
                        "reason": "fallback_content",
                    }
                )
        return sources

    def _contains_fallback_content(self, payload: Any) -> bool:
        text = json.dumps(payload, ensure_ascii=False, default=str).lower()
        blocked_terms = (
            "fallback_estimate",
            "live_research_fallback",
            "fallback_from",
            "estimated fallback",
            " fell back ",
            "fallback output",
            "offline fallback",
            "structured synthesis failed",
            "structured synthesis timed out",
            "error: toolexception",
            "please fix your mistakes",
            "generic placeholder",
            "named local checkpoints",
            "venue confirmation block",
            "main visitor district",
            "input-derived offline plan segment",
            "replace with live venue details",
        )
        return any(term in text for term in blocked_terms)

    async def _node_finish_human(self, state: SystemState) -> dict[str, Any]:
        question = state.get("question") or "Review requires user input."
        result = {
            "context": state.get("context"),
            "status": "HUMAN_INTERVENE",
            "question": question,
            "resume_mode": "boundary",
            "resume_state": self._boundary_resume_state(
                (state.get("context").request_id if state.get("context") else state.get("request", {}).get("request_id", "")),
                question,
                ["finish_human_intervene"],
            ),
        }
        self._emit_progress("finish_human_intervene", "done", "workflow waiting for human input", result)
        return result

    def _thread_config(self, request_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": request_id}}

    async def _await_graph_result(self, awaitable: Awaitable[Any], *, request_id: str, operation: str) -> Any:
        timeout_seconds = get_settings().plan_request_timeout_seconds
        try:
            return await asyncio.wait_for(awaitable, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            message = f"{operation} timed out request_id={request_id} timeout_seconds={timeout_seconds}"
            logger.error(message)
            self._emit_progress("workflow", "error", message)
            raise

    async def _normalize_interrupt_result(self, result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        interrupts = result.get("__interrupt__") or []
        if not interrupts:
            return result
        graph = await self._ensure_persistent_graph()
        snapshot = await graph.aget_state(config)
        interrupt_value = getattr(interrupts[0], "value", {}) if interrupts else {}
        question = interrupt_value.get("question") if isinstance(interrupt_value, dict) else str(interrupt_value)
        state_values = dict(snapshot.values or {})
        context = state_values.get("context")
        request_id = getattr(context, "request_id", None) or state_values.get("request", {}).get("request_id", config["configurable"]["thread_id"])
        state_values.update(
            {
                "status": "HUMAN_INTERVENE",
                "question": question,
                "resume_mode": "boundary",
                "resume_state": self._boundary_resume_state(
                    str(request_id),
                    str(question),
                    list(snapshot.next or []),
                    interrupt_id=getattr(interrupts[0], "id", None),
                ),
            }
        )
        return state_values

    def _boundary_resume_state(
        self,
        request_id: str,
        question: str,
        next_nodes: list[str],
        *,
        interrupt_id: str | None = None,
    ) -> dict[str, Any]:
        return {
            "mode": "boundary",
            "thread_id": request_id,
            "interrupt_id": interrupt_id,
            "question": question,
            "next": next_nodes,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def _apply_resume_payload(self, request: dict[str, Any], payload: Any) -> dict[str, Any]:
        data = copy.deepcopy(request)
        resume = payload if isinstance(payload, dict) else {}
        profile_updates = resume.get("profile_updates") or {}
        if profile_updates:
            profile = dict(data.get("profile") or {})
            profile.update(profile_updates)
            data["profile"] = profile
        if resume.get("user_message"):
            data["user_message"] = resume["user_message"]
        return PlanningRequest.model_validate(data).model_dump(mode="json")

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
    path = artifact_dir / f"{sanitize_request_id(request.request_id)}_travel_plan.md"
    itinerary = draft_packet["itinerary_draft"]
    weather = execution_results.get("WEATHER", {})
    budget = execution_results.get("BUDGET", {})
    accommodation = execution_results.get("ACCOMMODATION", {})
    flight = execution_results.get("FLIGHT_TRANSPORT", {})
    calendar = execution_results.get("CALENDAR", {})
    lines: list[str] = [f"# {draft_packet['destination']} Travel Plan", "", "## Overview", f"- Request ID: {request.request_id}", f"- Review Verdict: {review_packet['verdict']}", f"- Dashboard: {dashboard_url}", f"- Calendar File: {calendar.get('calendar_file', '')}", "", "## Data Sources", f"- Menxia Review: {review_packet.get('data_source', 'structured_llm')}"]
    for bureau_name in ("WEATHER", "BUDGET", "ACCOMMODATION", "FLIGHT_TRANSPORT", "CALENDAR"):
        result = execution_results.get(bureau_name)
        if not result:
            continue
        lines.append(f"- {bureau_name}: {result.get('status', 'unknown')} / {result.get('data_source', 'unavailable')}")
        warnings = result.get("warnings") or []
        if warnings:
            lines.append(f"  - Warnings: {'; '.join(str(item) for item in warnings)}")
    review_warnings = review_packet.get("warnings") or []
    if review_warnings:
        lines.append(f"- Menxia Warnings: {'; '.join(str(item) for item in review_warnings)}")
    lines.extend(["", "## Daily Itinerary"])
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
