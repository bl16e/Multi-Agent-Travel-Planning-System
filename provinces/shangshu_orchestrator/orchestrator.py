from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from langgraph.types import Send

from utils.permission_matrix import ActionType, AgentRole, enforce_permission
from utils.schemas import DepartmentTaskModel, ProgressEvent
from utils.state_machine import TravelWorkflowStateMachine, WorkflowState

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class DispatchBundle:
    sends: tuple[Send, ...]
    tasks: tuple[DepartmentTaskModel, ...]
    emitted_at: datetime


@dataclass(slots=True)
class ShangshuWorkflowContext:
    request_id: str
    user_request: dict[str, Any]
    dashboard_base_url: str = "http://127.0.0.1:8000/dashboard"
    current_state: WorkflowState = WorkflowState.DRAFT
    draft_payload: dict[str, Any] | None = None
    review_payload: dict[str, Any] | None = None
    execution_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    progress_events: list[dict[str, Any]] = field(default_factory=list)
    pending_user_inputs: list[str] = field(default_factory=list)
    assembled_output: dict[str, Any] | None = None
    rejection_count: int = 0
    max_rejection_rounds: int = 2


class ShangshuOrchestrator:
    def __init__(self) -> None:
        self._machines: dict[str, TravelWorkflowStateMachine] = {}

    def bootstrap(self, request_id: str, user_request: dict[str, Any]) -> ShangshuWorkflowContext:
        self._machines[request_id] = TravelWorkflowStateMachine()
        context = ShangshuWorkflowContext(
            request_id=request_id,
            user_request=user_request,
            current_state=self._machine_for(request_id).current_state,
        )
        self._record_progress(context, stage="bootstrap", message="Shangshu accepted the planning request.", actor=AgentRole.SHANGSHU)
        return context

    def dispatch_to_zhongshu(self, context: ShangshuWorkflowContext) -> DispatchBundle:
        enforce_permission(AgentRole.SHANGSHU, ActionType.SUBMIT_REQUEST, AgentRole.ZHONGSHU)
        task = DepartmentTaskModel(
            target=AgentRole.ZHONGSHU,
            task_type="plan_itinerary_draft",
            payload={
                "request_id": context.request_id,
                "user_request": context.user_request,
                "governance": {
                    "must_review_before_execution": True,
                    "must_use_structured_output": True,
                    "source_state": context.current_state.value,
                    "rejection_count": context.rejection_count,
                    "max_rejection_rounds": context.max_rejection_rounds,
                },
            },
            reason="Initial planning request enters Zhongshu for drafting.",
        )
        self._record_progress(context, stage="dispatch_zhongshu", message="Planning request dispatched to Zhongshu.", actor=AgentRole.SHANGSHU)
        return self._build_dispatch_bundle(task)

    def submit_draft_to_review(self, context: ShangshuWorkflowContext, draft_payload: dict[str, Any]) -> DispatchBundle:
        context.draft_payload = draft_payload
        if context.current_state == WorkflowState.REJECTED:
            self._transition(context, WorkflowState.DRAFT, actor=AgentRole.ZHONGSHU, reason="Zhongshu reopened the workflow with a revised draft.")
        self._transition(context, WorkflowState.REVIEW, actor=AgentRole.ZHONGSHU, reason="Zhongshu completed a draft and submitted it for review.")
        enforce_permission(AgentRole.ZHONGSHU, ActionType.SUBMIT_DRAFT_FOR_REVIEW, AgentRole.MENXIA)
        task = DepartmentTaskModel(
            target=AgentRole.MENXIA,
            task_type="review_itinerary_draft",
            payload={
                "request_id": context.request_id,
                "draft": draft_payload,
                "user_request": context.user_request,
                "governance": {"source_state": context.current_state.value, "review_required": True},
            },
            reason="Every draft must be reviewed by Menxia before execution.",
        )
        self._record_progress(context, stage="submit_review", message="Draft entered Menxia review.", actor=AgentRole.SHANGSHU)
        return self._build_dispatch_bundle(task)

    def apply_review_verdict(self, context: ShangshuWorkflowContext, review_payload: dict[str, Any]) -> DispatchBundle | None:
        verdict = str(review_payload.get("verdict", "")).upper()
        context.review_payload = review_payload

        if verdict == WorkflowState.APPROVED.value:
            enforce_permission(AgentRole.MENXIA, ActionType.RETURN_REVIEW_VERDICT, AgentRole.SHANGSHU)
            self._transition(context, WorkflowState.APPROVED, actor=AgentRole.MENXIA, reason="Menxia approved the draft.", metadata={"review_payload": review_payload})
            self._record_progress(context, stage="review_approved", message="Draft approved; Shangshu can dispatch Liubu execution.", actor=AgentRole.SHANGSHU)
            return None

        if verdict == WorkflowState.REJECTED.value:
            enforce_permission(AgentRole.MENXIA, ActionType.RETURN_REVISION_FEEDBACK, AgentRole.ZHONGSHU)
            context.rejection_count += 1
            self._transition(context, WorkflowState.REJECTED, actor=AgentRole.MENXIA, reason="Menxia rejected the draft and returned revision notes.", metadata={"review_payload": review_payload})
            retry_allowed = context.rejection_count < context.max_rejection_rounds
            status_note = "routed back to Zhongshu" if retry_allowed else "reached rejection limit"
            self._record_progress(context, stage="review_rejected", message=f"Draft rejected and {status_note}.", actor=AgentRole.SHANGSHU)
            task = DepartmentTaskModel(
                target=AgentRole.ZHONGSHU,
                task_type="revise_itinerary_draft",
                payload={
                    "request_id": context.request_id,
                    "draft": context.draft_payload or {},
                    "review_feedback": review_payload,
                    "user_request": context.user_request,
                    "governance": {
                        "source_state": context.current_state.value,
                        "must_fix_rejection_reasons": True,
                        "rejection_count": context.rejection_count,
                        "max_rejection_rounds": context.max_rejection_rounds,
                        "retry_allowed": retry_allowed,
                    },
                },
                reason="Menxia exercised veto and requested revisions.",
            )
            return self._build_dispatch_bundle(task)

        if verdict == WorkflowState.HUMAN_INTERVENE.value:
            self._transition(context, WorkflowState.HUMAN_INTERVENE, actor=AgentRole.MENXIA, reason="Menxia requested user intervention.", metadata={"review_payload": review_payload})
            question = str((review_payload.get("human_questions") or [review_payload.get("question") or "Review requires human intervention."])[0])
            context.pending_user_inputs.append(question)
            self._record_progress(context, stage="review_interrupt", message=question, actor=AgentRole.SHANGSHU)
            return None

        raise ValueError(f"Unsupported review verdict: {verdict}")

    def dispatch_liubu_execution(self, context: ShangshuWorkflowContext, execution_plan: dict[str, Any]) -> DispatchBundle:
        self._transition(context, WorkflowState.EXECUTE, actor=AgentRole.SHANGSHU, reason="Approved draft moved into execution dispatch.", metadata={"execution_plan": execution_plan})
        sends: list[Send] = []
        tasks: list[DepartmentTaskModel] = []
        for bureau in self._resolve_liubu_targets(execution_plan):
            enforce_permission(AgentRole.SHANGSHU, ActionType.DISPATCH_EXECUTION, bureau)
            task = DepartmentTaskModel(
                target=bureau,
                task_type="execute_specialist_task",
                payload={
                    "request_id": context.request_id,
                    "approved_draft": context.draft_payload or {},
                    "review_payload": context.review_payload or {},
                    "execution_plan": execution_plan,
                    "target_bureau": bureau.value,
                    "governance": {"source_state": context.current_state.value, "no_cross_bureau_send": True},
                },
                reason="Shangshu dispatches approved tasks to Liubu.",
            )
            tasks.append(task)
            sends.append(Send(self._graph_node_name_for(bureau), task.payload))
        self._record_progress(context, stage="dispatch_liubu", message=f"Execution dispatched to {len(tasks)} bureau(s).", actor=AgentRole.SHANGSHU)
        return DispatchBundle(sends=tuple(sends), tasks=tuple(tasks), emitted_at=datetime.now(timezone.utc))

    def register_execution_result(self, context: ShangshuWorkflowContext, bureau: AgentRole | str, result_payload: dict[str, Any]) -> None:
        role = AgentRole(bureau)
        enforce_permission(role, ActionType.RETURN_EXECUTION_RESULT, AgentRole.SHANGSHU)
        context.execution_results[role.value] = result_payload
        self._record_progress(context, stage="execution_result", message=f"{role.value} execution result received.", actor=role)

    def assemble_outputs(self, context: ShangshuWorkflowContext) -> dict[str, Any]:
        self._transition(context, WorkflowState.ASSEMBLE, actor=AgentRole.SHANGSHU, reason="All required Liubu responses received; assembling final outputs.")
        assembled = {
            "request_id": context.request_id,
            "workflow_state": WorkflowState.ASSEMBLE.value,
            "dashboard_url": self.build_dashboard_link(context),
            "draft": context.draft_payload or {},
            "review": context.review_payload or {},
            "execution_results": context.execution_results,
            "progress_events": context.progress_events,
            "state_history": self._machine_for(context.request_id).export_history(),
        }
        context.assembled_output = assembled
        self._record_progress(context, stage="assemble", message="Final package assembly completed.", actor=AgentRole.SHANGSHU)
        self._transition(context, WorkflowState.DONE, actor=AgentRole.SHANGSHU, reason="Final package is ready for delivery.")
        return assembled

    def build_dashboard_link(self, context: ShangshuWorkflowContext) -> str:
        return f"{context.dashboard_base_url}/{context.request_id}"

    def _build_dispatch_bundle(self, task: DepartmentTaskModel) -> DispatchBundle:
        return DispatchBundle(sends=(Send(self._graph_node_name_for(task.target), task.payload),), tasks=(task,), emitted_at=datetime.now(timezone.utc))

    def _resolve_liubu_targets(self, execution_plan: dict[str, Any]) -> list[AgentRole]:
        requested = execution_plan.get("required_bureaus", [])
        if not requested:
            return [AgentRole.WEATHER, AgentRole.CALENDAR, AgentRole.BUDGET, AgentRole.ACCOMMODATION, AgentRole.FLIGHT_TRANSPORT]
        return [AgentRole(item) for item in requested]

    def _graph_node_name_for(self, target: AgentRole) -> str:
        return {
            AgentRole.ZHONGSHU: "zhongshu_itinerary",
            AgentRole.MENXIA: "menxia_review",
            AgentRole.WEATHER: "liubu_weather",
            AgentRole.BUDGET: "liubu_budget",
            AgentRole.ACCOMMODATION: "liubu_accommodation",
            AgentRole.FLIGHT_TRANSPORT: "liubu_flight_transport",
            AgentRole.CALENDAR: "liubu_calendar",
        }[target]

    def _machine_for(self, request_id: str) -> TravelWorkflowStateMachine:
        if request_id not in self._machines:
            self._machines[request_id] = TravelWorkflowStateMachine()
        return self._machines[request_id]

    def _transition(self, context: ShangshuWorkflowContext, next_state: WorkflowState, *, actor: AgentRole, reason: str, metadata: dict[str, Any] | None = None) -> None:
        record = self._machine_for(context.request_id).transition_to(next_state, actor=actor.value, reason=reason, metadata=metadata)
        context.current_state = record.to_state

    def _record_progress(self, context: ShangshuWorkflowContext, *, stage: str, message: str, actor: AgentRole) -> None:
        event = ProgressEvent(stage=stage, message=message, actor=actor.value, state=context.current_state.value, timestamp=datetime.now(timezone.utc))
        context.progress_events.append(event.model_dump(mode="json"))
        logger.info("%s | %s", stage, message)
