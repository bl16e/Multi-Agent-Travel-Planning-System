from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class WorkflowState(StrEnum):
    DRAFT = "DRAFT"
    REVIEW = "REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    HUMAN_INTERVENE = "HUMAN_INTERVENE"
    EXECUTE = "EXECUTE"
    ASSEMBLE = "ASSEMBLE"
    DONE = "DONE"


STATES: dict[WorkflowState, list[WorkflowState]] = {
    WorkflowState.DRAFT: [WorkflowState.REVIEW],
    WorkflowState.REVIEW: [
        WorkflowState.APPROVED,
        WorkflowState.REJECTED,
        WorkflowState.HUMAN_INTERVENE,
    ],
    WorkflowState.APPROVED: [WorkflowState.EXECUTE],
    WorkflowState.REJECTED: [WorkflowState.DRAFT],
    WorkflowState.HUMAN_INTERVENE: [WorkflowState.REVIEW],
    WorkflowState.EXECUTE: [WorkflowState.ASSEMBLE],
    WorkflowState.ASSEMBLE: [WorkflowState.DONE],
    WorkflowState.DONE: [],
}


class InvalidTransitionError(ValueError):
    """Raised when a workflow transition violates the governance state machine."""


@dataclass(slots=True)
class TransitionRecord:
    from_state: WorkflowState
    to_state: WorkflowState
    actor: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


def validate_transition(
    current_state: WorkflowState | str,
    next_state: WorkflowState | str,
) -> bool:
    """Validate whether `current_state -> next_state` is legal."""

    source = WorkflowState(current_state)
    target = WorkflowState(next_state)
    allowed_targets = STATES.get(source, [])
    is_valid = target in allowed_targets

    if not is_valid:
        logger.warning(
            "Illegal workflow transition rejected: %s -> %s; allowed=%s",
            source,
            target,
            [item.value for item in allowed_targets],
        )

    return is_valid


@dataclass(slots=True)
class TravelWorkflowStateMachine:
    """
    Governance-oriented state machine for the three-province workflow.

    All provinces must transition through this machine instead of mutating
    workflow state directly.
    """

    current_state: WorkflowState = WorkflowState.DRAFT
    history: list[TransitionRecord] = field(default_factory=list)

    def can_transition_to(self, next_state: WorkflowState | str) -> bool:
        return validate_transition(self.current_state, next_state)

    def transition_to(
        self,
        next_state: WorkflowState | str,
        *,
        actor: str,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> TransitionRecord:
        target = WorkflowState(next_state)
        if not self.can_transition_to(target):
            raise InvalidTransitionError(
                f"Invalid transition: {self.current_state.value} -> {target.value}",
            )

        record = TransitionRecord(
            from_state=self.current_state,
            to_state=target,
            actor=actor,
            reason=reason,
            metadata=metadata or {},
        )
        self.history.append(record)
        self.current_state = target

        logger.info(
            "Workflow transition accepted: %s -> %s by %s",
            record.from_state.value,
            record.to_state.value,
            actor,
        )
        return record

    def export_history(self) -> list[dict[str, Any]]:
        return [
            {
                "from_state": item.from_state.value,
                "to_state": item.to_state.value,
                "actor": item.actor,
                "reason": item.reason,
                "metadata": item.metadata,
                "timestamp": item.timestamp.isoformat(),
            }
            for item in self.history
        ]
