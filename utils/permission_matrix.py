from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)


class AgentRole(StrEnum):
    USER = "USER"
    SHANGSHU = "SHANGSHU"
    ZHONGSHU = "ZHONGSHU"
    MENXIA = "MENXIA"
    WEATHER = "WEATHER"
    BUDGET = "BUDGET"
    ACCOMMODATION = "ACCOMMODATION"
    FLIGHT_TRANSPORT = "FLIGHT_TRANSPORT"
    CALENDAR = "CALENDAR"


class ActionType(StrEnum):
    SUBMIT_REQUEST = "SUBMIT_REQUEST"
    SUBMIT_DRAFT_FOR_REVIEW = "SUBMIT_DRAFT_FOR_REVIEW"
    RETURN_REVIEW_VERDICT = "RETURN_REVIEW_VERDICT"
    RETURN_REVISION_FEEDBACK = "RETURN_REVISION_FEEDBACK"
    DISPATCH_EXECUTION = "DISPATCH_EXECUTION"
    RETURN_EXECUTION_RESULT = "RETURN_EXECUTION_RESULT"
    REQUEST_HUMAN_INTERVENTION = "REQUEST_HUMAN_INTERVENTION"


LIUBU_ROLES: frozenset[AgentRole] = frozenset(
    {
        AgentRole.WEATHER,
        AgentRole.BUDGET,
        AgentRole.ACCOMMODATION,
        AgentRole.FLIGHT_TRANSPORT,
        AgentRole.CALENDAR,
    },
)


PERMISSION_MATRIX: dict[AgentRole, dict[ActionType, set[AgentRole]]] = {
    AgentRole.USER: {
        ActionType.SUBMIT_REQUEST: {AgentRole.SHANGSHU},
    },
    AgentRole.SHANGSHU: {
        ActionType.SUBMIT_REQUEST: {AgentRole.ZHONGSHU},
        ActionType.DISPATCH_EXECUTION: set(LIUBU_ROLES),
        ActionType.REQUEST_HUMAN_INTERVENTION: {AgentRole.USER},
    },
    AgentRole.ZHONGSHU: {
        ActionType.SUBMIT_DRAFT_FOR_REVIEW: {AgentRole.MENXIA},
        ActionType.REQUEST_HUMAN_INTERVENTION: {AgentRole.SHANGSHU},
    },
    AgentRole.MENXIA: {
        ActionType.RETURN_REVIEW_VERDICT: {AgentRole.SHANGSHU},
        ActionType.RETURN_REVISION_FEEDBACK: {AgentRole.ZHONGSHU},
        ActionType.REQUEST_HUMAN_INTERVENTION: {AgentRole.SHANGSHU},
    },
    AgentRole.WEATHER: {
        ActionType.RETURN_EXECUTION_RESULT: {AgentRole.SHANGSHU},
    },
    AgentRole.BUDGET: {
        ActionType.RETURN_EXECUTION_RESULT: {AgentRole.SHANGSHU},
    },
    AgentRole.ACCOMMODATION: {
        ActionType.RETURN_EXECUTION_RESULT: {AgentRole.SHANGSHU},
    },
    AgentRole.FLIGHT_TRANSPORT: {
        ActionType.RETURN_EXECUTION_RESULT: {AgentRole.SHANGSHU},
    },
    AgentRole.CALENDAR: {
        ActionType.RETURN_EXECUTION_RESULT: {AgentRole.SHANGSHU},
    },
}


class PermissionDeniedError(PermissionError):
    """Raised when an agent attempts an illegal cross-role action."""


@dataclass(frozen=True, slots=True)
class PermissionCheckResult:
    allowed: bool
    actor: AgentRole
    action: ActionType
    target: AgentRole
    reason: str


def is_liubu(role: AgentRole | str) -> bool:
    return AgentRole(role) in LIUBU_ROLES


def validate_permission(
    actor: AgentRole | str,
    action: ActionType | str,
    target: AgentRole | str,
) -> PermissionCheckResult:
    source = AgentRole(actor)
    verb = ActionType(action)
    destination = AgentRole(target)

    allowed_targets = PERMISSION_MATRIX.get(source, {}).get(verb, set())
    allowed = destination in allowed_targets

    if is_liubu(source) and is_liubu(destination):
        allowed = False
        reason = "Liubu cross-department direct SEND is forbidden."
    elif allowed:
        reason = "Permission granted by governance matrix."
    else:
        reason = "Permission denied by governance matrix."

    if not allowed:
        logger.warning(
            "Illegal permission rejected: actor=%s action=%s target=%s",
            source.value,
            verb.value,
            destination.value,
        )

    return PermissionCheckResult(
        allowed=allowed,
        actor=source,
        action=verb,
        target=destination,
        reason=reason,
    )


def enforce_permission(
    actor: AgentRole | str,
    action: ActionType | str,
    target: AgentRole | str,
) -> None:
    result = validate_permission(actor, action, target)
    if not result.allowed:
        raise PermissionDeniedError(
            f"{result.actor.value} cannot {result.action.value} -> "
            f"{result.target.value}: {result.reason}"
        )
