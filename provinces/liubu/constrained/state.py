from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

LiubuBureauName = Literal["WEATHER", "BUDGET", "ACCOMMODATION", "FLIGHT_TRANSPORT", "CALENDAR"]
EvidenceStatus = Literal["ok", "error", "timeout", "blocked", "missing"]


class LiubuWorkerInput(BaseModel):
    request_id: str
    bureau: LiubuBureauName
    destination: str
    approved_draft: dict[str, Any]
    review_payload: dict[str, Any] = Field(default_factory=dict)
    execution_plan: dict[str, Any] = Field(default_factory=dict)
    profile: dict[str, Any] = Field(default_factory=dict)
    daily_plan: list[dict[str, Any]] = Field(default_factory=list)
    trip_dates: list[str] = Field(default_factory=list)
    bureau_task: dict[str, Any] = Field(default_factory=dict)
    review_notes: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)


class LiubuToolEvidence(BaseModel):
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    status: EvidenceStatus
    result: Any | None = None
    error: str | None = None
    data_source: str = "live"


class LiubuValidationFinding(BaseModel):
    severity: Literal["warning", "error"]
    code: str
    message: str
    field_path: str | None = None
    evidence_index: int | None = None


class LiubuWorkerState(TypedDict, total=False):
    payload: dict[str, Any]
    worker_input: LiubuWorkerInput
    messages: list[Any]
    tool_requests: list[dict[str, Any]]
    tool_evidence: list[dict[str, Any]]
    validation_findings: list[dict[str, Any]]
    result: dict[str, Any]
    status: str
    data_source: str
    warnings: list[str]


def normalize_worker_input(payload: dict[str, Any], bureau: LiubuBureauName | str) -> LiubuWorkerInput:
    approved_draft = dict(payload.get("approved_draft") or {})
    review_payload = dict(payload.get("review_payload") or {})
    execution_plan = dict(payload.get("execution_plan") or {})
    user_request = dict(execution_plan.get("user_request") or {})
    profile = dict(user_request.get("profile") or {})
    draft = dict(approved_draft.get("itinerary_draft") or {})
    daily_plan = list(draft.get("daily_plan") or [])
    trip_dates = _extract_trip_dates(daily_plan)
    start_date = str(profile.get("start_date") or (trip_dates[0] if trip_dates else ""))
    end_date = str(profile.get("end_date") or (trip_dates[-1] if trip_dates else start_date))
    destination = str(approved_draft.get("destination") or draft.get("destination") or _first_destination_preference(profile))
    constraints = {
        "origin_city": profile.get("origin_city"),
        "origin_airport_code": profile.get("origin_airport_code"),
        "destination_airport_code": profile.get("destination_airport_code"),
        "start_date": start_date,
        "end_date": end_date,
        "adults": int(profile.get("adults") or 1),
        "children": int(profile.get("children") or 0),
        "currency": profile.get("currency") or "USD",
        "constraints": list(profile.get("constraints") or []),
        "interests": list(profile.get("interests") or []),
    }
    return LiubuWorkerInput(
        request_id=str(payload.get("request_id") or user_request.get("request_id") or "trip"),
        bureau=str(bureau),  # type: ignore[arg-type]
        destination=destination,
        approved_draft=approved_draft,
        review_payload=review_payload,
        execution_plan=execution_plan,
        profile=profile,
        daily_plan=daily_plan,
        trip_dates=trip_dates,
        bureau_task=_bureau_task_for(approved_draft, str(bureau)),
        review_notes=list(review_payload.get("review_notes") or []),
        constraints=constraints,
    )


def _extract_trip_dates(daily_plan: list[dict[str, Any]]) -> list[str]:
    dates: list[str] = []
    for day in daily_plan:
        value = day.get("date")
        if value:
            dates.append(str(value))
    return dates


def _bureau_task_for(approved_draft: dict[str, Any], bureau: str) -> dict[str, Any]:
    for item in approved_draft.get("bureau_tasks") or []:
        if str(item.get("bureau")) == bureau:
            return dict(item)
    return {}


def _first_destination_preference(profile: dict[str, Any]) -> str:
    preferences = profile.get("destination_preferences") or []
    return str(preferences[0]) if preferences else "Unknown Destination"
