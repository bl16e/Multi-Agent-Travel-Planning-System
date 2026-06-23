# Liubu LangGraph-Native Constrained Agents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor Liubu Flight Transport and Accommodation into LangGraph-native constrained specialist agents with graph-visible tool evidence, argument validation, quality gates, and compatible public outputs.

**Architecture:** Add a shared constrained-worker layer under `provinces/liubu/constrained/` that normalizes worker input, captures tool evidence, validates agent tool arguments before execution, and gates results before Shangshu assembly. Keep current FastAPI routes, workflow state shape, existing result schemas, and deterministic fallback behavior unchanged.

**Tech Stack:** Python 3.11, LangGraph 0.6.11 `StateGraph`, LangChain chat models/tools, Pydantic v2, pytest, pytest-asyncio, existing MCP client helpers.

---

## File Structure

- Create: `provinces/liubu/constrained/__init__.py`
  - Exports shared state, tool, and gate helpers for bureau service modules.
- Create: `provinces/liubu/constrained/state.py`
  - Defines `LiubuWorkerInput`, `LiubuToolEvidence`, `LiubuValidationFinding`, `LiubuWorkerState`, and `normalize_worker_input`.
- Create: `provinces/liubu/constrained/tools.py`
  - Loads allowlisted MCP tools, validates tool names and arguments against normalized trip constraints, calls tools with timeout, and returns evidence records.
- Create: `provinces/liubu/constrained/gates.py`
  - Contains shared and bureau-specific quality gate functions for Flight Transport and Accommodation.
- Create: `tests/test_liubu_constrained_state.py`
  - Covers worker input normalization and preservation of non-negotiable trip facts.
- Create: `tests/test_liubu_constrained_tools.py`
  - Covers unsupported tools, wrong dates, wrong airport codes, wrong currency, timeouts, and sanitized evidence.
- Create: `tests/test_liubu_constrained_gates.py`
  - Covers Flight and Accommodation quality gate success/failure cases.
- Modify: `utils/schemas.py`
  - Adds optional `liubu_evidence` and `liubu_quality` fields to migrated bureau result models so final package serialization preserves graph-visible metadata.
- Modify: `provinces/liubu/flight_transport/service.py`
  - Replace black-box `research_transport` with constrained agent graph nodes while preserving `FlightTransportBureau.run(payload) -> dict`.
- Modify: `provinces/liubu/accommodation/service.py`
  - Replace black-box `research_accommodation` with constrained agent graph nodes while preserving `AccommodationBureau.run(payload) -> dict`.
- Modify: `provinces/shangshu_orchestrator/orchestrator.py`
  - Record Liubu quality gate metadata in `ShangshuWorkflowContext` and progress events without changing public API status values.
- Modify: `workflow.py`
  - Pass bureau results through the existing `register_execution_result`; no route or final package schema changes.
- Modify: `tests/test_bureaus.py`
  - Update existing bureau fallback tests so they accept evidence-backed constrained fallbacks and still assert compatibility.
- Modify: `tests/test_orchestrator.py`
  - Cover quality gate metadata recording.
- Modify: `tests/test_live_e2e.py`
  - Assert live terminal states include evidence or fallback labels for migrated bureaus.

## Implementation Tasks

### Task 1: Shared Liubu Worker State

**Files:**
- Create: `tests/test_liubu_constrained_state.py`
- Create: `provinces/liubu/constrained/__init__.py`
- Create: `provinces/liubu/constrained/state.py`

- [ ] **Step 1: Write failing normalization tests**

Create `tests/test_liubu_constrained_state.py`:

```python
from provinces.liubu.constrained.state import normalize_worker_input


def _payload():
    return {
        "request_id": "liubu_state",
        "target_bureau": "FLIGHT_TRANSPORT",
        "approved_draft": {
            "destination": "Tokyo",
            "itinerary_draft": {
                "destination": "Tokyo",
                "daily_plan": [
                    {"day_index": 1, "date": "2026-10-01", "activities": []},
                    {"day_index": 2, "date": "2026-10-02", "activities": []},
                ],
            },
            "bureau_tasks": [
                {
                    "bureau": "FLIGHT_TRANSPORT",
                    "objective": "Find flights",
                    "inputs_required": ["origin_city", "destination"],
                    "deliverables": ["flight_options"],
                    "priority": "high",
                }
            ],
        },
        "review_payload": {"review_notes": ["Approved after review."]},
        "execution_plan": {
            "user_request": {
                "profile": {
                    "origin_city": "Beijing",
                    "origin_airport_code": "PEK",
                    "destination_airport_code": "HND",
                    "destination_preferences": ["Tokyo"],
                    "start_date": "2026-10-01",
                    "end_date": "2026-10-02",
                    "adults": 2,
                    "children": 1,
                    "currency": "USD",
                    "constraints": ["avoid red-eye flights"],
                    "interests": ["culture"],
                }
            }
        },
    }


def test_normalize_worker_input_preserves_trip_constraints():
    worker_input = normalize_worker_input(_payload(), "FLIGHT_TRANSPORT")

    assert worker_input.request_id == "liubu_state"
    assert worker_input.bureau == "FLIGHT_TRANSPORT"
    assert worker_input.destination == "Tokyo"
    assert worker_input.profile["origin_city"] == "Beijing"
    assert worker_input.constraints["origin_airport_code"] == "PEK"
    assert worker_input.constraints["destination_airport_code"] == "HND"
    assert worker_input.constraints["start_date"] == "2026-10-01"
    assert worker_input.constraints["end_date"] == "2026-10-02"
    assert worker_input.constraints["adults"] == 2
    assert worker_input.constraints["children"] == 1
    assert worker_input.constraints["currency"] == "USD"
    assert worker_input.trip_dates == ["2026-10-01", "2026-10-02"]
    assert worker_input.bureau_task["objective"] == "Find flights"
    assert worker_input.review_notes == ["Approved after review."]


def test_normalize_worker_input_uses_itinerary_dates_when_profile_dates_missing():
    payload = _payload()
    payload["execution_plan"]["user_request"]["profile"].pop("start_date")
    payload["execution_plan"]["user_request"]["profile"].pop("end_date")

    worker_input = normalize_worker_input(payload, "FLIGHT_TRANSPORT")

    assert worker_input.constraints["start_date"] == "2026-10-01"
    assert worker_input.constraints["end_date"] == "2026-10-02"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_liubu_constrained_state.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'provinces.liubu.constrained'`.

- [ ] **Step 3: Add shared state models and normalization**

Create `provinces/liubu/constrained/__init__.py`:

```python
from provinces.liubu.constrained.state import (
    LiubuToolEvidence,
    LiubuValidationFinding,
    LiubuWorkerInput,
    LiubuWorkerState,
    normalize_worker_input,
)

__all__ = [
    "LiubuToolEvidence",
    "LiubuValidationFinding",
    "LiubuWorkerInput",
    "LiubuWorkerState",
    "normalize_worker_input",
]
```

Create `provinces/liubu/constrained/state.py`:

```python
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
```

- [ ] **Step 4: Run state tests**

Run:

```bash
python -m pytest tests/test_liubu_constrained_state.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit state slice**

Run:

```bash
git add provinces/liubu/constrained/__init__.py provinces/liubu/constrained/state.py tests/test_liubu_constrained_state.py
git commit -m "feat: add liubu constrained worker state"
```

### Task 2: Constrained Tool Execution and Evidence Capture

**Files:**
- Create: `tests/test_liubu_constrained_tools.py`
- Create: `provinces/liubu/constrained/tools.py`

- [ ] **Step 1: Write failing tool constraint tests**

Create `tests/test_liubu_constrained_tools.py`:

```python
import asyncio

import pytest

from provinces.liubu.constrained.state import normalize_worker_input
from provinces.liubu.constrained.tools import execute_constrained_tool_call


class FakeTool:
    name = "google_flights"

    async def ainvoke(self, args):
        return {
            "search_parameters": args,
            "best_flights": [{"airline": "ANA", "price": 500, "departure_airport": "PEK", "arrival_airport": "HND"}],
            "secret": "must not be retained",
        }


class SlowTool:
    name = "google_flights"

    async def ainvoke(self, args):
        await asyncio.sleep(1)
        return {"ok": True}


def _worker_input():
    payload = {
        "request_id": "tool_constraints",
        "target_bureau": "FLIGHT_TRANSPORT",
        "approved_draft": {
            "destination": "Tokyo",
            "itinerary_draft": {"destination": "Tokyo", "daily_plan": [{"date": "2026-10-01", "activities": []}]},
        },
        "execution_plan": {
            "user_request": {
                "profile": {
                    "origin_city": "Beijing",
                    "origin_airport_code": "PEK",
                    "destination_airport_code": "HND",
                    "start_date": "2026-10-01",
                    "end_date": "2026-10-01",
                    "adults": 2,
                    "currency": "USD",
                }
            }
        },
    }
    return normalize_worker_input(payload, "FLIGHT_TRANSPORT")


@pytest.mark.asyncio
async def test_execute_constrained_tool_call_blocks_unsupported_tool():
    evidence = await execute_constrained_tool_call(
        worker_input=_worker_input(),
        tool_map={"google_flights": FakeTool()},
        allowed_tool_names={"google_flights"},
        tool_name="google_hotels",
        args={},
    )

    assert evidence.status == "blocked"
    assert evidence.error == "Tool google_hotels is not allowed for FLIGHT_TRANSPORT."


@pytest.mark.asyncio
async def test_execute_constrained_tool_call_blocks_wrong_airport_date_adults_and_currency():
    evidence = await execute_constrained_tool_call(
        worker_input=_worker_input(),
        tool_map={"google_flights": FakeTool()},
        allowed_tool_names={"google_flights"},
        tool_name="google_flights",
        args={
            "departure_id": "SHA",
            "arrival_id": "NRT",
            "outbound_date": "2023-10-01",
            "adults": 1,
            "currency": "JPY",
        },
    )

    assert evidence.status == "blocked"
    assert "departure_id must match PEK" in evidence.error
    assert "arrival_id must match HND" in evidence.error
    assert "outbound_date must match 2026-10-01" in evidence.error
    assert "adults must match 2" in evidence.error
    assert "currency must match USD" in evidence.error


@pytest.mark.asyncio
async def test_execute_constrained_tool_call_records_sanitized_success():
    evidence = await execute_constrained_tool_call(
        worker_input=_worker_input(),
        tool_map={"google_flights": FakeTool()},
        allowed_tool_names={"google_flights"},
        tool_name="google_flights",
        args={"departure_id": "PEK", "arrival_id": "HND", "outbound_date": "2026-10-01", "adults": 2, "currency": "USD", "api_key": "abc"},
    )

    assert evidence.status == "ok"
    assert evidence.args["api_key"] == "[REDACTED]"
    assert evidence.result["search_parameters"]["departure_id"] == "PEK"
    assert "secret" not in evidence.result


@pytest.mark.asyncio
async def test_execute_constrained_tool_call_records_timeout():
    evidence = await execute_constrained_tool_call(
        worker_input=_worker_input(),
        tool_map={"google_flights": SlowTool()},
        allowed_tool_names={"google_flights"},
        tool_name="google_flights",
        args={"departure_id": "PEK", "arrival_id": "HND", "outbound_date": "2026-10-01", "adults": 2, "currency": "USD"},
        timeout_seconds=0.01,
    )

    assert evidence.status == "timeout"
    assert evidence.error == "Tool google_flights timed out after 0.01s."
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_liubu_constrained_tools.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'provinces.liubu.constrained.tools'`.

- [ ] **Step 3: Add constrained tool execution helper**

Create `provinces/liubu/constrained/tools.py`:

```python
from __future__ import annotations

import asyncio
from typing import Any, Iterable

from provinces.liubu.constrained.state import LiubuToolEvidence, LiubuWorkerInput
from utils.agent_runtime import _compact_mcp_tool_result
from utils.mcp_client import load_mcp_tools

SENSITIVE_ARG_KEYS = {"api_key", "apikey", "key", "token", "authorization", "serpapi_api_key"}


async def load_allowed_tool_map(server_names: list[str], allowed_tool_names: set[str]) -> dict[str, Any]:
    tools = await load_mcp_tools(server_names)
    return {getattr(tool, "name", ""): tool for tool in tools if getattr(tool, "name", "") in allowed_tool_names}


async def execute_constrained_tool_call(
    *,
    worker_input: LiubuWorkerInput,
    tool_map: dict[str, Any],
    allowed_tool_names: set[str],
    tool_name: str,
    args: dict[str, Any],
    timeout_seconds: float = 15.0,
) -> LiubuToolEvidence:
    sanitized_args = sanitize_args(args)
    if tool_name not in allowed_tool_names:
        return LiubuToolEvidence(tool_name=tool_name, args=sanitized_args, status="blocked", error=f"Tool {tool_name} is not allowed for {worker_input.bureau}.")
    tool = tool_map.get(tool_name)
    if tool is None:
        return LiubuToolEvidence(tool_name=tool_name, args=sanitized_args, status="missing", error=f"Tool {tool_name} is not available.")
    errors = validate_tool_args(worker_input, tool_name, args)
    if errors:
        return LiubuToolEvidence(tool_name=tool_name, args=sanitized_args, status="blocked", error="; ".join(errors))
    try:
        result = await asyncio.wait_for(tool.ainvoke(args), timeout=timeout_seconds)
        return LiubuToolEvidence(tool_name=tool_name, args=sanitized_args, status="ok", result=sanitize_result(_compact_mcp_tool_result(result)), data_source="live")
    except asyncio.TimeoutError:
        return LiubuToolEvidence(tool_name=tool_name, args=sanitized_args, status="timeout", error=f"Tool {tool_name} timed out after {timeout_seconds:.2f}s.")
    except Exception as exc:
        return LiubuToolEvidence(tool_name=tool_name, args=sanitized_args, status="error", error=str(exc))


def validate_tool_args(worker_input: LiubuWorkerInput, tool_name: str, args: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    constraints = worker_input.constraints
    start_date = str(constraints.get("start_date") or "")
    origin_airport = constraints.get("origin_airport_code")
    destination_airport = constraints.get("destination_airport_code")
    adults = constraints.get("adults")
    currency = constraints.get("currency")
    if worker_input.bureau == "FLIGHT_TRANSPORT":
        _require_equal(errors, args, ("departure_id", "origin_airport", "from_airport"), origin_airport)
        _require_equal(errors, args, ("arrival_id", "destination_airport", "to_airport"), destination_airport)
        _require_equal(errors, args, ("outbound_date", "departure_date", "date"), start_date)
        _require_equal(errors, args, ("adults", "adult_count"), adults)
        _require_equal(errors, args, ("currency", "hl_currency"), currency)
    if worker_input.bureau == "ACCOMMODATION":
        _require_equal(errors, args, ("check_in_date", "check_in", "checkin_date"), start_date)
        _require_equal(errors, args, ("check_out_date", "check_out", "checkout_date"), str(constraints.get("end_date") or start_date))
        _require_equal(errors, args, ("adults", "adult_count"), adults)
        _require_equal(errors, args, ("currency", "hl_currency"), currency)
    return errors


def sanitize_args(args: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in args.items():
        if key.lower() in SENSITIVE_ARG_KEYS:
            clean[key] = "[REDACTED]"
        else:
            clean[key] = value
    return clean


def sanitize_result(result: Any) -> Any:
    if isinstance(result, dict):
        return {key: sanitize_result(value) for key, value in result.items() if key.lower() not in SENSITIVE_ARG_KEYS and key != "secret"}
    if isinstance(result, list):
        return [sanitize_result(item) for item in result[:8]]
    if isinstance(result, str):
        return result[:2000]
    return result


def _require_equal(errors: list[str], args: dict[str, Any], keys: Iterable[str], expected: Any) -> None:
    if expected in (None, ""):
        return
    for key in keys:
        if key in args and str(args[key]) != str(expected):
            errors.append(f"{key} must match {expected}")
```

- [ ] **Step 4: Run constrained tool tests**

Run:

```bash
python -m pytest tests/test_liubu_constrained_tools.py -q
```

Expected: PASS.

- [ ] **Step 5: Run existing agent runtime tests**

Run:

```bash
python -m pytest tests/test_agent_runtime.py tests/test_liubu_constrained_tools.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit tool slice**

Run:

```bash
git add provinces/liubu/constrained/tools.py tests/test_liubu_constrained_tools.py
git commit -m "feat: constrain liubu tool execution"
```

### Task 3: Flight and Accommodation Quality Gates

**Files:**
- Create: `tests/test_liubu_constrained_gates.py`
- Create: `provinces/liubu/constrained/gates.py`
- Modify: `utils/schemas.py`

- [ ] **Step 1: Write failing quality gate tests**

Create `tests/test_liubu_constrained_gates.py`:

```python
from provinces.liubu.constrained.gates import gate_accommodation_result, gate_flight_transport_result
from provinces.liubu.constrained.state import LiubuToolEvidence, normalize_worker_input


def _flight_input():
    return normalize_worker_input(
        {
            "request_id": "flight_gate",
            "approved_draft": {"destination": "Tokyo", "itinerary_draft": {"destination": "Tokyo", "daily_plan": [{"date": "2026-10-01", "activities": []}]}},
            "execution_plan": {"user_request": {"profile": {"origin_city": "Beijing", "origin_airport_code": "PEK", "destination_airport_code": "HND", "start_date": "2026-10-01", "end_date": "2026-10-01", "adults": 1, "currency": "USD"}}},
        },
        "FLIGHT_TRANSPORT",
    )


def _hotel_input():
    return normalize_worker_input(
        {
            "request_id": "hotel_gate",
            "approved_draft": {"destination": "Tokyo", "itinerary_draft": {"destination": "Tokyo", "daily_plan": [{"date": "2026-10-01", "activities": []}, {"date": "2026-10-02", "activities": []}]}},
            "execution_plan": {"user_request": {"profile": {"start_date": "2026-10-01", "end_date": "2026-10-02", "adults": 2, "currency": "USD"}}},
        },
        "ACCOMMODATION",
    )


def test_flight_gate_accepts_live_evidence_with_matching_airports_currency_and_date():
    findings = gate_flight_transport_result(
        _flight_input(),
        {
            "bureau": "FLIGHT_TRANSPORT",
            "status": "ok",
            "data_source": "live",
            "origin": "Beijing",
            "destination": "Tokyo",
            "flight_options": [{"airline": "ANA", "price": 500, "currency": "USD", "departure_airport": "PEK", "arrival_airport": "HND", "departure_time": "2026-10-01 08:00", "arrival_time": "2026-10-01 12:00"}],
            "transport_notes": [],
            "booking_links": [],
        },
        [LiubuToolEvidence(tool_name="google_flights", status="ok", args={"departure_id": "PEK", "arrival_id": "HND", "outbound_date": "2026-10-01"})],
    )

    assert findings == []


def test_flight_gate_rejects_clean_success_without_live_evidence_and_matching_facts():
    findings = gate_flight_transport_result(
        _flight_input(),
        {
            "bureau": "FLIGHT_TRANSPORT",
            "status": "ok",
            "data_source": "structured_llm",
            "origin": "Beijing",
            "destination": "Tokyo",
            "flight_options": [{"airline": "ANA", "price": 500, "currency": "JPY", "departure_airport": "SHA", "arrival_airport": "NRT", "departure_time": "2023-10-01 08:00", "arrival_time": "2023-10-01 12:00"}],
            "transport_notes": [],
            "booking_links": [],
        },
        [],
    )

    codes = {item.code for item in findings}
    assert {"missing_live_evidence", "wrong_departure_airport", "wrong_arrival_airport", "wrong_currency", "wrong_departure_date"} <= codes


def test_accommodation_gate_rejects_agent_hotel_date_conflicts():
    findings = gate_accommodation_result(
        _hotel_input(),
        {
            "bureau": "ACCOMMODATION",
            "status": "ok",
            "data_source": "live",
            "destination": "Tokyo",
            "hotel_options": [{"name": "Hotel", "currency": "USD", "notes": "check_in_date=2023-10-01 check_out_date=2023-10-02"}],
            "booking_links": [],
            "search_notes": [],
            "warnings": [],
        },
        [LiubuToolEvidence(tool_name="google_hotels", status="blocked", args={"check_in_date": "2023-10-01"}, error="check_in_date must match 2026-10-01")],
    )

    codes = {item.code for item in findings}
    assert "blocked_tool_call" in codes
    assert "hotel_date_conflict" in codes


def test_migrated_result_models_preserve_liubu_metadata():
    from utils.schemas import AccommodationExecutionResult, FlightTransportExecutionResult

    flight = FlightTransportExecutionResult.model_validate(
        {
            "bureau": "FLIGHT_TRANSPORT",
            "status": "fallback",
            "data_source": "fallback_estimate",
            "origin": "Beijing",
            "destination": "Tokyo",
            "flight_options": [],
            "transport_notes": [],
            "booking_links": [],
            "liubu_evidence": [{"tool_name": "google_flights", "status": "blocked"}],
            "liubu_quality": {"passed": False, "findings": [{"code": "wrong_departure_date"}]},
        }
    ).model_dump(mode="json")
    accommodation = AccommodationExecutionResult.model_validate(
        {
            "bureau": "ACCOMMODATION",
            "status": "fallback",
            "data_source": "fallback_estimate",
            "destination": "Tokyo",
            "hotel_options": [],
            "booking_links": [],
            "search_notes": [],
            "warnings": [],
            "liubu_evidence": [{"tool_name": "google_hotels", "status": "blocked"}],
            "liubu_quality": {"passed": False, "findings": [{"code": "hotel_date_conflict"}]},
        }
    ).model_dump(mode="json")

    assert flight["liubu_quality"]["passed"] is False
    assert flight["liubu_evidence"][0]["tool_name"] == "google_flights"
    assert accommodation["liubu_quality"]["findings"][0]["code"] == "hotel_date_conflict"
    assert accommodation["liubu_evidence"][0]["tool_name"] == "google_hotels"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_liubu_constrained_gates.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'provinces.liubu.constrained.gates'` or with missing `liubu_quality` after the gates module exists.

- [ ] **Step 3: Add optional Liubu metadata fields to migrated result models**

Modify `utils/schemas.py`:

```python
class AccommodationExecutionResult(BaseModel):
    bureau: Literal["ACCOMMODATION"] = "ACCOMMODATION"
    status: ExecutionStatus = "fallback"
    data_source: DataSource = "fallback_estimate"
    destination: str
    hotel_options: list[HotelOptionModel] = Field(default_factory=list)
    booking_links: list[HttpUrl | str] = Field(default_factory=list)
    search_notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    liubu_evidence: list[dict[str, Any]] = Field(default_factory=list)
    liubu_quality: dict[str, Any] = Field(default_factory=dict)
```

```python
class FlightTransportExecutionResult(BaseModel):
    bureau: Literal["FLIGHT_TRANSPORT"] = "FLIGHT_TRANSPORT"
    status: ExecutionStatus = "fallback"
    data_source: DataSource = "fallback_estimate"
    origin: str
    destination: str
    flight_options: list[FlightOptionModel] = Field(default_factory=list)
    transport_notes: list[str] = Field(default_factory=list)
    booking_links: list[HttpUrl | str] = Field(default_factory=list)
    liubu_evidence: list[dict[str, Any]] = Field(default_factory=list)
    liubu_quality: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 4: Add quality gate functions**

Create `provinces/liubu/constrained/gates.py`:

```python
from __future__ import annotations

from typing import Any

from provinces.liubu.constrained.state import LiubuToolEvidence, LiubuValidationFinding, LiubuWorkerInput
from utils.schemas import AccommodationExecutionResult, FlightTransportExecutionResult


def gate_flight_transport_result(worker_input: LiubuWorkerInput, result: dict[str, Any], evidence: list[LiubuToolEvidence]) -> list[LiubuValidationFinding]:
    findings = _shared_findings(worker_input, result, evidence, live_required=True)
    try:
        payload = FlightTransportExecutionResult.model_validate(result)
    except Exception as exc:
        return findings + [_finding("error", "schema_invalid", str(exc))]
    expected_departure = str(worker_input.constraints.get("origin_airport_code") or "")
    expected_arrival = str(worker_input.constraints.get("destination_airport_code") or "")
    expected_currency = str(worker_input.constraints.get("currency") or "")
    expected_date = str(worker_input.constraints.get("start_date") or "")
    for index, option in enumerate(payload.flight_options):
        if expected_departure and option.departure_airport != expected_departure:
            findings.append(_finding("error", "wrong_departure_airport", f"flight_options[{index}].departure_airport must be {expected_departure}", f"flight_options.{index}.departure_airport"))
        if expected_arrival and option.arrival_airport != expected_arrival:
            findings.append(_finding("error", "wrong_arrival_airport", f"flight_options[{index}].arrival_airport must be {expected_arrival}", f"flight_options.{index}.arrival_airport"))
        if expected_currency and option.currency != expected_currency:
            findings.append(_finding("error", "wrong_currency", f"flight_options[{index}].currency must be {expected_currency}", f"flight_options.{index}.currency"))
        if expected_date and not option.departure_time.startswith(expected_date):
            findings.append(_finding("error", "wrong_departure_date", f"flight_options[{index}].departure_time must start with {expected_date}", f"flight_options.{index}.departure_time"))
    return findings


def gate_accommodation_result(worker_input: LiubuWorkerInput, result: dict[str, Any], evidence: list[LiubuToolEvidence]) -> list[LiubuValidationFinding]:
    findings = _shared_findings(worker_input, result, evidence, live_required=True)
    try:
        payload = AccommodationExecutionResult.model_validate(result)
    except Exception as exc:
        return findings + [_finding("error", "schema_invalid", str(exc))]
    expected_start = str(worker_input.constraints.get("start_date") or "")
    expected_end = str(worker_input.constraints.get("end_date") or expected_start)
    for index, hotel in enumerate(payload.hotel_options):
        notes = str(hotel.notes or "")
        if "2023-10-01" in notes or (expected_start and f"check_in_date={expected_start}" not in notes and "check_in_date=" in notes):
            findings.append(_finding("error", "hotel_date_conflict", f"hotel_options[{index}] notes contain check-in date outside {expected_start}", f"hotel_options.{index}.notes"))
        if expected_end and "check_out_date=" in notes and f"check_out_date={expected_end}" not in notes:
            findings.append(_finding("error", "hotel_date_conflict", f"hotel_options[{index}] notes contain check-out date outside {expected_end}", f"hotel_options.{index}.notes"))
    return findings


def result_passed_gate(findings: list[LiubuValidationFinding]) -> bool:
    return not any(item.severity == "error" for item in findings)


def _shared_findings(worker_input: LiubuWorkerInput, result: dict[str, Any], evidence: list[LiubuToolEvidence], *, live_required: bool) -> list[LiubuValidationFinding]:
    findings: list[LiubuValidationFinding] = []
    if result.get("status") == "ok" and live_required and not any(item.status == "ok" for item in evidence):
        findings.append(_finding("error", "missing_live_evidence", f"{worker_input.bureau} status=ok requires successful tool evidence."))
    if result.get("status") in {"fallback", "error"} and result.get("data_source") not in {"fallback_estimate", "unavailable"}:
        findings.append(_finding("error", "fallback_source_invalid", "Fallback or error result must use fallback_estimate or unavailable data_source."))
    for index, item in enumerate(evidence):
        if item.status in {"blocked", "error", "timeout", "missing"}:
            findings.append(_finding("warning", f"{item.status}_tool_call", item.error or f"{item.tool_name} returned {item.status}.", evidence_index=index))
    return findings


def _finding(severity: str, code: str, message: str, field_path: str | None = None, evidence_index: int | None = None) -> LiubuValidationFinding:
    return LiubuValidationFinding(severity=severity, code=code, message=message, field_path=field_path, evidence_index=evidence_index)  # type: ignore[arg-type]
```

- [ ] **Step 4: Run quality gate tests**

Run:

```bash
python -m pytest tests/test_liubu_constrained_gates.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit gate slice**

Run:

```bash
git add provinces/liubu/constrained/gates.py tests/test_liubu_constrained_gates.py utils/schemas.py
git commit -m "feat: gate liubu constrained results"
```

### Task 4: Migrate Flight Transport Bureau

**Files:**
- Modify: `provinces/liubu/flight_transport/service.py`
- Modify: `tests/test_bureaus.py`

- [ ] **Step 1: Write failing Flight-specific constrained tests**

Append to `tests/test_bureaus.py`:

```python
@pytest.mark.asyncio
async def test_flight_transport_blocks_agent_tool_call_with_wrong_trip_facts(monkeypatch):
    class FakeAgent:
        async def ainvoke(self, state):
            return {
                "tool_requests": [
                    {
                        "tool": "google_flights",
                        "args": {
                            "departure_id": "SHA",
                            "arrival_id": "NRT",
                            "outbound_date": "2023-10-01",
                            "adults": 1,
                            "currency": "JPY",
                        },
                    }
                ]
            }

    monkeypatch.setattr(flight_service, "build_qwen_chat", lambda: None)
    async def fake_tool_map(server_names, allowed_tool_names):
        return {"google_flights": object()}

    monkeypatch.setattr(flight_service, "load_allowed_tool_map", fake_tool_map)
    bureau = FlightTransportBureau()
    bureau._agent_reasoning = FakeAgent().ainvoke
    payload = {
        "request_id": "flight_wrong_args",
        "approved_draft": {"destination": "Tokyo", "itinerary_draft": {"destination": "Tokyo", "daily_plan": [{"date": "2026-10-01", "activities": []}]}},
        "execution_plan": {"user_request": {"profile": {"origin_city": "Beijing", "origin_airport_code": "PEK", "destination_airport_code": "HND", "start_date": "2026-10-01", "end_date": "2026-10-01", "adults": 2, "currency": "USD"}}},
    }

    result = await bureau.run(payload)

    assert result["bureau"] == "FLIGHT_TRANSPORT"
    assert result["status"] == "fallback"
    assert result["data_source"] == "fallback_estimate"
    assert any("departure_id must match PEK" in note for note in result["transport_notes"])
    assert result["liubu_quality"]["passed"] is False
    assert result["liubu_evidence"][0]["status"] == "blocked"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_bureaus.py::test_flight_transport_blocks_agent_tool_call_with_wrong_trip_facts -q
```

Expected: FAIL because `FlightTransportBureau` has no constrained evidence fields and no `_agent_reasoning` hook.

- [ ] **Step 3: Replace Flight graph nodes with constrained nodes**

Modify `provinces/liubu/flight_transport/service.py`:

```python
from provinces.liubu.constrained.gates import gate_flight_transport_result, result_passed_gate
from provinces.liubu.constrained.state import LiubuToolEvidence, LiubuWorkerState, normalize_worker_input
from provinces.liubu.constrained.tools import execute_constrained_tool_call, load_allowed_tool_map

FLIGHT_ALLOWED_TOOLS = {"google_flights", "google_maps_directions", "geocode", "maps_directions", "amap_direction_driving"}
MAX_TOOL_STEPS = 3
```

Replace the graph build method with:

```python
    def _build_graph(self):
        graph = StateGraph(LiubuWorkerState)
        graph.add_node("prepare_context", self.prepare_context)
        graph.add_node("agent_reasoning", self.agent_reasoning)
        graph.add_node("tool_execution", self.tool_execution)
        graph.add_node("structured_result", self.structured_result)
        graph.add_node("quality_gate", self.quality_gate)
        graph.set_entry_point("prepare_context")
        graph.add_edge("prepare_context", "agent_reasoning")
        graph.add_conditional_edges("agent_reasoning", self._route_after_reasoning, {"tool_execution": "tool_execution", "structured_result": "structured_result"})
        graph.add_conditional_edges("tool_execution", self._route_after_tools, {"agent_reasoning": "agent_reasoning", "structured_result": "structured_result"})
        graph.add_edge("structured_result", "quality_gate")
        graph.add_edge("quality_gate", END)
        return graph.compile()
```

Add these methods:

```python
    async def prepare_context(self, state: LiubuWorkerState) -> dict[str, Any]:
        worker_input = normalize_worker_input(state["payload"], "FLIGHT_TRANSPORT")
        return {"worker_input": worker_input, "tool_evidence": [], "validation_findings": [], "warnings": []}

    async def agent_reasoning(self, state: LiubuWorkerState) -> dict[str, Any]:
        if hasattr(self, "_agent_reasoning"):
            return await self._agent_reasoning(state)
        worker_input = state["worker_input"]
        constraints = worker_input.constraints
        if constraints.get("origin_airport_code") and constraints.get("destination_airport_code"):
            return {
                "tool_requests": [
                    {
                        "tool": "google_flights",
                        "args": {
                            "departure_id": constraints["origin_airport_code"],
                            "arrival_id": constraints["destination_airport_code"],
                            "outbound_date": constraints["start_date"],
                            "adults": constraints["adults"],
                            "currency": constraints["currency"],
                        },
                    }
                ]
            }
        return {"tool_requests": []}

    async def tool_execution(self, state: LiubuWorkerState) -> dict[str, Any]:
        worker_input = state["worker_input"]
        existing = [LiubuToolEvidence.model_validate(item) for item in state.get("tool_evidence", [])]
        tool_map = await load_allowed_tool_map(["amap", "serpapi"], FLIGHT_ALLOWED_TOOLS)
        for request in state.get("tool_requests", [])[:MAX_TOOL_STEPS]:
            evidence = await execute_constrained_tool_call(
                worker_input=worker_input,
                tool_map=tool_map,
                allowed_tool_names=FLIGHT_ALLOWED_TOOLS,
                tool_name=str(request.get("tool") or ""),
                args=dict(request.get("args") or {}),
            )
            existing.append(evidence)
        return {"tool_evidence": [item.model_dump(mode="json") for item in existing], "tool_requests": []}

    async def structured_result(self, state: LiubuWorkerState) -> dict[str, Any]:
        evidence = [LiubuToolEvidence.model_validate(item) for item in state.get("tool_evidence", [])]
        live_notes = [str(item.result) for item in evidence if item.status == "ok"]
        if live_notes:
            result = await self.synthesize_transport(
                {
                    "origin_city": state["worker_input"].profile.get("origin_city") or "Unknown origin",
                    "destination": state["worker_input"].destination,
                    "profile": state["worker_input"].profile,
                    "daily_plan": state["worker_input"].daily_plan,
                    "research_notes": "\n".join(live_notes),
                }
            )
            payload = result["result"]
            payload["data_source"] = "live"
            payload["liubu_evidence"] = [item.model_dump(mode="json") for item in evidence]
            return {"result": payload}
        result = await self.synthesize_transport(
            {
                "origin_city": state["worker_input"].profile.get("origin_city") or "Unknown origin",
                "destination": state["worker_input"].destination,
                "profile": state["worker_input"].profile,
                "daily_plan": state["worker_input"].daily_plan,
                "research_notes": "; ".join(item.error or item.status for item in evidence) or "No successful live flight evidence.",
            }
        )
        payload = result["result"]
        payload["liubu_evidence"] = [item.model_dump(mode="json") for item in evidence]
        return {"result": payload}

    async def quality_gate(self, state: LiubuWorkerState) -> dict[str, Any]:
        evidence = [LiubuToolEvidence.model_validate(item) for item in state.get("tool_evidence", [])]
        findings = gate_flight_transport_result(state["worker_input"], state["result"], evidence)
        passed = result_passed_gate(findings)
        result = dict(state["result"])
        result["liubu_quality"] = {"passed": passed, "findings": [item.model_dump(mode="json") for item in findings]}
        if not passed:
            result["status"] = "fallback"
            result["data_source"] = "fallback_estimate"
            result.setdefault("transport_notes", [])
            result["transport_notes"].extend(item.message for item in findings)
        return {"result": result, "validation_findings": [item.model_dump(mode="json") for item in findings]}

    def _route_after_reasoning(self, state: LiubuWorkerState) -> str:
        return "tool_execution" if state.get("tool_requests") else "structured_result"

    def _route_after_tools(self, state: LiubuWorkerState) -> str:
        return "structured_result"
```

- [ ] **Step 4: Run targeted Flight test**

Run:

```bash
python -m pytest tests/test_bureaus.py::test_flight_transport_blocks_agent_tool_call_with_wrong_trip_facts -q
```

Expected: PASS.

- [ ] **Step 5: Run Flight and constrained tests**

Run:

```bash
python -m pytest tests/test_bureaus.py::test_bureau_structured_failures_log_and_return_labeled_fallback tests/test_bureaus.py::test_bureau_structured_synthesis_timeout_returns_fallback tests/test_liubu_constrained_state.py tests/test_liubu_constrained_tools.py tests/test_liubu_constrained_gates.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Flight migration**

Run:

```bash
git add provinces/liubu/flight_transport/service.py tests/test_bureaus.py
git commit -m "feat: migrate flight bureau to constrained graph"
```

### Task 5: Migrate Accommodation Bureau

**Files:**
- Modify: `provinces/liubu/accommodation/service.py`
- Modify: `tests/test_bureaus.py`

- [ ] **Step 1: Write failing Accommodation constrained test**

Append to `tests/test_bureaus.py`:

```python
@pytest.mark.asyncio
async def test_accommodation_blocks_agent_hotel_search_with_past_dates(monkeypatch):
    class FakeAgent:
        async def ainvoke(self, state):
            return {
                "tool_requests": [
                    {
                        "tool": "google_hotels",
                        "args": {
                            "q": "Tokyo hotels",
                            "check_in_date": "2023-10-01",
                            "check_out_date": "2023-10-02",
                            "adults": 1,
                            "currency": "JPY",
                        },
                    }
                ]
            }

    monkeypatch.setattr(accommodation_service, "build_qwen_chat", lambda: None)
    async def fake_tool_map(server_names, allowed_tool_names):
        return {"google_hotels": object()}

    monkeypatch.setattr(accommodation_service, "load_allowed_tool_map", fake_tool_map)
    bureau = AccommodationBureau()
    bureau._agent_reasoning = FakeAgent().ainvoke
    payload = {
        "request_id": "hotel_wrong_args",
        "approved_draft": {"destination": "Tokyo", "itinerary_draft": {"destination": "Tokyo", "daily_plan": [{"date": "2026-10-01", "activities": []}, {"date": "2026-10-02", "activities": []}]}},
        "execution_plan": {"user_request": {"profile": {"start_date": "2026-10-01", "end_date": "2026-10-02", "adults": 2, "currency": "USD"}}},
    }

    result = await bureau.run(payload)

    assert result["bureau"] == "ACCOMMODATION"
    assert result["status"] == "fallback"
    assert result["data_source"] == "fallback_estimate"
    assert any("check_in_date must match 2026-10-01" in warning for warning in result["warnings"])
    assert result["liubu_quality"]["passed"] is False
    assert result["liubu_evidence"][0]["status"] == "blocked"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_bureaus.py::test_accommodation_blocks_agent_hotel_search_with_past_dates -q
```

Expected: FAIL because `AccommodationBureau` has no constrained evidence fields and no `_agent_reasoning` hook.

- [ ] **Step 3: Replace Accommodation graph nodes with constrained nodes**

Modify `provinces/liubu/accommodation/service.py` using the same graph shape as Flight and these bureau constants:

```python
from provinces.liubu.constrained.gates import gate_accommodation_result, result_passed_gate
from provinces.liubu.constrained.state import LiubuToolEvidence, LiubuWorkerState, normalize_worker_input
from provinces.liubu.constrained.tools import execute_constrained_tool_call, load_allowed_tool_map

ACCOMMODATION_ALLOWED_TOOLS = {"google_hotels", "google_maps", "search_poi", "nearby_search", "amap_place_search"}
MAX_TOOL_STEPS = 3
```

Use these Accommodation-specific methods:

```python
    async def prepare_context(self, state: LiubuWorkerState) -> dict[str, Any]:
        worker_input = normalize_worker_input(state["payload"], "ACCOMMODATION")
        return {"worker_input": worker_input, "tool_evidence": [], "validation_findings": [], "warnings": []}

    async def agent_reasoning(self, state: LiubuWorkerState) -> dict[str, Any]:
        if hasattr(self, "_agent_reasoning"):
            return await self._agent_reasoning(state)
        worker_input = state["worker_input"]
        constraints = worker_input.constraints
        return {
            "tool_requests": [
                {
                    "tool": "google_hotels",
                    "args": {
                        "q": f"{worker_input.destination} hotels",
                        "check_in_date": constraints["start_date"],
                        "check_out_date": constraints["end_date"],
                        "adults": constraints["adults"],
                        "currency": constraints["currency"],
                    },
                }
            ]
        }

    async def tool_execution(self, state: LiubuWorkerState) -> dict[str, Any]:
        worker_input = state["worker_input"]
        existing = [LiubuToolEvidence.model_validate(item) for item in state.get("tool_evidence", [])]
        tool_map = await load_allowed_tool_map(["amap", "serpapi"], ACCOMMODATION_ALLOWED_TOOLS)
        for request in state.get("tool_requests", [])[:MAX_TOOL_STEPS]:
            evidence = await execute_constrained_tool_call(
                worker_input=worker_input,
                tool_map=tool_map,
                allowed_tool_names=ACCOMMODATION_ALLOWED_TOOLS,
                tool_name=str(request.get("tool") or ""),
                args=dict(request.get("args") or {}),
            )
            existing.append(evidence)
        return {"tool_evidence": [item.model_dump(mode="json") for item in existing], "tool_requests": []}

    async def structured_result(self, state: LiubuWorkerState) -> dict[str, Any]:
        evidence = [LiubuToolEvidence.model_validate(item) for item in state.get("tool_evidence", [])]
        live_notes = [str(item.result) for item in evidence if item.status == "ok"]
        result = await self.synthesize_accommodation(
            {
                "destination": state["worker_input"].destination,
                "profile": state["worker_input"].profile,
                "daily_plan": state["worker_input"].daily_plan,
                "research_notes": "\n".join(live_notes) if live_notes else "; ".join(item.error or item.status for item in evidence) or "No successful live hotel evidence.",
            }
        )
        payload = result["result"]
        if live_notes:
            payload["data_source"] = "live"
        payload["liubu_evidence"] = [item.model_dump(mode="json") for item in evidence]
        return {"result": payload}

    async def quality_gate(self, state: LiubuWorkerState) -> dict[str, Any]:
        evidence = [LiubuToolEvidence.model_validate(item) for item in state.get("tool_evidence", [])]
        findings = gate_accommodation_result(state["worker_input"], state["result"], evidence)
        passed = result_passed_gate(findings)
        result = dict(state["result"])
        result["liubu_quality"] = {"passed": passed, "findings": [item.model_dump(mode="json") for item in findings]}
        if not passed:
            result["status"] = "fallback"
            result["data_source"] = "fallback_estimate"
            result.setdefault("warnings", [])
            result["warnings"].extend(item.message for item in findings)
        return {"result": result, "validation_findings": [item.model_dump(mode="json") for item in findings]}
```

- [ ] **Step 4: Run targeted Accommodation test**

Run:

```bash
python -m pytest tests/test_bureaus.py::test_accommodation_blocks_agent_hotel_search_with_past_dates -q
```

Expected: PASS.

- [ ] **Step 5: Run all bureau tests**

Run:

```bash
python -m pytest tests/test_bureaus.py tests/test_liubu_constrained_state.py tests/test_liubu_constrained_tools.py tests/test_liubu_constrained_gates.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Accommodation migration**

Run:

```bash
git add provinces/liubu/accommodation/service.py tests/test_bureaus.py
git commit -m "feat: migrate accommodation bureau to constrained graph"
```

### Task 6: Shangshu Quality Metadata Aggregation

**Files:**
- Modify: `provinces/shangshu_orchestrator/orchestrator.py`
- Modify: `workflow.py`
- Modify: `tests/test_orchestrator.py`

- [ ] **Step 1: Write failing metadata aggregation test**

Append to `tests/test_orchestrator.py`:

```python
def test_register_execution_result_records_liubu_quality_metadata():
    orchestrator = ShangshuOrchestrator()
    context = orchestrator.bootstrap("quality_meta", {"request_id": "quality_meta"})

    orchestrator.register_execution_result(
        context,
        AgentRole.FLIGHT_TRANSPORT,
        {
            "bureau": "FLIGHT_TRANSPORT",
            "status": "fallback",
            "data_source": "fallback_estimate",
            "liubu_quality": {
                "passed": False,
                "findings": [{"severity": "error", "code": "wrong_departure_date", "message": "date mismatch"}],
            },
        },
    )

    assert context.execution_results["FLIGHT_TRANSPORT"]["liubu_quality"]["passed"] is False
    assert context.quality_gate_results["FLIGHT_TRANSPORT"]["passed"] is False
    assert context.progress_events[-1]["stage"] == "execution_quality_gate"
    assert "wrong_departure_date" in context.progress_events[-1]["message"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_orchestrator.py::test_register_execution_result_records_liubu_quality_metadata -q
```

Expected: FAIL because `ShangshuWorkflowContext` has no `quality_gate_results`.

- [ ] **Step 3: Add quality metadata to context and registration**

Modify `ShangshuWorkflowContext` in `provinces/shangshu_orchestrator/orchestrator.py`:

```python
    quality_gate_results: dict[str, dict[str, Any]] = field(default_factory=dict)
```

Modify `register_execution_result`:

```python
    def register_execution_result(self, context: ShangshuWorkflowContext, bureau: AgentRole | str, result_payload: dict[str, Any]) -> None:
        role = AgentRole(bureau)
        enforce_permission(role, ActionType.RETURN_EXECUTION_RESULT, AgentRole.SHANGSHU)
        context.execution_results[role.value] = result_payload
        quality = result_payload.get("liubu_quality")
        if isinstance(quality, dict):
            context.quality_gate_results[role.value] = quality
            finding_codes = [str(item.get("code")) for item in quality.get("findings") or [] if isinstance(item, dict) and item.get("code")]
            self._record_progress(
                context,
                stage="execution_quality_gate",
                message=f"{role.value} quality passed={bool(quality.get('passed'))}; findings={','.join(finding_codes)}",
                actor=role,
            )
        self._record_progress(context, stage="execution_result", message=f"{role.value} execution result received.", actor=role)
```

Modify `_serialize_context` in `workflow.py`:

```python
            "quality_gate_results": context.quality_gate_results,
```

Modify `_deserialize_context` in `workflow.py`:

```python
            quality_gate_results=dict(payload.get("quality_gate_results") or {}),
```

- [ ] **Step 4: Run metadata test**

Run:

```bash
python -m pytest tests/test_orchestrator.py::test_register_execution_result_records_liubu_quality_metadata -q
```

Expected: PASS.

- [ ] **Step 5: Run workflow fanout tests**

Run:

```bash
python -m pytest tests/test_orchestrator.py tests/test_workflow.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit aggregation slice**

Run:

```bash
git add provinces/shangshu_orchestrator/orchestrator.py workflow.py tests/test_orchestrator.py
git commit -m "feat: record liubu quality metadata"
```

### Task 7: Live E2E Evidence Assertions

**Files:**
- Modify: `tests/test_live_e2e.py`

- [ ] **Step 1: Add live E2E assertions for migrated bureaus**

Modify the `DONE` branch in `tests/test_live_e2e.py`:

```python
        for bureau_name in ("accommodation", "flight_transport"):
            bureau_payload = data[bureau_name]
            assert bureau_payload["data_source"] in {"live", "structured_llm", "fallback_estimate"}
            assert "liubu_quality" in bureau_payload
            assert "liubu_evidence" in bureau_payload
            if bureau_payload["status"] == "ok":
                assert bureau_payload["liubu_quality"]["passed"] is True
            else:
                assert bureau_payload["warnings"] or bureau_payload.get("transport_notes")
```

- [ ] **Step 2: Run live marker-safe test collection**

Run:

```bash
python -m pytest tests/test_live_e2e.py -q
```

Expected without live env: SKIPPED. Expected with `RUN_LIVE_TESTS=1` and credentials: PASS or a product-level terminal status assertion failure that includes response text.

- [ ] **Step 3: Commit live assertion slice**

Run:

```bash
git add tests/test_live_e2e.py
git commit -m "test: assert liubu evidence in live e2e"
```

### Task 8: Full Verification

**Files:**
- No source edits.

- [ ] **Step 1: Run migrated constrained unit tests**

Run:

```bash
python -m pytest tests/test_liubu_constrained_state.py tests/test_liubu_constrained_tools.py tests/test_liubu_constrained_gates.py tests/test_bureaus.py tests/test_orchestrator.py tests/test_workflow.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full offline suite**

Run:

```bash
python -m pytest -q
```

Expected: PASS with live tests skipped unless `RUN_LIVE_TESTS=1`.

- [ ] **Step 3: Run live E2E suite when credentials are present**

Run:

```bash
RUN_LIVE_TESTS=1 python -m pytest -q -m live
```

Expected with configured Qwen and MCP credentials: PASS. If a provider fails, the failure must be a structured terminal response or a test assertion with provider status, not a raw secret-bearing traceback.

- [ ] **Step 4: Commit verification evidence if docs are updated**

If verification evidence is recorded in `specs/001-resolve-report-issues/verification-evidence.md`, run:

```bash
git add specs/001-resolve-report-issues/verification-evidence.md
git commit -m "docs: record liubu constrained agent verification"
```

## Acceptance Mapping

- Graph-visible tool evidence: Tasks 2, 4, 5, and 7.
- Tool argument validation before external calls: Task 2, then Flight and Accommodation usage in Tasks 4 and 5.
- Bad hotel dates blocked before SerpApi: Tasks 2, 3, and 5.
- Fallback warnings explain timeout, provider error, validation failure, or missing credentials: Tasks 2, 3, 4, and 5.
- Existing public API fields remain compatible: Tasks 4, 5, 6, and full suite in Task 8.
- Live E2E reaches terminal state with labels: Task 7 and Task 8.

## Execution Notes

- Preserve `FlightTransportBureau.run(payload)` and `AccommodationBureau.run(payload)` signatures.
- Keep `WeatherBureau`, `BudgetBureau`, and `CalendarBureau` behavior unchanged in this migration.
- Do not expose `DONE_WITH_WARNINGS` as a public API status in this migration.
- Keep fallback outputs valid against existing Pydantic result models.
- Do not remove `run_react_mcp_task` because Zhongshu, Menxia, Weather, Budget, and Calendar still use existing runtime helpers.
