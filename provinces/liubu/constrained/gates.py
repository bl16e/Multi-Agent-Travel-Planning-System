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
