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
        compact_result = sanitize_result(_compact_mcp_tool_result(result))
        if isinstance(compact_result, dict):
            compact_result.setdefault("search_parameters", sanitized_args)
            if not compact_result["search_parameters"]:
                compact_result["search_parameters"] = sanitized_args
        return LiubuToolEvidence(tool_name=tool_name, args=sanitized_args, status="ok", result=compact_result, data_source="live")
    except asyncio.TimeoutError:
        return LiubuToolEvidence(tool_name=tool_name, args=sanitized_args, status="timeout", error=f"Tool {tool_name} timed out after {timeout_seconds:.2f}s.")
    except Exception as exc:
        return LiubuToolEvidence(tool_name=tool_name, args=sanitized_args, status="error", error=str(exc))


def validate_tool_args(worker_input: LiubuWorkerInput, tool_name: str, args: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    constraints = worker_input.constraints
    start_date = str(constraints.get("start_date") or "")
    end_date = str(constraints.get("end_date") or start_date)
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
        _require_equal(errors, args, ("check_out_date", "check_out", "checkout_date"), end_date)
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
