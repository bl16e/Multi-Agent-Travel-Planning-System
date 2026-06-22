from __future__ import annotations

import asyncio
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent

from utils.llm_factory import build_qwen_chat
from utils.mcp_client import load_mcp_tools


FALLBACK_MESSAGE = "MCP or LLM unavailable; falling back to heuristic synthesis."
DEFAULT_TOOL_LOAD_TIMEOUT_SECONDS = 6.0
DEFAULT_AGENT_INVOKE_TIMEOUT_SECONDS = 18.0
DEFAULT_STRUCTURED_SYNTHESIS_TIMEOUT_SECONDS = 20.0


def load_soul_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


async def run_react_mcp_task(
    *,
    soul_path: str | Path,
    server_names: list[str],
    user_task: str,
    tool_load_timeout_seconds: float = DEFAULT_TOOL_LOAD_TIMEOUT_SECONDS,
    invoke_timeout_seconds: float = DEFAULT_AGENT_INVOKE_TIMEOUT_SECONDS,
) -> str:
    llm = build_qwen_chat()
    if llm is None:
        return FALLBACK_MESSAGE
    try:
        tools = await asyncio.wait_for(load_mcp_tools(server_names), timeout=tool_load_timeout_seconds)
    except asyncio.TimeoutError:
        return f"{FALLBACK_MESSAGE} MCP tool loading timed out after {tool_load_timeout_seconds:.0f}s."
    except Exception as exc:
        return f"{FALLBACK_MESSAGE} MCP tool loading failed: {exc}"
    prompt = load_soul_prompt(soul_path)

    if not tools:
        return FALLBACK_MESSAGE

    agent = create_react_agent(model=llm, tools=tools, prompt=prompt)
    try:
        result = await asyncio.wait_for(agent.ainvoke({"messages": [("user", user_task)]}), timeout=invoke_timeout_seconds)
    except asyncio.TimeoutError:
        return f"{FALLBACK_MESSAGE} Agent execution timed out after {invoke_timeout_seconds:.0f}s."
    except Exception as exc:
        return f"{FALLBACK_MESSAGE} Agent tool execution failed: {exc}"
    messages = result.get("messages", [])
    if not messages:
        return "No MCP research output returned."
    return str(messages[-1].content)


async def run_structured_synthesis(
    *,
    soul_path: str | Path,
    output_model: Any,
    user_prompt: str,
    variables: dict[str, Any],
    timeout_seconds: float = DEFAULT_STRUCTURED_SYNTHESIS_TIMEOUT_SECONDS,
) -> Any:
    llm = build_qwen_chat()
    if llm is None:
        return _offline_structured_output(output_model, variables)
    try:
        structured = llm.with_structured_output(output_model)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", load_soul_prompt(soul_path)),
                ("user", user_prompt),
            ]
        )
        return await asyncio.wait_for((prompt | structured).ainvoke(variables), timeout=timeout_seconds)
    except Exception as exc:
        try:
            return _offline_structured_output(output_model, {**variables, "fallback_reason": str(exc)})
        except RuntimeError as fallback_exc:
            raise RuntimeError(f"Structured synthesis failed and no offline fallback exists: {fallback_exc}") from exc


def soul_path_for(file_path: str | Path) -> Path:
    return Path(file_path).with_name("SOUL.md")


def _offline_structured_output(output_model: Any, variables: dict[str, Any]) -> Any:
    model_name = getattr(output_model, "__name__", "")
    if model_name == "ItineraryDraftModel":
        return _offline_itinerary_draft(output_model, variables)
    raise RuntimeError("LLM unavailable for structured synthesis.")


def _offline_itinerary_draft(output_model: Any, variables: dict[str, Any]) -> Any:
    destination = str(variables.get("destination") or "Destination")
    start = _parse_date(variables.get("start_date")) or date.today()
    end = _parse_date(variables.get("end_date")) or start
    day_count = max((end - start).days + 1, 1)
    interests = [item.strip() for item in str(variables.get("interests") or "sightseeing").split(",") if item.strip()]
    if not interests:
        interests = ["sightseeing"]

    daily_plan = []
    for index in range(day_count):
        current = start + timedelta(days=index)
        interest = interests[index % len(interests)]
        primary_title = f"{destination} {interest} route with named local checkpoints"
        secondary_title = f"{destination} {interest} venue confirmation block"
        daily_plan.append(
            {
                "day_index": index + 1,
                "date": current,
                "city": destination,
                "theme": f"{destination} {interest}",
                "summary": f"Offline estimate for {destination} focused on {interest}; verify live opening hours before booking.",
                "activities": [
                    {
                        "start_time": "09:00",
                        "end_time": "11:30",
                        "title": primary_title,
                        "location_name": f"{destination} main visitor district",
                        "description": f"Input-derived offline plan segment for {interest}; replace with live venue details before booking.",
                        "estimated_cost": 0,
                        "status": "pending",
                    },
                    {
                        "start_time": "14:00",
                        "end_time": "16:30",
                        "title": secondary_title,
                        "location_name": f"{destination} {interest} area",
                        "description": "Offline estimate derived from traveler interests; confirm named venues, opening hours, and ticket availability.",
                        "estimated_cost": 40,
                        "status": "pending",
                    },
                ],
                "accommodation_note": "Choose a central base near primary transit.",
            }
        )

    return output_model.model_validate(
        {
            "destination": destination,
            "overview": "Offline fallback itinerary generated because live LLM/MCP synthesis was unavailable.",
            "trip_style": "balanced",
            "daily_plan": daily_plan,
            "planning_notes": [
                "Did not use real-time data; verify hours, prices, and booking availability.",
                "data_source=fallback_estimate; status=fallback; offline synthesis did not use live travel data.",
                str(variables.get("research_context") or ""),
            ],
            "pending_confirmations": [
                "Confirm final accommodation booking before issuing the calendar bundle.",
                "Confirm one primary paid attraction per day to reduce queue risk.",
            ],
            "risk_flags": [
                "Opening hours and ticket availability may change and must be reviewed downstream.",
                "Weather suitability is not yet validated and may alter outdoor blocks.",
            ],
        }
    )


def _parse_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None
