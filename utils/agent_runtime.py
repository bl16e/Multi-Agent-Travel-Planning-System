from __future__ import annotations

import asyncio
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from utils.llm_factory import build_qwen_chat
from utils.schemas import DayPlanModel, ItineraryDraftModel
from utils.settings import get_settings


FALLBACK_MESSAGE = "MCP or LLM unavailable; falling back to heuristic synthesis."
DEFAULT_STRUCTURED_SYNTHESIS_TIMEOUT_SECONDS = 20.0


class ItineraryDraftStructuredOutput(BaseModel):
    itinerary_draft: ItineraryDraftModel | list[DayPlanModel]
    destination: str | None = None
    overview: str | None = None
    trip_style: str | None = None
    daily_plan: list[Any] | None = None
    planning_notes: list[str] = Field(default_factory=list)
    pending_confirmations: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)


def load_soul_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def escape_prompt_template_text(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def _compact_mcp_tool_result(result: Any) -> Any:
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            return result[:2000]
    if not isinstance(result, dict):
        return result

    compact: dict[str, Any] = {}
    for key in ("search_metadata", "search_parameters"):
        value = result.get(key)
        if isinstance(value, dict):
            compact[key] = {item_key: value.get(item_key) for item_key in ("status", "engine", "location_used") if value.get(item_key)}

    if isinstance(result.get("local_results"), list):
        compact["local_results"] = [_compact_place_result(item) for item in result["local_results"][:8] if isinstance(item, dict)]
    if isinstance(result.get("place_results"), dict):
        compact["place_results"] = _compact_place_result(result["place_results"])
    if isinstance(result.get("directions"), list):
        compact["directions"] = result["directions"][:5]

    return compact or {key: result.get(key) for key in ("title", "address", "type", "rating") if key in result}


def _compact_place_result(item: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "title",
        "type",
        "address",
        "rating",
        "reviews",
        "hours",
        "website",
        "link",
        "gps_coordinates",
    )
    return {key: item.get(key) for key in fields if item.get(key) is not None}



async def run_structured_synthesis(
    *,
    soul_path: str | Path,
    output_model: Any,
    user_prompt: str,
    variables: dict[str, Any],
    timeout_seconds: float | None = None,
) -> Any:
    llm = build_qwen_chat()
    if llm is None:
        return _offline_structured_output(output_model, variables)
    try:
        structured_model = _structured_output_model(output_model)
        structured = llm.with_structured_output(structured_model)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", escape_prompt_template_text(load_soul_prompt(soul_path))),
                ("user", f"{user_prompt}\n\nReturn valid JSON only, matching the requested structured schema."),
            ]
        )
        effective_timeout = timeout_seconds or get_settings().qwen_timeout_seconds
        result = await asyncio.wait_for((prompt | structured).ainvoke(variables), timeout=effective_timeout)
        return _normalize_structured_output(output_model, result, variables)
    except Exception as exc:
        try:
            return _offline_structured_output(output_model, {**variables, "fallback_reason": str(exc)})
        except RuntimeError as fallback_exc:
            raise RuntimeError(f"Structured synthesis failed and no offline fallback exists: {fallback_exc}") from exc


def soul_path_for(file_path: str | Path) -> Path:
    return Path(file_path).with_name("SOUL.md")


def _structured_output_model(output_model: Any) -> Any:
    if getattr(output_model, "__name__", "") == "ItineraryDraftModel":
        return ItineraryDraftStructuredOutput
    return output_model


def _normalize_structured_output(output_model: Any, result: Any, variables: dict[str, Any]) -> Any:
    if isinstance(result, output_model):
        return result
    if getattr(output_model, "__name__", "") != "ItineraryDraftModel":
        return result

    data = result.model_dump(mode="json") if isinstance(result, BaseModel) else dict(result)
    wrapped = data.get("itinerary_draft")
    if isinstance(wrapped, dict):
        candidate = dict(wrapped)
    elif isinstance(wrapped, list):
        candidate = {"daily_plan": wrapped}
    else:
        candidate = data

    if not candidate.get("destination"):
        candidate["destination"] = str(variables.get("destination") or "Destination")
    if not candidate.get("overview"):
        candidate["overview"] = data.get("overview") or "Structured itinerary generated from live research."
    if not candidate.get("trip_style"):
        candidate["trip_style"] = data.get("trip_style") or "structured"
    if not candidate.get("planning_notes"):
        candidate["planning_notes"] = data.get("planning_notes") or []
    if not candidate.get("pending_confirmations"):
        candidate["pending_confirmations"] = data.get("pending_confirmations") or []
    if not candidate.get("risk_flags"):
        candidate["risk_flags"] = data.get("risk_flags") or []
    return output_model.model_validate(candidate)


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
    live_places = _extract_live_places(variables.get("research_context"))
    if live_places:
        return _live_research_itinerary_draft(output_model, variables, destination, start, day_count, interests, live_places)

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


def _live_research_itinerary_draft(
    output_model: Any,
    variables: dict[str, Any],
    destination: str,
    start: date,
    day_count: int,
    interests: list[str],
    places: list[dict[str, Any]],
) -> Any:
    daily_plan = []
    slots = [("09:00", "11:00"), ("13:00", "15:00"), ("16:00", "18:00")]
    place_index = 0
    for index in range(day_count):
        current = start + timedelta(days=index)
        interest = interests[index % len(interests)]
        activities = []
        for slot_index, (start_time, end_time) in enumerate(slots):
            place = places[place_index % len(places)]
            place_index += 1
            title = str(place.get("title") or f"{destination} researched place")
            address = str(place.get("address") or destination)
            place_type = str(place.get("type") or "place")
            rating = place.get("rating")
            rating_text = f" Rating: {rating}." if rating is not None else ""
            activities.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "title": title,
                    "location_name": address,
                    "description": f"Live MCP place result for {place_type}.{rating_text} Verify opening hours and ticket availability before booking.",
                    "map_link": place.get("link") or f"https://www.google.com/maps/search/?api=1&query={title.replace(' ', '+')}",
                    "booking_link": place.get("website"),
                    "status": "pending",
                    "transport": {
                        "from_location": "previous activity or hotel",
                        "to_location": address,
                        "mode": "transit",
                        "duration_text": "Confirm live transit duration before final approval.",
                        "status": "pending",
                    },
                }
            )
        daily_plan.append(
            {
                "day_index": index + 1,
                "date": current,
                "city": destination,
                "theme": f"{destination} {interest}",
                "summary": f"Uses live MCP place results for {destination}; downstream bureaus must verify hours, bookings, weather, and transport.",
                "activities": activities,
                "accommodation_note": "Choose a central base near the selected activity cluster.",
            }
        )

    fallback_reason = str(variables.get("fallback_reason") or "structured_llm_timeout")
    return output_model.model_validate(
        {
            "destination": destination,
            "overview": "Live MCP research was used to build this draft; structured LLM synthesis was unavailable, so the plan remains pending review.",
            "trip_style": "live_research_fallback",
            "daily_plan": daily_plan,
            "planning_notes": [
                f"structured_llm_timeout_or_error={fallback_reason}",
                "data_source=live_mcp_research; synthesis=fallback_from_compact_tool_results.",
                "Review all opening hours, reservation requirements, and transport durations before final approval.",
            ],
            "pending_confirmations": [
                "Confirm official opening hours and ticket availability for each listed place.",
                "Confirm transit duration and routing between daily activities.",
            ],
            "risk_flags": [
                "LLM structured synthesis timed out; itinerary order is deterministic from live place results.",
                "Live place search does not guarantee booking availability.",
            ],
        }
    )


def _extract_live_places(research_context: Any) -> list[dict[str, Any]]:
    if not isinstance(research_context, str) or research_context.startswith(FALLBACK_MESSAGE):
        return []
    try:
        payload = json.loads(research_context)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []

    places: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in payload:
        if not isinstance(item, dict) or item.get("status") != "ok":
            continue
        result = item.get("result")
        if not isinstance(result, dict):
            continue
        candidates: list[dict[str, Any]] = []
        local_results = result.get("local_results")
        if isinstance(local_results, list):
            candidates.extend(candidate for candidate in local_results if isinstance(candidate, dict))
        place_result = result.get("place_results")
        if isinstance(place_result, dict):
            candidates.append(place_result)
        for candidate in candidates:
            title = str(candidate.get("title") or "").strip()
            if not title or title in seen:
                continue
            seen.add(title)
            places.append(candidate)
    return places[:9]


def _parse_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None



