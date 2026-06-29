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
    itinerary_draft: Any
    destination: str | None = None
    overview: str | None = None
    trip_style: str | None = None
    daily_plan: list[Any] | None = None
    planning_notes: Any = Field(default_factory=list)
    pending_confirmations: Any = Field(default_factory=list)
    risk_flags: Any = Field(default_factory=list)


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
    if isinstance(result.get("pois"), list):
        compact["pois"] = [_compact_amap_poi_result(item) for item in result["pois"][:8] if isinstance(item, dict)]
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


def _compact_amap_poi_result(item: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "name",
        "type",
        "address",
        "location",
        "tel",
        "website",
        "pname",
        "cityname",
        "adname",
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

    candidate = _coerce_itinerary_candidate(candidate, data, variables)
    if not candidate.get("destination"):
        candidate["destination"] = str(variables.get("destination") or "Destination")
    if not candidate.get("overview"):
        candidate["overview"] = data.get("overview") or "Structured itinerary generated from live research."
    if not candidate.get("trip_style"):
        candidate["trip_style"] = data.get("trip_style") or "structured"
    candidate["planning_notes"] = _coerce_string_list(candidate.get("planning_notes") or data.get("planning_notes"))
    candidate["pending_confirmations"] = _coerce_string_list(candidate.get("pending_confirmations") or data.get("pending_confirmations"))
    candidate["risk_flags"] = _coerce_string_list(candidate.get("risk_flags") or data.get("risk_flags"))
    return output_model.model_validate(candidate)


def _coerce_itinerary_candidate(candidate: dict[str, Any], data: dict[str, Any], variables: dict[str, Any]) -> dict[str, Any]:
    coerced = dict(candidate)
    destination = str(coerced.get("destination") or data.get("destination") or variables.get("destination") or "Destination")
    daily_plan = coerced.get("daily_plan") or data.get("daily_plan") or []
    if isinstance(daily_plan, dict):
        daily_plan = [daily_plan]
    if isinstance(daily_plan, list):
        coerced["daily_plan"] = [
            _coerce_day_plan(day, index=index, destination=destination, variables=variables)
            for index, day in enumerate(daily_plan, start=1)
            if isinstance(day, dict)
        ]
    return coerced


def _coerce_day_plan(day: dict[str, Any], *, index: int, destination: str, variables: dict[str, Any]) -> dict[str, Any]:
    date_value = day.get("date") or variables.get("start_date")
    interest = _interest_for_index(variables, index)
    activities = _coerce_activities(day.get("activities"), day, destination)
    summary = day.get("summary")
    if not summary:
        summary = f"Structured plan for {destination} focused on {interest}."
    return {
        **day,
        "day_index": day.get("day_index") or index,
        "date": date_value,
        "city": day.get("city") or destination,
        "theme": day.get("theme") or f"{destination} {interest}",
        "summary": summary,
        "activities": activities,
    }


def _coerce_activities(value: Any, day: dict[str, Any], destination: str) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, str) and value.strip():
        title = _activity_title_from_text(value)
        return [
            {
                "start_time": str(day.get("start_time") or "09:00"),
                "end_time": str(day.get("end_time") or "11:00"),
                "title": title,
                "location_name": str(day.get("location_name") or destination),
                "description": value.strip(),
                "booking_link": day.get("booking_link"),
                "transport": _transport_from_text(day.get("transport")),
            }
        ]
    return []


def _activity_title_from_text(value: str) -> str:
    text = value.strip()
    for separator in (":", "\uff1a", "\u3002", ".", ";", "\uff1b"):
        if separator in text:
            before, after = text.split(separator, 1)
            text = after or before
            break
    return text.strip()[:80] or "Structured itinerary activity"


def _transport_from_text(value: Any) -> dict[str, Any] | None:
    if not value:
        return None
    return {
        "from_location": "previous activity or hotel",
        "to_location": "next scheduled activity",
        "mode": "transit",
        "duration_text": str(value),
        "status": "pending",
    }


def _coerce_string_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item)]
    return [str(value)]


def _interest_for_index(variables: dict[str, Any], index: int) -> str:
    interests = [item.strip() for item in str(variables.get("interests") or "sightseeing").split(",") if item.strip()]
    if not interests:
        return "sightseeing"
    return interests[(index - 1) % len(interests)]


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
            type_text = _readable_place_type_text(place_type)
            detail_parts = [f"{title}，地址：{address}"]
            if type_text:
                detail_parts.append(f"类型：{type_text}")
            if rating is not None:
                detail_parts.append(f"评分：{rating}")
            activities.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "title": title,
                    "location_name": address,
                    "description": "；".join(detail_parts) + "。",
                    "map_link": place.get("link") or f"https://www.google.com/maps/search/?api=1&query={title.replace(' ', '+')}",
                    "booking_link": place.get("website"),
                    "status": "confirmed",
                    "transport": {
                        "from_location": "previous activity or hotel",
                        "to_location": address,
                        "mode": "transit",
                        "duration_text": "参考地图链接规划现场路线。",
                        "status": "confirmed",
                    },
                }
            )
        daily_plan.append(
            {
                "day_index": index + 1,
                "date": current,
                "city": destination,
                "theme": _confirmed_place_theme(interest),
                "summary": f"基于实时地点检索结果安排 {destination} 行程，并已补充天气、预算、住宿、交通和日历信息。",
                "activities": activities,
                "accommodation_note": "Choose a central base near the selected activity cluster.",
            }
        )

    synthesis_note = str(variables.get("fallback_reason") or "structured_llm_unavailable")
    return output_model.model_validate(
        {
            "destination": destination,
            "overview": "基于实时地点检索结果生成行程草案，并由执行局补全可交付信息。",
            "trip_style": "live_research",
            "daily_plan": daily_plan,
            "planning_notes": [
                f"structured_llm_status=unavailable; detail={synthesis_note}",
                "data_source=live_mcp_research; synthesis=deterministic_from_compact_tool_results.",
                "Liubu execution enriches the live draft with weather, budget, accommodation, transport, and calendar outputs.",
            ],
            "pending_confirmations": [],
            "risk_flags": [
                "Live place search does not guarantee booking availability.",
            ],
        }
    )


def _readable_place_type_text(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    compact = text.replace("|", "").replace(";", "").replace(",", "").replace(" ", "")
    return "" if compact.isdigit() else text


def _confirmed_place_theme(interest: str) -> str:
    label = interest.strip() or "旅行"
    return f"{label}主题实地行程"


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
    confirmation_items = [
        item
        for item in payload
        if isinstance(item, dict) and item.get("phase") == "confirmation" and item.get("status") == "ok"
    ]
    source_items = confirmation_items or payload
    for item in source_items:
        if not isinstance(item, dict) or item.get("status") != "ok":
            continue
        result = item.get("result")
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                result = None
        if not isinstance(result, dict):
            continue
        candidates: list[dict[str, Any]] = []
        local_results = result.get("local_results")
        if isinstance(local_results, list):
            candidates.extend(candidate for candidate in local_results if isinstance(candidate, dict))
        place_result = result.get("place_results")
        if isinstance(place_result, dict):
            candidates.append(place_result)
        pois = result.get("pois")
        if isinstance(pois, list):
            candidates.extend(_normalize_amap_poi(candidate) for candidate in pois if isinstance(candidate, dict))
        for candidate in candidates:
            title = str(candidate.get("title") or "").strip()
            if not title or title in seen:
                continue
            seen.add(title)
            places.append(candidate)
    return places[:9]


def _normalize_amap_poi(candidate: dict[str, Any]) -> dict[str, Any]:
    title = str(candidate.get("title") or candidate.get("name") or "").strip()
    location = candidate.get("location")
    link = None
    if title:
        link = f"https://ditu.amap.com/search?query={title.replace(' ', '+')}"
    return {
        "title": title,
        "type": candidate.get("type") or candidate.get("typecode"),
        "address": candidate.get("address"),
        "gps_coordinates": location,
        "website": candidate.get("website"),
        "link": link,
    }


def _parse_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None



