from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from utils.agent_runtime import run_structured_synthesis, soul_path_for
from utils.mcp_client import load_mcp_tools
from utils.mcp_tool_registry import load_agent_tools
from utils.schemas import BureauTaskSpec, ItineraryDraftModel, ZhongshuDraftPacketModel
from utils.settings import get_settings


logger = logging.getLogger(__name__)


class ZhongshuState(TypedDict, total=False):
    request_id: str
    user_request: dict[str, Any]
    governance: dict[str, Any]
    normalized_request: dict[str, Any]
    research_notes: str
    draft: dict[str, Any]
    previous_draft: dict[str, Any]
    review_feedback: dict[str, Any]
    bureau_tasks: list[dict[str, Any]]
    required_bureaus: list[str]
    finalized_packet: dict[str, Any]


class ZhongshuItineraryAgent:
    def __init__(self) -> None:
        self.soul_path = soul_path_for(__file__)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ZhongshuState)
        graph.add_node("ingest_request", self.ingest_request)
        graph.add_node("draft_itinerary", self.draft_itinerary)
        graph.add_node("decompose_tasks", self.decompose_tasks)
        graph.add_node("finalize_draft", self.finalize_draft)
        
        graph.set_entry_point("ingest_request")
        graph.add_edge("ingest_request", "draft_itinerary")
        graph.add_edge("draft_itinerary", "decompose_tasks")
        graph.add_edge("decompose_tasks", "finalize_draft")
        graph.add_edge("finalize_draft", END)
        return graph.compile()

    async def ingest_request(self, state: ZhongshuState) -> dict[str, Any]:
        user_request = state["user_request"]
        review_feedback = state.get("review_feedback") or {}
        previous_draft = state.get("draft") or {}
        profile = user_request.get("profile", {})
        destination_preferences = profile.get("destination_preferences") or []
        destination = next((str(item).strip() for item in destination_preferences if str(item).strip()), "")
        if not destination:
            raise ValueError("destination preference is required")
        rejection_reasons = [
            *[str(item) for item in review_feedback.get("blocking_issues", []) if item],
            *[str(item) for item in review_feedback.get("review_notes", []) if item],
        ]
        revision_requests = [str(item) for item in review_feedback.get("revision_requests", []) if item]
        normalized = {
            "destination": destination,
            "origin_city": profile.get("origin_city", ""),
            "origin_airport_code": profile.get("origin_airport_code", ""),
            "destination_airport_code": profile.get("destination_airport_code", ""),
            "start_date": profile.get("start_date"),
            "end_date": profile.get("end_date"),
            "budget_level": profile.get("budget_level", "mid_range"),
            "total_budget": profile.get("total_budget"),
            "currency": profile.get("currency", "USD"),
            "adults": profile.get("adults", 1),
            "children": profile.get("children", 0),
            "interests": profile.get("interests", []),
            "constraints": profile.get("constraints", []),
            "pace": profile.get("pace", "balanced"),
            "user_message": user_request.get("user_message", ""),
            "revision_round": int((state.get("governance") or {}).get("rejection_count") or 0),
            "rejection_reasons": rejection_reasons,
            "revision_requests": revision_requests,
        }
        previous_notes = (((previous_draft or {}).get("itinerary_draft") or {}).get("planning_notes") or [])
        context_notes = [
            f"Destination: {destination}",
            f"Dates: {profile.get('start_date')} to {profile.get('end_date')}",
            f"Origin: {profile.get('origin_city') or 'unknown'} ({profile.get('origin_airport_code') or 'n/a'})",
            f"Budget: {profile.get('total_budget')} {profile.get('currency', 'USD')}",
            f"Travelers: {profile.get('adults', 1)} adults, {profile.get('children', 0)} children",
            f"Interests: {', '.join(profile.get('interests', [])) or 'general sightseeing'}",
            f"Constraints: {', '.join(profile.get('constraints', [])) or 'none'}",
            *[f"Blocking issue: {item}" for item in rejection_reasons],
            *[f"Revision request: {item}" for item in revision_requests],
        ]
        merged_notes = [str(item) for item in [*previous_notes, *context_notes] if item and str(item) != "None"]
        return {
            "normalized_request": normalized,
            "review_feedback": review_feedback,
            "previous_draft": previous_draft,
            "research_notes": "\n".join(merged_notes),
        }

    async def draft_itinerary(self, state: ZhongshuState) -> dict[str, Any]:
        normalized = state["normalized_request"]

        try:
            research_context = await self._collect_research_context(normalized, request_id=str(state.get("request_id") or ""))

            draft = await run_structured_synthesis(
                soul_path=self.soul_path,
                output_model=ItineraryDraftModel,
                user_prompt=(
                    "请根据用户需求生成详细的旅行行程草案（使用中文）。\n"
                    "目的地: {destination}\n"
                    "日期: {start_date} 至 {end_date}\n"
                    "出发地: {origin_city} ({origin_airport_code})\n"
                    "预算: {total_budget} {currency}\n"
                    "旅行者: {adults} 成人, {children} 儿童\n"
                    "兴趣: {interests}\n"
                    "约束: {constraints}\n"
                    "之前的拒绝原因: {rejection_reasons}\n"
                    "修订要求: {revision_requests}\n"
                    "研究背景: {research_context}\n\n"
                    "生成具体景点名称、真实预订链接、交通细节和天气应急方案。"
                ),
                variables={
                    "destination": normalized["destination"],
                    "start_date": str(normalized.get("start_date")),
                    "end_date": str(normalized.get("end_date")),
                    "origin_city": normalized.get("origin_city", ""),
                    "origin_airport_code": normalized.get("origin_airport_code", ""),
                    "total_budget": normalized.get("total_budget"),
                    "currency": normalized.get("currency", "USD"),
                    "adults": normalized.get("adults", 1),
                    "children": normalized.get("children", 0),
                    "interests": ", ".join(normalized.get("interests", [])),
                    "constraints": ", ".join(normalized.get("constraints", [])),
                    "rejection_reasons": "\n".join(normalized.get("rejection_reasons", [])),
                    "revision_requests": "\n".join(normalized.get("revision_requests", [])),
                    "research_context": research_context,
                },
            )
            return {"draft": draft.model_dump(mode="json")}
        except Exception as e:
            raise RuntimeError(f"中书省生成行程失败: {str(e)}") from e

    async def _collect_research_context(self, normalized: dict[str, Any], *, request_id: str) -> str:
        destination = str(normalized.get("destination") or "").strip()
        if not destination:
            return "Live research skipped: destination missing."

        timeout_seconds = get_settings().mcp_tooling_timeout_seconds
        categories = {
            "global_place_discovery",
            "semantic_local_discovery",
            "domestic_poi_confirmation",
        }
        try:
            tools = await load_agent_tools("ZHONGSHU", categories)
        except Exception as exc:
            logger.warning(
                "Zhongshu live research tool loading failed request_id=%s destination=%s error_type=%s error=%s",
                request_id,
                destination,
                type(exc).__name__,
                exc,
            )
            return f"Live research unavailable: {type(exc).__name__}"

        discovery_tools = [
            tool
            for tool in tools
            if getattr(tool, "name", None) in {"search_google_maps", "search_local_places"}
        ]
        text_search = next((tool for tool in tools if getattr(tool, "name", None) == "maps_text_search"), None)
        if not discovery_tools:
            logger.info(
                "Zhongshu live research skipped because SerpAPI discovery tools are unavailable request_id=%s destination=%s",
                request_id,
                destination,
            )
            return "Live research unavailable: SerpAPI discovery tools missing."
        if text_search is None:
            logger.info(
                "Zhongshu live research skipped because Amap confirmation tool is unavailable request_id=%s destination=%s",
                request_id,
                destination,
            )
            return "Live research unavailable: Amap maps_text_search missing."

        interests = [str(item).strip() for item in normalized.get("interests", []) if str(item).strip()]
        queries = [f"{destination} {interest}" for interest in interests] or [destination]

        evidence: list[dict[str, Any]] = []
        candidates: list[str] = []
        for query in queries[:4]:
            discovery_tool = discovery_tools[0]
            tool_name = str(getattr(discovery_tool, "name", "unknown_tool"))
            args = {"query": query}
            if tool_name == "search_local_places":
                args["location"] = destination
            try:
                logger.info(
                    "Zhongshu live research invoking discovery tool request_id=%s destination=%s tool=%s query=%s",
                    request_id,
                    destination,
                    tool_name,
                    query,
                )
                result = await asyncio.wait_for(discovery_tool.ainvoke(args), timeout=timeout_seconds)
                decoded = _decode_tool_result(result)
                evidence.append({"phase": "discovery", "tool": tool_name, "status": "ok", "args": args, "result": decoded})
                candidates.extend(_extract_candidate_place_names(decoded))
            except asyncio.TimeoutError:
                logger.warning(
                    "Zhongshu live research discovery timed out request_id=%s destination=%s query=%s timeout_seconds=%s",
                    request_id,
                    destination,
                    query,
                    timeout_seconds,
                )
                evidence.append({"phase": "discovery", "tool": tool_name, "status": "timeout", "args": args})
            except Exception as exc:
                logger.warning(
                    "Zhongshu live research discovery failed request_id=%s destination=%s query=%s error_type=%s error=%s",
                    request_id,
                    destination,
                    query,
                    type(exc).__name__,
                    exc,
                )
                evidence.append({"phase": "discovery", "tool": tool_name, "status": "error", "args": args, "error": str(exc)})

        confirmed_queries = _dedupe_preserving_order(candidates)[:8]
        for place_name in confirmed_queries:
            args = {"keywords": place_name, "city": destination, "citylimit": True}
            try:
                logger.info(
                    "Zhongshu live research invoking Amap confirmation request_id=%s destination=%s place=%s",
                    request_id,
                    destination,
                    place_name,
                )
                result = await asyncio.wait_for(text_search.ainvoke(args), timeout=timeout_seconds)
                evidence.append({"phase": "confirmation", "tool": "maps_text_search", "status": "ok", "args": args, "result": _decode_tool_result(result)})
            except asyncio.TimeoutError:
                logger.warning(
                    "Zhongshu live research Amap confirmation timed out request_id=%s destination=%s place=%s timeout_seconds=%s",
                    request_id,
                    destination,
                    place_name,
                    timeout_seconds,
                )
                evidence.append({"phase": "confirmation", "tool": "maps_text_search", "status": "timeout", "args": args})
            except Exception as exc:
                logger.warning(
                    "Zhongshu live research Amap confirmation failed request_id=%s destination=%s place=%s error_type=%s error=%s",
                    request_id,
                    destination,
                    place_name,
                    type(exc).__name__,
                    exc,
                )
                evidence.append({"phase": "confirmation", "tool": "maps_text_search", "status": "error", "args": args, "error": str(exc)})

        if not any(item.get("phase") == "confirmation" and item.get("status") == "ok" for item in evidence):
            return "Live research unavailable: no successful SerpAPI-to-Amap confirmed place result."
        return json.dumps(evidence, ensure_ascii=False)

    async def decompose_tasks(self, state: ZhongshuState) -> dict[str, Any]:
        draft = ItineraryDraftModel.model_validate(state["draft"])
        normalized = state["normalized_request"]
        required_bureaus = self._infer_required_bureaus(draft, normalized)
        bureau_tasks = self._build_bureau_tasks(required_bureaus)
        return {"required_bureaus": required_bureaus, "bureau_tasks": [task.model_dump(mode="json") for task in bureau_tasks]}

    async def finalize_draft(self, state: ZhongshuState) -> dict[str, Any]:
        draft = ItineraryDraftModel.model_validate(state["draft"])
        packet = ZhongshuDraftPacketModel.model_validate(
            {
                "request_id": state["request_id"],
                "destination": draft.destination,
                "itinerary_draft": draft.model_dump(mode="json"),
                "required_bureaus": list(state.get("required_bureaus", [])),
                "bureau_tasks": [BureauTaskSpec.model_validate(item).model_dump(mode="json") for item in state.get("bureau_tasks", [])],
                "governance": {
                    "producer": "ZHONGSHU",
                    "next_hop": "MENXIA",
                    "review_required": True,
                    "source_state": "DRAFT",
                    "self_check_notes": [],
                    "human_intervened": False,
                    "revision_round": state["normalized_request"].get("revision_round", 0),
                    "rejection_reasons": state["normalized_request"].get("rejection_reasons", []),
                    "revision_requests": state["normalized_request"].get("revision_requests", []),
                },
            }
        )
        return {"finalized_packet": packet.model_dump(mode="json")}



    def _infer_required_bureaus(self, draft: ItineraryDraftModel, normalized: dict[str, Any]) -> list[str]:
        required = {"WEATHER", "CALENDAR", "BUDGET"}
        if draft.daily_plan:
            required.add("ACCOMMODATION")
        if normalized.get("origin_city"):
            required.add("FLIGHT_TRANSPORT")
        return sorted(required)

    def _build_bureau_tasks(self, required_bureaus: list[str]) -> list[BureauTaskSpec]:
        tasks: list[BureauTaskSpec] = []
        for bureau in required_bureaus:
            if bureau == "WEATHER":
                tasks.append(BureauTaskSpec(bureau="WEATHER", objective="Provide trip-date weather forecast, clothing advice, and packing list.", inputs_required=["destination", "daily_plan_dates"], deliverables=["forecast_days", "packing_list", "warnings"], priority="high"))
            elif bureau == "CALENDAR":
                tasks.append(BureauTaskSpec(bureau="CALENDAR", objective="Convert approved day blocks into importable .ics events.", inputs_required=["daily_plan", "activity_time_blocks"], deliverables=["calendar_file", "events_created"], priority="high"))
            elif bureau == "BUDGET":
                tasks.append(BureauTaskSpec(bureau="BUDGET", objective="Estimate total trip cost and produce a category-level budget table.", inputs_required=["daily_plan", "estimated_costs", "currency", "total_budget"], deliverables=["budget_breakdown", "total_estimated_cost", "warnings"], priority="high"))
            elif bureau == "ACCOMMODATION":
                tasks.append(BureauTaskSpec(bureau="ACCOMMODATION", objective="Recommend booking-ready accommodation options aligned to itinerary geography.", inputs_required=["destination", "daily_plan", "budget_level", "constraints"], deliverables=["hotel_options", "booking_links", "search_notes"], priority="medium"))
            elif bureau == "FLIGHT_TRANSPORT":
                tasks.append(BureauTaskSpec(bureau="FLIGHT_TRANSPORT", objective="Recommend inbound, outbound, and key local transport options.", inputs_required=["origin_city", "destination", "start_date", "end_date", "daily_plan"], deliverables=["flight_options", "transport_notes", "booking_links"], priority="medium"))
        return tasks


def _decode_tool_result(result: Any) -> Any:
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return result
    return result


def _extract_candidate_place_names(result: Any) -> list[str]:
    if not isinstance(result, dict):
        return []
    candidates: list[str] = []
    local_results = result.get("local_results")
    if isinstance(local_results, list):
        candidates.extend(_place_title(item) for item in local_results if isinstance(item, dict))
    place_results = result.get("place_results")
    if isinstance(place_results, dict):
        candidates.append(_place_title(place_results))
    organic_results = result.get("organic_results")
    if isinstance(organic_results, list):
        candidates.extend(_place_title(item) for item in organic_results if isinstance(item, dict))
    return [item for item in candidates if item]


def _place_title(item: dict[str, Any]) -> str:
    return str(item.get("title") or item.get("name") or "").strip()


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


