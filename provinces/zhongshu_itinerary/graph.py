from __future__ import annotations

from datetime import date, timedelta
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from utils.agent_runtime import run_react_mcp_task, run_structured_synthesis, soul_path_for
from utils.schemas import BureauTaskSpec, ItineraryDraftModel, ZhongshuDraftPacketModel


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
        destination = destination_preferences[0] if destination_preferences else "Tokyo"
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
            research_context = await run_react_mcp_task(
                soul_path=self.soul_path,
                server_names=["serpapi"],
                user_task=(
                    f"搜索{normalized['destination']}的热门景点，与以下兴趣相关：{', '.join(normalized.get('interests', []))}。"
                    f"找到具体景点名称、地址和预订信息。"
                ),
            )

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
                timeout_seconds=500.0,
            )
            return {"draft": draft.model_dump(mode="json")}
        except Exception as e:
            raise RuntimeError(f"中书省生成行程失败: {str(e)}") from e

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

    def _build_pending_confirmations(self, normalized: dict[str, Any]) -> list[str]:
        items = [
            "Confirm final accommodation booking before issuing the calendar bundle.",
            "Confirm one primary paid attraction per day to reduce queue risk.",
        ]
        if normalized.get("total_budget") is None:
            items.append("Confirm total budget cap so Budget bureau can evaluate overruns.")
        if not normalized.get("origin_city"):
            items.append("Confirm departure city or airport before Flight & Transport research.")
        return items

    def _build_risk_flags(self, normalized: dict[str, Any]) -> list[str]:
        flags = [
            "Opening hours and ticket availability may change and must be reviewed downstream.",
            "Weather suitability is not yet validated and may alter outdoor blocks.",
        ]
        if normalized.get("constraints"):
            flags.append("User constraints may invalidate some attractions and require review filtering.")
        return flags

    def _parse_date(self, value: Any) -> date | None:
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError:
                return None
        return None
