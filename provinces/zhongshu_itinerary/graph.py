from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from utils.agent_runtime import run_structured_synthesis, soul_path_for
from utils.mcp_client import load_mcp_tools
from utils.mcp_tool_registry import load_agent_tools
from utils.schemas import (
    BureauTaskSpec,
    ItineraryDraftModel,
    SerpApiCandidateSelectionModel,
    SerpApiQueryPlanModel,
    ZhongshuDraftPacketModel,
)
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
                    "请根据用户需求生成详细的旅行行程草案（全程中文输出）。\n"
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
                    "关键规则:\n"
                    "1. 全程中文：所有地点名、活动标题、描述、摘要必须为中文。研究背景中的英文名必须翻译为规范中文名。\n"
                    "2. 每天必须有一个明确主题（来自用户兴趣列表），当天所有活动必须与该主题严格匹配。\n"
                    "   例如「food/美食」主题日只能安排餐厅、小吃街、夜市、烹饪课等饮食相关活动，禁止安排博物馆或展览馆。\n"
                    "3. 只选择地址确实在 {destination} 市内的真实地点。地址在国外的同名商户绝对不能选。\n"
                    "4. 每个活动必须有研究背景中确认过的真实地点名称、地址和链接，不要编造。\n"
                    "5. 每天安排2-3个活动，留出用餐和交通时间。\n"
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
            "web_search_discovery",
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
            if getattr(tool, "name", None) in {"search_google_web", "search_google_maps", "search_local_places"}
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

        evidence: list[dict[str, Any]] = []
        planned_queries = await self._plan_serpapi_queries(normalized, discovery_tools, evidence)
        candidates_by_id: dict[str, dict[str, Any]] = {}
        for query_index, planned in enumerate(planned_queries):
            tool_name = str(planned.get("tool") or "search_google_maps")
            discovery_tool = next((tool for tool in discovery_tools if getattr(tool, "name", None) == tool_name), discovery_tools[0])
            tool_name = str(getattr(discovery_tool, "name", "unknown_tool"))
            query = str(planned.get("query") or destination).strip()
            args = {"query": query}
            location = str(planned.get("location") or "").strip()
            if tool_name == "search_local_places":
                args["location"] = location or destination
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
                result_candidates: dict[str, dict[str, Any]] = {}
                for result_index, candidate in enumerate(_extract_candidate_places(decoded)):
                    candidate_id = f"{tool_name}:{query_index}:{result_index}"
                    enriched = {**candidate, "_candidate_id": candidate_id}
                    candidates_by_id[candidate_id] = enriched
                    result_candidates[candidate_id] = enriched
                discovery_result: dict[str, Any] = {"candidates": _candidate_summaries(result_candidates)}
                # Preserve raw organic_results for web search extraction step
                if tool_name == "search_google_web" and isinstance(decoded, dict):
                    org = decoded.get("organic_results")
                    if isinstance(org, list):
                        discovery_result["organic_results"] = org
                evidence.append({"phase": "discovery", "tool": tool_name, "status": "ok", "args": args, "result": discovery_result})
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

        # --- Post-discovery: extract concrete place names from web search results ---
        web_organic_results: list[dict[str, Any]] = []
        for item in evidence:
            if item.get("phase") != "discovery" or item.get("tool") != "search_google_web":
                continue
            decoded = item.get("result", {})
            if isinstance(decoded, dict):
                # Organic results may be at top level (from discovery) or nested in candidates
                org = decoded.get("organic_results") or []
                if not org:
                    candidates_val = decoded.get("candidates")
                    if isinstance(candidates_val, dict):
                        org = candidates_val.get("organic_results") or []
                if isinstance(org, list):
                    web_organic_results.extend(r for r in org if isinstance(r, dict))

        if web_organic_results and candidates_by_id:
            extracted_names = await self._extract_place_names_from_web(
                destination, web_organic_results, normalized
            )
            if extracted_names:
                # Build synthetic candidates from extracted place names.
                # These bypass _select_serpapi_candidates (which expects
                # map/local-result fields like address/place_id) and go
                # directly to Amap confirmation.
                extracted_candidates: dict[str, dict[str, Any]] = {}
                for idx, name in enumerate(extracted_names):
                    cid = f"extracted_place:{idx}"
                    extracted_candidates[cid] = {"_candidate_id": cid, "title": name, "name": name, "_extracted": True}
                # Replace web article candidates with extracted real place names
                maps_candidates = {k: v for k, v in candidates_by_id.items() if "search_google_web" not in k}
                candidates_by_id = {**maps_candidates, **extracted_candidates}
                evidence.append({"phase": "extraction", "tool": "search_google_web", "status": "ok", "result": {"extracted_place_names": extracted_names}})

        confirmed_places = await self._select_serpapi_candidates(normalized, candidates_by_id, evidence)
        # --- Also confirm extracted place names directly via Amap ---
        extracted = [c for c in candidates_by_id.values() if c.get("_extracted")]
        if extracted:
            confirmed_places = list(confirmed_places) + extracted
        for source_place in confirmed_places:
            place_name = str(source_place.get("title") or source_place.get("name") or "").strip()
            if not place_name:
                continue
            args = {"keywords": place_name, "city": destination, "citylimit": True}
            try:
                logger.info(
                    "Zhongshu live research invoking Amap confirmation request_id=%s destination=%s place=%s",
                    request_id,
                    destination,
                    place_name,
                )
                result = await asyncio.wait_for(text_search.ainvoke(args), timeout=timeout_seconds)
                decoded = _decode_tool_result(result)
                # --- Geo-filter: reject POIs whose Amap results don't match the destination city ---
                if not _amap_result_matches_city(decoded, destination):
                    logger.info(
                        "Zhongshu Amap confirmation rejected non-local POI request_id=%s place=%s destination=%s",
                        request_id,
                        place_name,
                        destination,
                    )
                    evidence.append({"phase": "confirmation", "tool": "maps_text_search", "status": "rejected", "args": args, "reason": f"Amap result address does not match destination city '{destination}'"})
                    continue
                evidence.append({"phase": "confirmation", "tool": "maps_text_search", "status": "ok", "args": args, "source_place": source_place, "result": decoded})
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

    async def _extract_place_names_from_web(
        self,
        destination: str,
        organic_results: list[dict[str, Any]],
        normalized: dict[str, Any],
    ) -> list[str]:
        """Use LLM to extract concrete attraction/restaurant names from web search results.

        Web search returns articles and blog posts *about* attractions (e.g.
        "上海10大必去景点"), not the attractions themselves.  The LLM reads
        the titles and snippets and returns ONLY the actual place names.
        """
        if not organic_results:
            return []

        # Collect titles and snippets (max 15 results to stay within context)
        snippets_text = ""
        for i, r in enumerate(organic_results[:15]):
            title = str(r.get("title") or "").strip()
            snippet = str(r.get("snippet") or r.get("about") or "").strip()
            if title:
                snippets_text += f"{i+1}. {title}"
                if snippet:
                    snippets_text += f" — {snippet[:200]}"
                snippets_text += "\n"

        if not snippets_text.strip():
            return []

        try:
            from utils.llm_factory import build_qwen_chat
            from langchain_core.messages import HumanMessage, SystemMessage
            from pydantic import BaseModel, Field

            llm = build_qwen_chat()
            if llm is None:
                logger.warning("Place name extraction skipped: LLM unavailable")
                return []

            class PlaceNameList(BaseModel):
                places: list[str] = Field(
                    default_factory=list,
                    description="具体景点/餐厅/地标名称列表",
                )

            structured = llm.with_structured_output(PlaceNameList)
            result = await structured.ainvoke([
                SystemMessage(content=(
                    "你是一个旅行地点提取助手。从网页搜索结果的标题和摘要中，"
                    "提取出文章中提到的具体景点、餐厅、博物馆名称。"
                    "输出格式为 JSON：{\"places\": [\"景点1\", \"景点2\", ...]}。\n"
                    "规则：\n"
                    "1. 只提取具体地点名（如「外滩」「豫园」），不要文章标题\n"
                    "2. 每个名称必须是真实存在的景点或餐厅\n"
                    "3. 去重，每处只列一次\n"
                    "4. 返回5-12个"
                )),
                HumanMessage(content=(
                    f"从以下关于 {destination} 的搜索结果中提取具体地点名：\n\n{snippets_text}"
                )),
            ])
            return [p.strip() for p in result.places if p.strip()]
        except Exception as exc:
            logger.warning(
                "Place name extraction from web results failed: %s", exc
            )
            return []

    async def _plan_serpapi_queries(
        self,
        normalized: dict[str, Any],
        discovery_tools: list[Any],
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        destination = str(normalized.get("destination") or "").strip()
        available_tools = [str(getattr(tool, "name", "")) for tool in discovery_tools if getattr(tool, "name", "")]
        interests = [str(item).strip() for item in normalized.get("interests", []) if str(item).strip()]
        try:
            plan = await run_structured_synthesis(
                soul_path=self.soul_path,
                output_model=SerpApiQueryPlanModel,
                user_prompt=(
                    "你是中书省旅行研究员。请生成 SerpAPI 检索计划，不要硬拼目的地和兴趣词。\n"
                    "目的地: {destination}\n兴趣: {interests}\n约束: {constraints}\n用户原文: {user_message}\n"
                    "可用工具: {available_tools}\n"
                    "返回 2-4 个适合发现真实 POI 的 query，优先能找到具体地点、官网或 provider place_id 的查询。"
                ),
                variables={
                    "destination": destination,
                    "interests": ", ".join(interests),
                    "constraints": ", ".join(str(item) for item in normalized.get("constraints", [])),
                    "user_message": normalized.get("user_message", ""),
                    "available_tools": ", ".join(available_tools),
                },
            )
            queries = [
                item.model_dump(mode="json")
                for item in plan.queries
                if item.query.strip() and item.tool in available_tools
            ][:4]
            if queries:
                evidence.append({"phase": "query_planning", "status": "ok", "result": plan.model_dump(mode="json")})
                return queries
        except Exception as exc:
            evidence.append({"phase": "query_planning", "status": "error", "error": str(exc)})

        # Simple algorithmic fallback: "{destination} {interest}" for each interest.
        # The LLM-based planning above is the primary path; this only runs on LLM failure.
        # Prefer web search for discovery (finds curated attraction lists),
        # fall back to maps search if web search isn't available.
        pref = "search_google_web" if "search_google_web" in available_tools else (available_tools[0] if available_tools else "search_google_maps")
        fallback = [{"tool": pref, "query": f"{destination} {interest} 景点 推荐", "location": destination, "reason": "deterministic fallback"} for interest in interests[:4]]
        if not fallback:
            fallback = [{"tool": pref, "query": f"{destination} 必去景点 推荐", "location": destination, "reason": "deterministic fallback"}]
        evidence.append({"phase": "query_planning_fallback", "status": "fallback", "result": fallback})
        return fallback

    async def _select_serpapi_candidates(
        self,
        normalized: dict[str, Any],
        candidates_by_id: dict[str, dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not candidates_by_id:
            evidence.append({"phase": "candidate_selection", "status": "empty", "result": []})
            return []
        selection_candidates = _candidate_dict_for_selection(normalized, candidates_by_id)
        summaries = _candidate_summaries(selection_candidates)
        try:
            selection = await run_structured_synthesis(
                soul_path=self.soul_path,
                output_model=SerpApiCandidateSelectionModel,
                user_prompt=(
                    "你是中书省旅行研究员。请从 SerpAPI 候选中选择要交给高德确认的真实 POI。\n"
                    "目的地: {destination}\n兴趣: {interests}\n约束: {constraints}\n候选摘要 JSON: {candidate_summaries}\n"
                    "重要: 若兴趣包含 food/美食，必须优先选择餐厅、小吃、夜市、美食街等饮食类 POI。\n"
                    "只返回候选中已有的 candidate_id，不要编造地点或链接。排除停车场、泛商业楼、低相关或非旅行 POI。"
                ),
                variables={
                    "destination": normalized.get("destination", ""),
                    "interests": ", ".join(str(item) for item in normalized.get("interests", [])),
                    "constraints": ", ".join(str(item) for item in normalized.get("constraints", [])),
                    "candidate_summaries": json.dumps(summaries, ensure_ascii=False),
                },
            )
            selected: list[dict[str, Any]] = []
            for item in selection.selected_candidates:
                candidate = selection_candidates.get(item.candidate_id) or candidates_by_id.get(item.candidate_id)
                if candidate is not None:
                    selected.append(candidate)
            selected = _dedupe_places(selected)[:8]
            if selected:
                evidence.append({"phase": "candidate_selection", "status": "ok", "result": selection.model_dump(mode="json")})
                return selected
        except Exception as exc:
            evidence.append({"phase": "candidate_selection", "status": "error", "error": str(exc)})

        fallback = _fallback_relevant_candidates(normalized, list(candidates_by_id.values()))[:8]
        evidence.append({"phase": "candidate_selection_fallback", "status": "fallback", "result": _candidate_summaries({str(item.get("_candidate_id") or index): item for index, item in enumerate(fallback)})})
        return fallback

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


def _extract_candidate_places(result: Any) -> list[dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    candidates: list[dict[str, Any]] = []
    local_results = result.get("local_results")
    if isinstance(local_results, list):
        candidates.extend(item for item in local_results if isinstance(item, dict) and _place_title(item))
    place_results = result.get("place_results")
    if isinstance(place_results, dict) and _place_title(place_results):
        candidates.append(place_results)
    organic_results = result.get("organic_results")
    if isinstance(organic_results, list):
        candidates.extend(item for item in organic_results if isinstance(item, dict) and _place_title(item))
    return candidates


def _place_title(item: dict[str, Any]) -> str:
    return str(item.get("title") or item.get("name") or "").strip()


def _dedupe_places(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for value in values:
        title = _place_title(value)
        if title in seen:
            continue
        seen.add(title)
        output.append(value)
    return output


def _candidate_summaries(candidates_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for candidate_id, item in candidates_by_id.items():
        summaries.append(
            {
                "candidate_id": candidate_id,
                "title": item.get("title") or item.get("name"),
                "type": item.get("type"),
                "address": item.get("address"),
                "rating": item.get("rating"),
                "reviews": item.get("reviews"),
                "place_id": item.get("place_id"),
                "website": item.get("website"),
                "gps_coordinates": item.get("gps_coordinates"),
            }
        )
    return summaries


def _fallback_relevant_candidates(normalized: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = _dedupe_places(candidates)
    scored = [(_candidate_relevance_score(normalized, item), index, item) for index, item in enumerate(deduped)]
    positive = [(score, index, item) for score, index, item in scored if score > 0]
    if not positive:
        return deduped[:8]
    positive.sort(key=lambda value: (-value[0], value[1]))
    return [item for _, _, item in positive]


def _candidate_dict_for_selection(normalized: dict[str, Any], candidates_by_id: dict[str, dict[str, Any]], *, limit: int = 16) -> dict[str, dict[str, Any]]:
    ranked = _fallback_relevant_candidates(normalized, list(candidates_by_id.values()))[:limit]
    return {str(item.get("_candidate_id")): item for item in ranked if item.get("_candidate_id")}


def _amap_result_matches_city(result: Any, destination_city: str) -> bool:
    """Return True if *result* from Amap maps_text_search contains at least one
    POI whose city/address suggests it's actually in *destination_city*."""
    if not isinstance(result, dict):
        return False
    pois = result.get("pois") or []
    if not isinstance(pois, list) or not pois:
        return False
    dest = destination_city.strip()
    for poi in pois:
        if not isinstance(poi, dict):
            continue
        city = str(poi.get("cityname") or poi.get("pname") or "")
        address = str(poi.get("address") or "")
        name = str(poi.get("name") or "")
        combined = f"{city} {address} {name}"
        if dest in combined:
            return True
    return False


def _candidate_relevance_score(normalized: dict[str, Any], item: dict[str, Any]) -> int:
    text = " ".join(str(item.get(key) or "") for key in ("title", "name", "type", "address")).lower()
    if any(term in text for term in ("garage", "parking", "car park", "hotel", "lounge", "bar", "restaurant")):
        return -10
    score = 0
    if item.get("place_id"):
        score += 1
    if item.get("website"):
        score += 1
    interests = " ".join(str(value).lower() for value in normalized.get("interests", []))
    if any(term in interests for term in ("文化", "culture", "museum", "art", "历史")):
        for term in ("museum", "博物馆", "art", "美术馆", "gallery", "history", "历史", "garden", "园", "landmark", "文化"):
            if term in text:
                score += 4
    if any(term in interests for term in ("food", "美食", "餐", "小吃")):
        for term in ("food", "restaurant", "market", "street", "小吃", "餐", "美食", "夜市"):
            if term in text:
                score += 4
    return score


