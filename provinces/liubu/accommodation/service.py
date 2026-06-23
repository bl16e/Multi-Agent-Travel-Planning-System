from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import quote_plus

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from provinces.liubu.constrained.gates import gate_accommodation_result, result_passed_gate
from provinces.liubu.constrained.state import LiubuToolEvidence, LiubuWorkerState, normalize_worker_input
from provinces.liubu.constrained.tools import execute_constrained_tool_call, load_allowed_tool_map
from utils.agent_runtime import escape_prompt_template_text, run_react_mcp_task, soul_path_for
from utils.schemas import AccommodationExecutionResult
from utils.llm_factory import build_qwen_chat
from utils.settings import get_settings

logger = logging.getLogger(__name__)
STRUCTURED_SYNTHESIS_TIMEOUT_SECONDS: float | None = None
ACCOMMODATION_ALLOWED_TOOLS = {"google_hotels", "google_maps", "search_poi", "nearby_search", "amap_place_search"}
MAX_TOOL_STEPS = 3


class AccommodationState(TypedDict, total=False):
    payload: dict[str, Any]
    destination: str
    profile: dict[str, Any]
    daily_plan: list[dict[str, Any]]
    research_notes: str
    result: dict[str, Any]


class AccommodationBureau:
    def __init__(self) -> None:
        self.soul_path = soul_path_for(__file__)
        self.graph = self._build_graph()

    async def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = await self.graph.ainvoke({"payload": payload})
        return result["result"]

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

    async def prepare_context(self, state: LiubuWorkerState) -> dict[str, Any]:
        worker_input = normalize_worker_input(state["payload"], "ACCOMMODATION")
        return {"worker_input": worker_input, "tool_evidence": [], "validation_findings": [], "warnings": []}

    async def agent_reasoning(self, state: LiubuWorkerState) -> dict[str, Any]:
        injected_reasoner = getattr(self, "_agent_reasoning", None)
        if injected_reasoner is not None:
            return await injected_reasoner(state)
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
        worker_input = state["worker_input"]
        evidence = [LiubuToolEvidence.model_validate(item) for item in state.get("tool_evidence", [])]
        live_notes = [str(item.result) for item in evidence if item.status == "ok"]
        research_notes = "\n".join(live_notes) if live_notes else "; ".join(item.error or item.status for item in evidence) or "No successful live hotel evidence."
        result = await self.synthesize_accommodation(
            {
                "destination": worker_input.destination,
                "profile": worker_input.profile,
                "daily_plan": worker_input.daily_plan,
                "research_notes": research_notes,
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

    def _route_after_reasoning(self, state: LiubuWorkerState) -> str:
        return "tool_execution" if state.get("tool_requests") else "structured_result"

    def _route_after_tools(self, state: LiubuWorkerState) -> str:
        return "structured_result"

    async def ingest(self, state: AccommodationState) -> dict[str, Any]:
        payload = state["payload"]
        approved_draft = payload.get("approved_draft", {})
        execution_plan = payload.get("execution_plan", {})
        user_request = execution_plan.get("user_request", {})
        draft = approved_draft.get("itinerary_draft", {})
        return {
            "destination": approved_draft.get("destination") or draft.get("destination") or "Unknown Destination",
            "profile": user_request.get("profile", {}),
            "daily_plan": draft.get("daily_plan", []),
        }

    async def research_accommodation(self, state: AccommodationState) -> dict[str, Any]:
        notes = await run_react_mcp_task(
            soul_path=self.soul_path,
            server_names=["amap", "serpapi"],
            user_task=(
                f"Find practical areas and hotel options in {state['destination']}. "
                f"Traveler profile: {state.get('profile', {})}. "
                "Use google_maps, google_hotels, search_poi, and nearby_search when available. Return concise notes about districts, hotel candidates, transport convenience, and booking caveats."
            ),
        )
        return {"research_notes": notes}

    async def synthesize_accommodation(self, state: AccommodationState) -> dict[str, Any]:
        llm = build_qwen_chat()
        if llm:
            try:
                structured = llm.with_structured_output(AccommodationExecutionResult)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", escape_prompt_template_text(Path(self.soul_path).read_text(encoding="utf-8"))),
                    ("user", "Destination: {destination}\nProfile: {profile}\nResearch notes: {research_notes}\nReturn valid JSON structured accommodation output."),
                ])
                result = await asyncio.wait_for(
                    (prompt | structured).ainvoke({"destination": state["destination"], "profile": str(state.get("profile", {})), "research_notes": state.get("research_notes", "")}),
                    timeout=STRUCTURED_SYNTHESIS_TIMEOUT_SECONDS or get_settings().qwen_timeout_seconds,
                )
                data = result.model_dump(mode="json")
                data.update({"status": "ok", "data_source": "structured_llm"})
                return {"result": data}
            except asyncio.TimeoutError:
                logger.warning("Accommodation structured synthesis timed out; using fallback result.")
                failure_note = "Accommodation structured synthesis timed out."
            except Exception as exc:
                logger.warning("Accommodation structured synthesis failed; using fallback result: %s", exc)
                failure_note = f"Accommodation structured synthesis failed: {exc}"
        else:
            failure_note = "Accommodation structured synthesis unavailable; using fallback estimate."
        destination = state["destination"]
        currency = state.get("profile", {}).get("currency", "USD")
        zones = ["Central Station Area", "Old Town Core", "Museum Quarter"]
        hotels = []
        nights = max(len(state.get("daily_plan", [])) - 1, 1)
        fallback_note = "Estimated fallback; did not use real-time data. Confirm availability and rates before booking."
        research_note = state.get("research_notes") or "MCP or LLM unavailable."
        for index, zone in enumerate(zones, start=1):
            query = quote_plus(f"{destination} {zone} hotel")
            nightly_rate = 120 + index * 20
            hotels.append({"name": f"{destination} {zone} Hotel {index}", "nightly_rate": nightly_rate, "total_rate": nightly_rate * nights, "currency": currency, "rating": 4.0 + (index * 0.2), "booking_link": f"https://www.booking.com/searchresults.html?ss={query}", "address": f"{zone}, {destination}", "notes": f"{fallback_note} {research_note}"})
        return {"result": AccommodationExecutionResult(destination=destination, hotel_options=hotels, booking_links=[item["booking_link"] for item in hotels], search_notes=[fallback_note, research_note, failure_note], warnings=[failure_note]).model_dump(mode="json")}
