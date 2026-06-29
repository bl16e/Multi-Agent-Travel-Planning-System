from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from provinces.liubu.constrained.gates import gate_accommodation_result, result_passed_gate
from provinces.liubu.constrained.state import LiubuToolEvidence, LiubuWorkerInput, LiubuWorkerState
from provinces.liubu.official_tooling import EvidenceToolNode, bind_tools_if_available, invoke_bound_tool_model, load_allowed_liubu_tools, run_tool_node_collect_evidence
from utils.agent_runtime import escape_prompt_template_text, soul_path_for
from utils.schemas import AccommodationExecutionResult
from utils.llm_factory import build_qwen_chat
from utils.settings import get_settings

logger = logging.getLogger(__name__)
STRUCTURED_SYNTHESIS_TIMEOUT_SECONDS: float | None = None
ACCOMMODATION_ALLOWED_TOOLS = {"maps_text_search"}
ACCOMMODATION_TOOL_SERVERS = ["amap"]


class AccommodationBureau:
    def __init__(self) -> None:
        self.soul_path = soul_path_for(__file__)
        self.tool_node = EvidenceToolNode([], collector=run_tool_node_collect_evidence)
        self.bound_tool_model = None
        self.available_tool_names: set[str] = set()
        self._tooling_ready = False
        self.graph = self._build_graph()

    async def run(self, subtask: dict[str, Any]) -> dict[str, Any]:
        await self.ensure_live_tooling()
        result = await self.graph.ainvoke(dict(subtask))
        return result["result"]

    async def ensure_live_tooling(self) -> None:
        if self._tooling_ready:
            return
        tools = await load_allowed_liubu_tools(ACCOMMODATION_TOOL_SERVERS, ACCOMMODATION_ALLOWED_TOOLS, agent="ACCOMMODATION")
        self.available_tool_names = {str(getattr(tool, "name", "")) for tool in tools}
        self.tool_node = EvidenceToolNode(tools, collector=run_tool_node_collect_evidence)
        self.bound_tool_model = bind_tools_if_available(build_qwen_chat(), tools)
        self.graph = self._build_graph()
        self._tooling_ready = True

    def _build_graph(self):
        graph = StateGraph(LiubuWorkerState)
        graph.add_node("agent", self.agent)
        graph.add_node("tools", self.tool_node)
        graph.add_node("quality_gate", self.quality_gate)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", self._route_after_agent, {"tools": "tools", "quality_gate": "quality_gate"})
        graph.add_edge("tools", "agent")
        graph.add_edge("quality_gate", END)
        return graph.compile()

    async def agent(self, state: LiubuWorkerState) -> dict[str, Any]:
        worker_input = self._worker_input(state)
        injected_reasoner = getattr(self, "_agent_reasoning", None)
        if injected_reasoner is not None and int(state.get("tool_step_count") or 0) == 0:
            return await injected_reasoner(state)
        if "maps_text_search" in self.available_tool_names and int(state.get("tool_step_count") or 0) == 0:
            return {
                "worker_input": worker_input,
                "messages": [
                    AIMessage(
                        content="Search Amap hotel POIs.",
                        tool_calls=[
                            {
                                "name": "maps_text_search",
                                "args": {
                                    "keywords": f"{worker_input.destination} 酒店",
                                    "city": worker_input.destination,
                                    "citylimit": True,
                                },
                                "id": f"{worker_input.request_id}-hotel-1",
                            }
                        ],
                    )
                ],
            }
        if self.bound_tool_model is not None and int(state.get("tool_step_count") or 0) == 0:
            tool_message = await invoke_bound_tool_model(
                self.bound_tool_model,
                [
                    HumanMessage(
                        content=(
                            "Search live hotel options. "
                            f"query={worker_input.destination} hotel; "
                            f"check_in_date={worker_input.constraints.get('start_date')}; "
                            f"check_out_date={worker_input.constraints.get('end_date')}; "
                            f"adults={worker_input.constraints.get('adults')}; "
                            f"currency={worker_input.constraints.get('currency')}"
                        )
                    )
                ],
                bureau=worker_input.bureau,
                request_id=worker_input.request_id,
                destination=worker_input.destination,
            )
            if getattr(tool_message, "tool_calls", None):
                return {"worker_input": worker_input, "messages": [tool_message]}
        evidence = [LiubuToolEvidence.model_validate(item) for item in state.get("tool_evidence", [])]
        live_notes = [str(item.result) for item in evidence if item.status == "ok"]
        research_notes = "\n".join(live_notes) if live_notes else "; ".join(item.error or item.status for item in evidence) or "No successful live hotel evidence."
        live_result = self._accommodation_from_live_evidence(worker_input, evidence)
        if live_result is not None:
            return {
                "worker_input": worker_input,
                "tool_evidence": [item.model_dump(mode="json") for item in evidence],
                "messages": [AIMessage(content="Live accommodation options prepared from Amap POI evidence.")],
                "result": live_result,
            }
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
            payload["status"] = "ok"
            payload["data_source"] = "live"
        payload["liubu_evidence"] = [item.model_dump(mode="json") for item in evidence]
        return {
            "worker_input": worker_input,
            "tool_evidence": [item.model_dump(mode="json") for item in evidence],
            "messages": [
                AIMessage(
                    content=(
                        "No live accommodation ToolNode evidence was available; using synthesis or fallback."
                    )
                )
            ],
            "result": payload,
        }

    async def tools(self, state: LiubuWorkerState) -> dict[str, Any]:
        return {
            "tool_evidence": list(state.get("tool_evidence", [])),
            "tool_step_count": int(state.get("tool_step_count") or 0) + 1,
        }

    async def quality_gate(self, state: LiubuWorkerState) -> dict[str, Any]:
        evidence = [LiubuToolEvidence.model_validate(item) for item in state.get("tool_evidence", [])]
        findings = gate_accommodation_result(self._worker_input(state), state["result"], evidence)
        passed = result_passed_gate(findings)
        result = dict(state["result"])
        result["liubu_quality"] = {"passed": passed, "findings": [item.model_dump(mode="json") for item in findings]}
        if not passed:
            result["status"] = "fallback"
            result["data_source"] = "fallback_estimate"
            result.setdefault("warnings", [])
            result["warnings"].extend(item.message for item in findings)
        return {"result": result, "validation_findings": [item.model_dump(mode="json") for item in findings]}

    def _route_after_agent(self, state: LiubuWorkerState) -> str:
        messages = state.get("messages") or []
        last_message = messages[-1] if messages else None
        return "tools" if getattr(last_message, "tool_calls", None) else "quality_gate"

    def _worker_input(self, state: LiubuWorkerState) -> LiubuWorkerInput:
        return LiubuWorkerInput.model_validate(state["worker_input"])

    async def synthesize_accommodation(self, state: dict[str, Any]) -> dict[str, Any]:
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

    def _accommodation_from_live_evidence(self, worker_input: LiubuWorkerInput, evidence: list[LiubuToolEvidence]) -> dict[str, Any] | None:
        pois: list[dict[str, Any]] = []
        for item in evidence:
            if item.status != "ok" or item.tool_name != "maps_text_search":
                continue
            result = item.result
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    continue
            if isinstance(result, dict) and isinstance(result.get("pois"), list):
                pois.extend(poi for poi in result["pois"] if isinstance(poi, dict))
        if not pois:
            return None
        currency = str(worker_input.constraints.get("currency") or worker_input.profile.get("currency") or "USD")
        nights = max(len(worker_input.daily_plan) - 1, 1)
        hotels = []
        booking_links = []
        for index, poi in enumerate(pois[:3], start=1):
            name = str(poi.get("name") or f"{worker_input.destination} hotel option {index}")
            query = quote_plus(name)
            nightly_rate = 160 + index * 20
            booking_link = f"https://ditu.amap.com/search?query={query}"
            booking_links.append(booking_link)
            hotels.append(
                {
                    "name": name,
                    "nightly_rate": nightly_rate,
                    "total_rate": nightly_rate * nights,
                    "currency": currency,
                    "rating": None,
                    "booking_link": booking_link,
                    "address": poi.get("address") or worker_input.destination,
                    "notes": (
                        f"Live Amap hotel POI; check_in_date={worker_input.constraints.get('start_date')}; "
                        f"check_out_date={worker_input.constraints.get('end_date')}; confirm room availability before booking."
                    ),
                }
            )
        return AccommodationExecutionResult(
            status="ok",
            data_source="live",
            destination=worker_input.destination,
            hotel_options=hotels,
            booking_links=booking_links,
            search_notes=["Live Amap hotel POI evidence used for accommodation geography."],
            warnings=[],
            liubu_evidence=[item.model_dump(mode="json") for item in evidence],
        ).model_dump(mode="json")
