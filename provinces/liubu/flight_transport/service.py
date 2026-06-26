from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from provinces.liubu.constrained.gates import gate_flight_transport_result, result_passed_gate
from provinces.liubu.constrained.state import LiubuToolEvidence, LiubuWorkerInput, LiubuWorkerState
from provinces.liubu.official_tooling import EvidenceToolNode, bind_tools_if_available, invoke_bound_tool_model, load_allowed_liubu_tools, run_tool_node_collect_evidence
from utils.agent_runtime import escape_prompt_template_text, soul_path_for
from utils.schemas import FlightTransportExecutionResult
from utils.llm_factory import build_qwen_chat
from utils.settings import get_settings

logger = logging.getLogger(__name__)
STRUCTURED_SYNTHESIS_TIMEOUT_SECONDS: float | None = None
FLIGHT_ALLOWED_TOOLS = {"search_google_flights"}
FLIGHT_TOOL_SERVERS = ["serpapi"]


class FlightTransportBureau:
    def __init__(self) -> None:
        self.soul_path = soul_path_for(__file__)
        self.tool_node = EvidenceToolNode([], collector=run_tool_node_collect_evidence)
        self.bound_tool_model = None
        self._tooling_ready = False
        self.graph = self._build_graph()

    async def run(self, subtask: dict[str, Any]) -> dict[str, Any]:
        await self.ensure_live_tooling()
        result = await self.graph.ainvoke(dict(subtask))
        return result["result"]

    async def ensure_live_tooling(self) -> None:
        if self._tooling_ready:
            return
        tools = await load_allowed_liubu_tools(FLIGHT_TOOL_SERVERS, FLIGHT_ALLOWED_TOOLS)
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
        if self.bound_tool_model is not None and int(state.get("tool_step_count") or 0) == 0:
            tool_message = await invoke_bound_tool_model(
                self.bound_tool_model,
                [
                    HumanMessage(
                        content=(
                            "Search live flight options. "
                            f"departure_id={worker_input.constraints.get('origin_airport_code') or worker_input.profile.get('origin_city')}; "
                            f"arrival_id={worker_input.constraints.get('destination_airport_code') or worker_input.destination}; "
                            f"outbound_date={worker_input.constraints.get('start_date')}; "
                            f"return_date={worker_input.constraints.get('end_date')}; "
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
        research_notes = "\n".join(live_notes) if live_notes else "; ".join(item.error or item.status for item in evidence) or "No successful live flight evidence."
        result = await self.synthesize_transport(
            {
                "origin_city": worker_input.profile.get("origin_city") or "Unknown origin",
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
                        "No live flight ToolNode evidence was available; using synthesis or fallback."
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
        findings = gate_flight_transport_result(self._worker_input(state), state["result"], evidence)
        passed = result_passed_gate(findings)
        result = dict(state["result"])
        result["liubu_quality"] = {"passed": passed, "findings": [item.model_dump(mode="json") for item in findings]}
        if not passed:
            result["status"] = "fallback"
            result["data_source"] = "fallback_estimate"
            result.setdefault("transport_notes", [])
            result["transport_notes"].extend(item.message for item in findings)
        return {"result": result, "validation_findings": [item.model_dump(mode="json") for item in findings]}

    def _route_after_agent(self, state: LiubuWorkerState) -> str:
        messages = state.get("messages") or []
        last_message = messages[-1] if messages else None
        return "tools" if getattr(last_message, "tool_calls", None) else "quality_gate"

    def _worker_input(self, state: LiubuWorkerState) -> LiubuWorkerInput:
        return LiubuWorkerInput.model_validate(state["worker_input"])

    async def synthesize_transport(self, state: dict[str, Any]) -> dict[str, Any]:
        llm = build_qwen_chat()
        if llm:
            try:
                structured = llm.with_structured_output(FlightTransportExecutionResult)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", escape_prompt_template_text(Path(self.soul_path).read_text(encoding="utf-8"))),
                    ("user", "Origin: {origin}\nDestination: {destination}\nProfile: {profile}\nResearch notes: {research_notes}\nReturn valid JSON structured transport result."),
                ])
                result = await asyncio.wait_for(
                    (prompt | structured).ainvoke({"origin": state["origin_city"], "destination": state["destination"], "profile": str(state.get("profile", {})), "research_notes": state.get("research_notes", "")}),
                    timeout=STRUCTURED_SYNTHESIS_TIMEOUT_SECONDS or get_settings().qwen_timeout_seconds,
                )
                data = result.model_dump(mode="json")
                data.update({"status": "ok", "data_source": "structured_llm"})
                return {"result": data}
            except asyncio.TimeoutError:
                logger.warning("Flight transport structured synthesis timed out; using fallback result.")
                failure_note = "Flight transport structured synthesis timed out."
            except Exception as exc:
                logger.warning("Flight transport structured synthesis failed; using fallback result: %s", exc)
                failure_note = f"Flight transport structured synthesis failed: {exc}"
        else:
            failure_note = "Flight transport structured synthesis unavailable; using fallback estimate."
        origin_city = state["origin_city"]
        destination = state["destination"]
        profile = state.get("profile", {})
        departure_airport = profile.get("origin_airport_code") or origin_city
        arrival_airport = profile.get("destination_airport_code") or destination
        route_query = quote_plus(f"{origin_city} to {destination} flights")
        fallback_note = "Estimated fallback; did not use real-time data. Confirm carrier, routing, fare, and timing before booking."
        research_note = state.get("research_notes") or "MCP or LLM unavailable."
        departure_date = profile.get("start_date") or self._first_trip_date(state.get("daily_plan", []))
        options = [
            {"airline": "Estimated direct-flight option", "price": 320.0, "currency": profile.get("currency", "USD"), "departure_airport": departure_airport, "arrival_airport": arrival_airport, "departure_time": f"{departure_date} 08:30", "arrival_time": f"{departure_date} 12:15", "duration_minutes": 225, "booking_link": f"https://www.google.com/travel/flights?q={route_query}", "notes": f"{fallback_note} {research_note}"},
            {"airline": "Estimated connection option", "price": 255.0, "currency": profile.get("currency", "USD"), "departure_airport": departure_airport, "arrival_airport": arrival_airport, "departure_time": f"{departure_date} 10:20", "arrival_time": f"{departure_date} 15:50", "duration_minutes": 330, "booking_link": f"https://www.skyscanner.com/transport/flights/{route_query}", "notes": f"{fallback_note} {research_note}"},
        ]
        return {"result": FlightTransportExecutionResult(origin=origin_city, destination=destination, flight_options=options, transport_notes=[fallback_note, research_note, failure_note], booking_links=[item["booking_link"] for item in options]).model_dump(mode="json")}

    def _first_trip_date(self, daily_plan: list[dict[str, Any]]) -> str:
        for day in daily_plan:
            if day.get("date"):
                return str(day["date"])
        return "unknown-date"
