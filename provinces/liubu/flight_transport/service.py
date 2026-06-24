from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import quote_plus

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import tools_condition

from provinces.liubu.constrained.gates import gate_flight_transport_result, result_passed_gate
from provinces.liubu.constrained.state import LiubuToolEvidence, LiubuWorkerState, normalize_worker_input
from provinces.liubu.constrained.tool_node import load_constrained_mcp_tools, run_constrained_tool_node
from utils.agent_runtime import escape_prompt_template_text, run_react_mcp_task, soul_path_for
from utils.schemas import FlightTransportExecutionResult
from utils.llm_factory import build_qwen_chat
from utils.settings import get_settings

logger = logging.getLogger(__name__)
STRUCTURED_SYNTHESIS_TIMEOUT_SECONDS: float | None = None
FLIGHT_ALLOWED_TOOLS = {"google_flights", "google_maps_directions", "geocode", "maps_directions", "amap_direction_driving"}
MAX_TOOL_STEPS = 3


class FlightState(TypedDict, total=False):
    payload: dict[str, Any]
    origin_city: str
    destination: str
    profile: dict[str, Any]
    daily_plan: list[dict[str, Any]]
    research_notes: str
    result: dict[str, Any]


class FlightTransportBureau:
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
        worker_input = normalize_worker_input(state["payload"], "FLIGHT_TRANSPORT")
        return {
            "worker_input": worker_input,
            "messages": [HumanMessage(content=self._research_prompt(worker_input))],
            "tool_evidence": [],
            "tool_step_count": 0,
            "validation_findings": [],
            "warnings": [],
        }

    async def agent_reasoning(self, state: LiubuWorkerState) -> dict[str, Any]:
        injected_reasoner = getattr(self, "_agent_reasoning", None)
        if injected_reasoner is not None:
            update = await injected_reasoner(state)
            if "tool_requests" in update:
                return {"messages": [AIMessage(content="", tool_calls=[self._tool_request_to_call(item, index) for index, item in enumerate(update["tool_requests"])])]}
            return update
        if state.get("tool_evidence"):
            return {"messages": [AIMessage(content="Flight transport tool evidence collected.")]}
        worker_input = state["worker_input"]
        tools = await load_constrained_mcp_tools(worker_input, ["amap", "serpapi"], FLIGHT_ALLOWED_TOOLS)
        llm = build_qwen_chat()
        if llm and tools:
            response = await llm.bind_tools(tools).ainvoke(state.get("messages", []))
            return {"messages": [response]}
        constraints = worker_input.constraints
        if constraints.get("origin_airport_code") and constraints.get("destination_airport_code"):
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "google_flights",
                                "args": {
                                    "departure_id": constraints["origin_airport_code"],
                                    "arrival_id": constraints["destination_airport_code"],
                                    "outbound_date": constraints["start_date"],
                                    "adults": constraints["adults"],
                                    "currency": constraints["currency"],
                                },
                                "id": "flight_transport_call_1",
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            }
        return {"messages": [AIMessage(content="No valid flight tool call could be formed from trip constraints.")]}

    async def tool_execution(self, state: LiubuWorkerState) -> dict[str, Any]:
        worker_input = state["worker_input"]
        existing = [LiubuToolEvidence.model_validate(item) for item in state.get("tool_evidence", [])]
        tools = await load_constrained_mcp_tools(worker_input, ["amap", "serpapi"], FLIGHT_ALLOWED_TOOLS)
        tool_messages, evidence = await run_constrained_tool_node(messages=state.get("messages", []), tools=tools)
        existing.extend(evidence)
        return {
            "messages": tool_messages,
            "tool_evidence": [item.model_dump(mode="json") for item in existing],
            "tool_step_count": int(state.get("tool_step_count") or 0) + 1,
        }

    async def structured_result(self, state: LiubuWorkerState) -> dict[str, Any]:
        worker_input = state["worker_input"]
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
            payload["data_source"] = "live"
        payload["liubu_evidence"] = [item.model_dump(mode="json") for item in evidence]
        return {"result": payload}

    async def quality_gate(self, state: LiubuWorkerState) -> dict[str, Any]:
        evidence = [LiubuToolEvidence.model_validate(item) for item in state.get("tool_evidence", [])]
        findings = gate_flight_transport_result(state["worker_input"], state["result"], evidence)
        passed = result_passed_gate(findings)
        result = dict(state["result"])
        result["liubu_quality"] = {"passed": passed, "findings": [item.model_dump(mode="json") for item in findings]}
        if not passed:
            result["status"] = "fallback"
            result["data_source"] = "fallback_estimate"
            result.setdefault("transport_notes", [])
            result["transport_notes"].extend(item.message for item in findings)
        return {"result": result, "validation_findings": [item.model_dump(mode="json") for item in findings]}

    def _route_after_reasoning(self, state: LiubuWorkerState) -> str:
        if int(state.get("tool_step_count") or 0) >= MAX_TOOL_STEPS:
            return "structured_result"
        return "tool_execution" if tools_condition(state) == "tools" else "structured_result"

    def _route_after_tools(self, state: LiubuWorkerState) -> str:
        return "structured_result" if int(state.get("tool_step_count") or 0) >= MAX_TOOL_STEPS else "agent_reasoning"

    def _research_prompt(self, worker_input) -> str:
        return (
            f"Research flight and transfer options from {worker_input.profile.get('origin_city') or 'Unknown origin'} "
            f"to {worker_input.destination}. Use only the bound tools when live data is needed. "
            f"Hard constraints: {worker_input.constraints}. Return concise evidence before final synthesis."
        )

    def _tool_request_to_call(self, request: dict[str, Any], index: int) -> dict[str, Any]:
        return {
            "name": str(request.get("tool") or ""),
            "args": dict(request.get("args") or {}),
            "id": f"legacy_flight_tool_call_{index + 1}",
            "type": "tool_call",
        }

    async def ingest(self, state: FlightState) -> dict[str, Any]:
        payload = state["payload"]
        approved_draft = payload.get("approved_draft", {})
        execution_plan = payload.get("execution_plan", {})
        user_request = execution_plan.get("user_request", {})
        draft = approved_draft.get("itinerary_draft", {})
        profile = user_request.get("profile", {})
        return {
            "origin_city": profile.get("origin_city") or "Origin TBD",
            "destination": approved_draft.get("destination") or draft.get("destination") or "Destination TBD",
            "profile": profile,
            "daily_plan": draft.get("daily_plan", []),
        }

    async def research_transport(self, state: FlightState) -> dict[str, Any]:
        notes = await run_react_mcp_task(
            soul_path=self.soul_path,
            server_names=["amap", "serpapi"],
            user_task=(
                f"Research flight and transfer options from {state['origin_city']} to {state['destination']}. "
                f"Traveler profile: {state.get('profile', {})}. "
                "Use google_flights, google_maps_directions, and geocode tools when available. Return concise notes about best flight candidates, airport choice, transfer friction, and booking caveats."
            ),
        )
        return {"research_notes": notes}

    async def synthesize_transport(self, state: FlightState) -> dict[str, Any]:
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
        route_query = quote_plus(f"{origin_city} to {destination} flights")
        fallback_note = "Estimated fallback; did not use real-time data. Confirm carrier, routing, fare, and timing before booking."
        research_note = state.get("research_notes") or "MCP or LLM unavailable."
        departure_date = profile.get("start_date") or self._first_trip_date(state.get("daily_plan", []))
        options = [
            {"airline": "Estimated direct-flight option", "price": 320.0, "currency": profile.get("currency", "USD"), "departure_airport": origin_city, "arrival_airport": destination, "departure_time": f"{departure_date} 08:30", "arrival_time": f"{departure_date} 12:15", "duration_minutes": 225, "booking_link": f"https://www.google.com/travel/flights?q={route_query}", "notes": f"{fallback_note} {research_note}"},
            {"airline": "Estimated connection option", "price": 255.0, "currency": profile.get("currency", "USD"), "departure_airport": origin_city, "arrival_airport": destination, "departure_time": f"{departure_date} 10:20", "arrival_time": f"{departure_date} 15:50", "duration_minutes": 330, "booking_link": f"https://www.skyscanner.com/transport/flights/{route_query}", "notes": f"{fallback_note} {research_note}"},
        ]
        return {"result": FlightTransportExecutionResult(origin=origin_city, destination=destination, flight_options=options, transport_notes=[fallback_note, research_note, failure_note], booking_links=[item["booking_link"] for item in options]).model_dump(mode="json")}

    def _first_trip_date(self, daily_plan: list[dict[str, Any]]) -> str:
        for day in daily_plan:
            if day.get("date"):
                return str(day["date"])
        return "unknown-date"
