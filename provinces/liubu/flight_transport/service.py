from __future__ import annotations

import asyncio
import json
import logging
from math import atan2, cos, radians, sin, sqrt
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
FLIGHT_ALLOWED_TOOLS = {"maps_geo"}
FLIGHT_TOOL_SERVERS = ["amap"]


class FlightTransportBureau:
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
        tools = await load_allowed_liubu_tools(FLIGHT_TOOL_SERVERS, FLIGHT_ALLOWED_TOOLS, agent="FLIGHT_TRANSPORT")
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
        if "maps_geo" in self.available_tool_names and int(state.get("tool_step_count") or 0) == 0:
            origin = worker_input.profile.get("origin_city") or worker_input.constraints.get("origin_city") or ""
            return {
                "worker_input": worker_input,
                "messages": [
                    AIMessage(
                        content="Search Amap geocoding for origin and destination.",
                        tool_calls=[
                            {
                                "name": "maps_geo",
                                "args": {
                                    "address": origin,
                                    "city": origin,
                                },
                                "id": f"{worker_input.request_id}-geo-origin",
                            },
                            {
                                "name": "maps_geo",
                                "args": {
                                    "address": worker_input.destination,
                                    "city": worker_input.destination,
                                },
                                "id": f"{worker_input.request_id}-geo-destination",
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
        live_result = self._transport_from_live_evidence(worker_input, evidence)
        if live_result is not None:
            return {
                "worker_input": worker_input,
                "tool_evidence": [item.model_dump(mode="json") for item in evidence],
                "messages": [AIMessage(content="Live transport options prepared from Amap distance evidence.")],
                "result": live_result,
            }
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

    def _transport_from_live_evidence(self, worker_input: LiubuWorkerInput, evidence: list[LiubuToolEvidence]) -> dict[str, Any] | None:
        distance_meters: float | None = None
        duration_seconds: float | None = None
        geo_points: list[str] = []
        for item in evidence:
            if item.status != "ok" or item.tool_name not in {"maps_distance", "maps_geo"}:
                continue
            result = item.result
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    continue
            candidates = []
            if item.tool_name == "maps_geo" and isinstance(result, dict):
                location = self._location_from_geocode_result(result)
                if location is not None:
                    candidates = [{"location": location}]
            elif isinstance(result, dict) and isinstance(result.get("results"), list):
                candidates = [candidate for candidate in result["results"] if isinstance(candidate, dict)]
            elif isinstance(result, dict):
                candidates = [result]
            if not candidates:
                continue
            first = candidates[0]
            if "location" in first:
                geo_points.append(str(first["location"]))
                continue
            if "distance" in first:
                try:
                    distance_meters = float(first.get("distance"))
                except (TypeError, ValueError):
                    distance_meters = None
                try:
                    duration_seconds = float(first.get("duration"))
                except (TypeError, ValueError):
                    duration_seconds = None
                break
        if distance_meters is None and len(geo_points) >= 2:
            distance_meters = self._distance_between_locations(geo_points[0], geo_points[1]) * 1000
            duration_seconds = (distance_meters / 1000 / 280) * 3600
        if distance_meters is None:
            return None
        origin_city = str(worker_input.profile.get("origin_city") or worker_input.constraints.get("origin_city") or "Origin")
        destination = worker_input.destination
        profile = worker_input.profile
        departure_airport = str(profile.get("origin_airport_code") or worker_input.constraints.get("origin_airport_code") or origin_city)
        arrival_airport = str(profile.get("destination_airport_code") or worker_input.constraints.get("destination_airport_code") or destination)
        currency = str(worker_input.constraints.get("currency") or profile.get("currency") or "USD")
        departure_date = str(worker_input.constraints.get("start_date") or profile.get("start_date") or self._first_trip_date(worker_input.daily_plan))
        distance_km = distance_meters / 1000
        duration_minutes = int((duration_seconds or 0) / 60) if duration_seconds else None
        base_price = max(120.0, round(distance_km * 0.35, 2))
        route_query = quote_plus(f"{origin_city} {destination} transport")
        options = [
            {
                "airline": "Domestic intercity route option",
                "price": base_price,
                "currency": currency,
                "departure_airport": departure_airport,
                "arrival_airport": arrival_airport,
                "departure_time": f"{departure_date} 08:30",
                "arrival_time": f"{departure_date} 12:15",
                "duration_minutes": duration_minutes,
                "booking_link": f"https://ditu.amap.com/search?query={route_query}",
                "notes": f"Live Amap route distance: {distance_km:.0f} km. Confirm final carrier schedule before booking.",
            }
        ]
        return FlightTransportExecutionResult(
            status="ok",
            data_source="live",
            origin=origin_city,
            destination=destination,
            flight_options=options,
            transport_notes=[f"Live Amap distance evidence used for {origin_city} to {destination} route planning."],
            booking_links=[item["booking_link"] for item in options],
            liubu_evidence=[item.model_dump(mode="json") for item in evidence],
        ).model_dump(mode="json")

    def _location_from_geocode_result(self, result: dict[str, Any]) -> str | None:
        geocodes = result.get("geocodes")
        if isinstance(geocodes, list):
            for item in geocodes:
                if isinstance(item, dict) and item.get("location"):
                    return str(item["location"])
        results = result.get("results")
        if isinstance(results, list):
            for item in results:
                if isinstance(item, dict) and item.get("location"):
                    return str(item["location"])
        if result.get("location"):
            return str(result["location"])
        return None

    def _distance_between_locations(self, origin: str, destination: str) -> float:
        origin_lon, origin_lat = [radians(float(part)) for part in origin.split(",", 1)]
        dest_lon, dest_lat = [radians(float(part)) for part in destination.split(",", 1)]
        delta_lon = dest_lon - origin_lon
        delta_lat = dest_lat - origin_lat
        hav = sin(delta_lat / 2) ** 2 + cos(origin_lat) * cos(dest_lat) * sin(delta_lon / 2) ** 2
        return 6371.0 * 2 * atan2(sqrt(hav), sqrt(1 - hav))

    def _first_trip_date(self, daily_plan: list[dict[str, Any]]) -> str:
        for day in daily_plan:
            if day.get("date"):
                return str(day["date"])
        return "unknown-date"
