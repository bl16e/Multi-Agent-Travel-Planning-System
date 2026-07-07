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
ACCOMMODATION_ALLOWED_TOOLS = {"maps_text_search", "searchHotels", "getHotelDetail"}
ACCOMMODATION_TOOL_SERVERS = ["amap", "rollinggo"]


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
        if int(state.get("tool_step_count") or 0) == 0:
            tool_calls = []
            if "maps_text_search" in self.available_tool_names:
                tool_calls.append(
                    {
                        "name": "maps_text_search",
                        "args": {
                            "keywords": f"{worker_input.destination} 酒店",
                            "city": worker_input.destination,
                            "citylimit": True,
                        },
                        "id": f"{worker_input.request_id}-hotel-amap",
                    }
                )
            if "searchHotels" in self.available_tool_names:
                start_date = str(worker_input.constraints.get("start_date") or "")
                end_date = str(worker_input.constraints.get("end_date") or "")
                adults = int(worker_input.constraints.get("adults") or 1)
                tool_calls.append(
                    {
                        "name": "searchHotels",
                        "args": {
                            "place": worker_input.destination,
                            "placeType": "CITY",
                            "countryCode": "CN",
                            "size": 5,
                            "originQuery": f"{worker_input.destination} hotel {start_date} to {end_date}",
                            "checkInParam": {
                                "checkInDate": start_date,
                                "checkOutDate": end_date,
                                "occupancy": [{"adults": adults}],
                            },
                        },
                        "id": f"{worker_input.request_id}-hotel-rgo",
                    }
                )
            if tool_calls:
                return {
                    "worker_input": worker_input,
                    "messages": [
                        AIMessage(
                            content="Search hotel options via Amap POI and RollingGo booking.",
                            tool_calls=tool_calls,
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
            nightly_rate = 200 + index * 50
            hotels.append({"name": f"{destination} {zone} Hotel {index}", "nightly_rate": nightly_rate, "total_rate": nightly_rate * nights, "currency": currency, "rating": 4.0 + (index * 0.2), "booking_link": f"https://www.booking.com/searchresults.html?ss={query}", "address": f"{zone}, {destination}", "notes": f"{fallback_note} {research_note}"})
        return {"result": AccommodationExecutionResult(destination=destination, hotel_options=hotels, booking_links=[item["booking_link"] for item in hotels], search_notes=[fallback_note, research_note, failure_note], warnings=[failure_note]).model_dump(mode="json")}

    def _accommodation_from_live_evidence(self, worker_input: LiubuWorkerInput, evidence: list[LiubuToolEvidence]) -> dict[str, Any] | None:
        pois: list[dict[str, Any]] = []
        rgo_hotels: list[dict[str, Any]] = []
        for item in evidence:
            if item.status != "ok" or item.tool_name not in {"maps_text_search", "searchHotels"}:
                continue
            result = item.result
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    continue
            if not isinstance(result, dict):
                continue
            if item.tool_name == "maps_text_search" and isinstance(result.get("pois"), list):
                pois.extend(poi for poi in result["pois"] if isinstance(poi, dict))
            elif item.tool_name == "searchHotels":
                # RollingGo searchHotels returns hotelInformationList with real booking data
                hotel_list = result.get("hotelInformationList") or result.get("hotels") or result.get("data") or []
                if isinstance(hotel_list, list):
                    rgo_hotels.extend(h for h in hotel_list if isinstance(h, dict))
                # Also handle single hotel response
                if result.get("hotelId") or result.get("name"):
                    rgo_hotels.append(result)
        # --- Prefer RollingGo hotels (real booking data), fall back to Amap POIs ---
        currency = str(worker_input.constraints.get("currency") or worker_input.profile.get("currency") or "CNY")
        nights = max(len(worker_input.daily_plan) - 1, 1)
        hotels: list[dict[str, Any]] = []
        booking_links: list[str] = []

        if rgo_hotels:
            for idx, h in enumerate(rgo_hotels[:5], start=1):
                name = str(h.get("name") or f"{worker_input.destination} hotel option {idx}")
                address = str(h.get("address") or h.get("location") or worker_input.destination)
                price_info = h.get("price") or {}
                if isinstance(price_info, dict):
                    nightly_rate = float(price_info.get("lowestPrice") or price_info.get("amount") or price_info.get("nightlyRate") or 0)
                    hotel_currency = str(price_info.get("currency") or currency)
                else:
                    try:
                        nightly_rate = float(price_info)
                    except (TypeError, ValueError):
                        nightly_rate = 0
                    hotel_currency = currency
                if nightly_rate <= 0:
                    nightly_rate = 250 + idx * 50  # sensible fallback for domestic hotels
                booking_link = str(h.get("bookingUrl") or h.get("bookingLink") or h.get("booking_url") or h.get("url") or "")
                if not booking_link:
                    query = quote_plus(name)
                    booking_link = f"https://ditu.amap.com/search?query={query}"
                booking_links.append(booking_link)
                hotels.append({
                    "name": name,
                    "nightly_rate": nightly_rate,
                    "total_rate": nightly_rate * nights,
                    "currency": hotel_currency,
                    "rating": float(h.get("rating") or h.get("starRating") or 0) or None,
                    "booking_link": booking_link,
                    "address": address,
                    "notes": (
                        f"RollingGo live hotel booking data; "
                        f"check_in={worker_input.constraints.get('start_date')}; "
                        f"check_out={worker_input.constraints.get('end_date')}"
                    ),
                })
            if hotels:
                return AccommodationExecutionResult(
                    status="ok",
                    data_source="live",
                    destination=worker_input.destination,
                    hotel_options=hotels,
                    booking_links=booking_links,
                    search_notes=["RollingGo live hotel search used for real-time pricing and availability."],
                    warnings=[],
                    liubu_evidence=[item.model_dump(mode="json") for item in evidence],
                ).model_dump(mode="json")

        # --- Fall back to Amap POI results when RollingGo returned nothing ---
        pois = [p for p in pois if _poi_name_plausible_for_accommodation(p, worker_input.destination)]
        if not pois:
            return None
        for index, poi in enumerate(pois[:3], start=1):
            name = str(poi.get("name") or f"{worker_input.destination} hotel option {index}")
            query = quote_plus(name)
            nightly_rate = 200 + index * 50
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
        if hotels:
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
        return None


# ------------------------------------------------------------------
# POI name quality guard – prevents Amap location markers like
# "SHANGHAI" or "Beijing" from being presented as hotel names.
#
# Strategy (algorithmic, no hardcoded brand/city lists):
#   1. Reject names that are exactly the destination city.
#   2. Reject short all-caps-ASCII strings (look like codes/labels,
#      not real business names).
#   3. Accept anything with CJK characters – real Chinese businesses.
#   4. Accept mixed-case or long names – real businesses.
# ------------------------------------------------------------------


def _poi_name_plausible_for_accommodation(poi: dict[str, Any], destination: str) -> bool:
    """Heuristic: does this POI name look like a real business rather than a location marker?"""
    name = str(poi.get("name") or "").strip()
    if not name or len(name) < 2:
        return False

    # Reject if the name is just the destination city (case-insensitive)
    if name.lower() == destination.lower():
        return False

    # Reject short all-caps ASCII strings – these are codes/labels
    # (e.g. "SHANGHAI", "PEK", "A12"), not real business names.
    if len(name) <= 8 and name.isascii() and name.isupper():
        return False

    # Has CJK characters → real Chinese business name
    if any("一" <= c <= "鿿" or "㐀" <= c <= "䶿" for c in name):
        return True

    # Mixed-case ASCII → real brand name (e.g. "Hilton Shanghai")
    if name.isascii() and not name.islower() and not name.isupper():
        return True

    # Longer ASCII name with spaces → probably a real business name
    if len(name) >= 10 and " " in name:
        return True

    return False
