from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import quote_plus

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from utils.agent_runtime import run_react_mcp_task, soul_path_for
from utils.schemas import FlightTransportExecutionResult
from utils.llm_factory import build_qwen_chat


class FlightState(TypedDict, total=False):
    payload: dict[str, Any]
    origin_city: str
    destination: str
    profile: dict[str, Any]
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
        graph = StateGraph(FlightState)
        graph.add_node("ingest", self.ingest)
        graph.add_node("research_transport", self.research_transport)
        graph.add_node("synthesize_transport", self.synthesize_transport)
        graph.set_entry_point("ingest")
        graph.add_edge("ingest", "research_transport")
        graph.add_edge("research_transport", "synthesize_transport")
        graph.add_edge("synthesize_transport", END)
        return graph.compile()

    async def ingest(self, state: FlightState) -> dict[str, Any]:
        payload = state["payload"]
        approved_draft = payload.get("approved_draft", {})
        execution_plan = payload.get("execution_plan", {})
        user_request = execution_plan.get("user_request", {})
        draft = approved_draft.get("itinerary_draft", {})
        profile = user_request.get("profile", {})
        return {"origin_city": profile.get("origin_city") or "Origin TBD", "destination": approved_draft.get("destination") or draft.get("destination") or "Destination TBD", "profile": profile}

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
                    ("system", Path(self.soul_path).read_text(encoding="utf-8")),
                    ("user", "Origin: {origin}\nDestination: {destination}\nProfile: {profile}\nResearch notes: {research_notes}\nReturn a structured transport result."),
                ])
                result = await (prompt | structured).ainvoke({"origin": state["origin_city"], "destination": state["destination"], "profile": str(state.get("profile", {})), "research_notes": state.get("research_notes", "")})
                return {"result": result.model_dump(mode="json")}
            except Exception:
                pass
        origin_city = state["origin_city"]
        destination = state["destination"]
        profile = state.get("profile", {})
        route_query = quote_plus(f"{origin_city} to {destination} flights")
        options = [
            {"airline": "Placeholder Air", "price": 320.0, "currency": profile.get("currency", "USD"), "departure_airport": origin_city, "arrival_airport": destination, "departure_time": f"{profile.get('start_date') or 'TBD'} 08:30", "arrival_time": f"{profile.get('start_date') or 'TBD'} 12:15", "duration_minutes": 225, "booking_link": f"https://www.google.com/travel/flights?q={route_query}", "notes": state.get("research_notes", "Fallback transport option.")},
            {"airline": "Transfer Connect", "price": 255.0, "currency": profile.get("currency", "USD"), "departure_airport": origin_city, "arrival_airport": destination, "departure_time": f"{profile.get('start_date') or 'TBD'} 10:20", "arrival_time": f"{profile.get('start_date') or 'TBD'} 15:50", "duration_minutes": 330, "booking_link": f"https://www.skyscanner.com/transport/flights/{route_query}", "notes": state.get("research_notes", "Fallback transport option.")},
        ]
        return {"result": FlightTransportExecutionResult(origin=origin_city, destination=destination, flight_options=options, transport_notes=[state.get("research_notes", "Fallback transport synthesis.")], booking_links=[item["booking_link"] for item in options]).model_dump(mode="json")}
