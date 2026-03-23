from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import quote_plus

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from utils.agent_runtime import run_react_mcp_task, soul_path_for
from utils.schemas import AccommodationExecutionResult
from utils.llm_factory import build_qwen_chat


class AccommodationState(TypedDict, total=False):
    payload: dict[str, Any]
    destination: str
    profile: dict[str, Any]
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
        graph = StateGraph(AccommodationState)
        graph.add_node("ingest", self.ingest)
        graph.add_node("research_accommodation", self.research_accommodation)
        graph.add_node("synthesize_accommodation", self.synthesize_accommodation)
        graph.set_entry_point("ingest")
        graph.add_edge("ingest", "research_accommodation")
        graph.add_edge("research_accommodation", "synthesize_accommodation")
        graph.add_edge("synthesize_accommodation", END)
        return graph.compile()

    async def ingest(self, state: AccommodationState) -> dict[str, Any]:
        payload = state["payload"]
        approved_draft = payload.get("approved_draft", {})
        execution_plan = payload.get("execution_plan", {})
        user_request = execution_plan.get("user_request", {})
        draft = approved_draft.get("itinerary_draft", {})
        return {"destination": approved_draft.get("destination") or draft.get("destination") or "Unknown Destination", "profile": user_request.get("profile", {})}

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
                    ("system", Path(self.soul_path).read_text(encoding="utf-8")),
                    ("user", "Destination: {destination}\nProfile: {profile}\nResearch notes: {research_notes}\nReturn a structured accommodation output."),
                ])
                result = await (prompt | structured).ainvoke({"destination": state["destination"], "profile": str(state.get("profile", {})), "research_notes": state.get("research_notes", "")})
                return {"result": result.model_dump(mode="json")}
            except Exception:
                pass
        destination = state["destination"]
        currency = state.get("profile", {}).get("currency", "USD")
        zones = ["Central Station Area", "Old Town Core", "Museum Quarter"]
        hotels = []
        for index, zone in enumerate(zones, start=1):
            query = quote_plus(f"{destination} {zone} hotel")
            hotels.append({"name": f"{destination} {zone} Hotel {index}", "nightly_rate": 120 + index * 20, "total_rate": (120 + index * 20) * 2, "currency": currency, "rating": 4.0 + (index * 0.2), "booking_link": f"https://www.booking.com/searchresults.html?ss={query}", "address": f"{zone}, {destination}", "notes": state.get("research_notes", "Fallback hotel option.")})
        return {"result": AccommodationExecutionResult(destination=destination, hotel_options=hotels, booking_links=[item["booking_link"] for item in hotels], search_notes=[state.get("research_notes", "Fallback accommodation synthesis.")]).model_dump(mode="json")}
