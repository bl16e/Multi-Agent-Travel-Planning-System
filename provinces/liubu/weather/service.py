from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from utils.agent_runtime import run_react_mcp_task, soul_path_for
from utils.schemas import WeatherExecutionResult
from utils.llm_factory import build_qwen_chat


class WeatherState(TypedDict, total=False):
    payload: dict[str, Any]
    destination: str
    daily_plan: list[dict[str, Any]]
    research_notes: str
    result: dict[str, Any]


class WeatherBureau:
    def __init__(self) -> None:
        self.soul_path = soul_path_for(__file__)
        self.graph = self._build_graph()

    async def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = await self.graph.ainvoke({"payload": payload})
        return result["result"]

    def _build_graph(self):
        graph = StateGraph(WeatherState)
        graph.add_node("ingest", self.ingest)
        graph.add_node("research_weather", self.research_weather)
        graph.add_node("synthesize_weather", self.synthesize_weather)
        graph.set_entry_point("ingest")
        graph.add_edge("ingest", "research_weather")
        graph.add_edge("research_weather", "synthesize_weather")
        graph.add_edge("synthesize_weather", END)
        return graph.compile()

    async def ingest(self, state: WeatherState) -> dict[str, Any]:
        payload = state["payload"]
        approved_draft = payload.get("approved_draft", {})
        draft = approved_draft.get("itinerary_draft", {})
        return {"destination": approved_draft.get("destination") or draft.get("destination") or "Unknown Destination", "daily_plan": draft.get("daily_plan", [])}

    async def research_weather(self, state: WeatherState) -> dict[str, Any]:
        notes = await run_react_mcp_task(
            soul_path=self.soul_path,
            server_names=["amap"],
            user_task=(
                f"Research weather guidance for destination {state['destination']}. "
                f"Trip dates: {[item.get('date') for item in state.get('daily_plan', [])]}. "
                "Use MCP tools when available and return concise planning notes about weather, temperature, rain risk, and packing impact."
            ),
        )
        return {"research_notes": notes}

    async def synthesize_weather(self, state: WeatherState) -> dict[str, Any]:
        llm = build_qwen_chat()
        if llm:
            try:
                structured = llm.with_structured_output(WeatherExecutionResult)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", Path(self.soul_path).read_text(encoding="utf-8")),
                    ("user", "Destination: {destination}\nDaily plan: {daily_plan}\nResearch notes: {research_notes}\nReturn structured weather guidance."),
                ])
                result = await (prompt | structured).ainvoke({"destination": state["destination"], "daily_plan": str(state.get("daily_plan", [])), "research_notes": state.get("research_notes", "")})
                return {"result": result.model_dump(mode="json")}
            except Exception:
                pass
        fallback_date = date.today()
        result = WeatherExecutionResult(destination=state["destination"], forecast_days=[{"date": fallback_date, "condition": "Weather unavailable", "min_temp_c": 18, "max_temp_c": 26, "precipitation_probability": 0.2, "activity_suitability": "Use flexible scheduling.", "clothing_advice": ["Pack light layers."], "warnings": ["MCP research unavailable."], "is_estimated": True}], packing_list=["passport", "phone charger", "comfortable walking shoes", "light layers"], warnings=["Weather output fell back because MCP or structured synthesis failed."], summary=state.get("research_notes", "Fallback weather guidance."))
        return {"result": result.model_dump(mode="json")}
