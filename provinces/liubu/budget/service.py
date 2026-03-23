from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from utils.agent_runtime import run_react_mcp_task, soul_path_for
from utils.schemas import BudgetExecutionResult
from utils.llm_factory import build_qwen_chat


class BudgetState(TypedDict, total=False):
    payload: dict[str, Any]
    draft: dict[str, Any]
    profile: dict[str, Any]
    research_notes: str
    result: dict[str, Any]


class BudgetBureau:
    def __init__(self) -> None:
        self.soul_path = soul_path_for(__file__)
        self.graph = self._build_graph()

    async def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = await self.graph.ainvoke({"payload": payload})
        return result["result"]

    def _build_graph(self):
        graph = StateGraph(BudgetState)
        graph.add_node("ingest", self.ingest)
        graph.add_node("research_budget", self.research_budget)
        graph.add_node("synthesize_budget", self.synthesize_budget)
        graph.set_entry_point("ingest")
        graph.add_edge("ingest", "research_budget")
        graph.add_edge("research_budget", "synthesize_budget")
        graph.add_edge("synthesize_budget", END)
        return graph.compile()

    async def ingest(self, state: BudgetState) -> dict[str, Any]:
        payload = state["payload"]
        approved_draft = payload.get("approved_draft", {})
        execution_plan = payload.get("execution_plan", {})
        user_request = execution_plan.get("user_request", {})
        return {"draft": approved_draft.get("itinerary_draft", {}), "profile": user_request.get("profile", {})}

    async def research_budget(self, state: BudgetState) -> dict[str, Any]:
        destination = state["draft"].get("destination") or "destination"
        notes = await run_react_mcp_task(
            soul_path=self.soul_path,
            server_names=["serpapi"],
            user_task=(
                f"Research practical trip cost context for {destination}. "
                f"Traveler profile: {state.get('profile', {})}. "
                f"Draft itinerary: {state.get('draft', {})}. "
                "Use MCP tools when available and return concise notes about daily costs, accommodation budget level, and transport cost pressure."
            ),
        )
        return {"research_notes": notes}

    async def synthesize_budget(self, state: BudgetState) -> dict[str, Any]:
        llm = build_qwen_chat()
        if llm:
            try:
                structured = llm.with_structured_output(BudgetExecutionResult)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", Path(self.soul_path).read_text(encoding="utf-8")),
                    ("user", "Draft: {draft}\nProfile: {profile}\nResearch notes: {research_notes}\nReturn a structured budget output."),
                ])
                result = await (prompt | structured).ainvoke({"draft": str(state.get("draft", {})), "profile": str(state.get("profile", {})), "research_notes": state.get("research_notes", "")})
                return {"result": result.model_dump(mode="json")}
            except Exception:
                pass
        profile = state.get("profile", {})
        draft = state.get("draft", {})
        currency = profile.get("currency", "USD")
        total_budget = profile.get("total_budget")
        activity_total = sum(float(activity.get("estimated_cost") or 0) for day in draft.get("daily_plan", []) for activity in day.get("activities", []))
        line_items = [
            {"category": "activities", "item": "Planned activity blocks", "estimated_cost": round(activity_total, 2), "currency": currency, "notes": "Fallback estimate from itinerary."},
            {"category": "misc", "item": "Buffer and incidentals", "estimated_cost": round(max(activity_total * 0.2, 50), 2), "currency": currency, "notes": state.get("research_notes", "Fallback budget synthesis.")},
        ]
        total = round(sum(item["estimated_cost"] for item in line_items), 2)
        warnings = ["Budget output fell back because MCP or structured synthesis failed."]
        if total_budget is not None and total > float(total_budget):
            warnings.append("Estimated trip cost exceeds the declared budget cap.")
        return {"result": BudgetExecutionResult(currency=currency, budget_breakdown=line_items, total_estimated_cost=total, warnings=warnings).model_dump(mode="json")}
