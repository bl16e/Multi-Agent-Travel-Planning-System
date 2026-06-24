from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from math import ceil
from typing import Any, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from utils.agent_runtime import escape_prompt_template_text, run_react_mcp_task, soul_path_for
from utils.schemas import BudgetExecutionResult
from utils.llm_factory import build_qwen_chat
from utils.settings import get_settings

logger = logging.getLogger(__name__)
STRUCTURED_SYNTHESIS_TIMEOUT_SECONDS: float | None = None


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
                    ("system", escape_prompt_template_text(Path(self.soul_path).read_text(encoding="utf-8"))),
                    ("user", "Draft: {draft}\nProfile: {profile}\nResearch notes: {research_notes}\nReturn valid JSON structured budget output."),
                ])
                result = await asyncio.wait_for(
                    (prompt | structured).ainvoke({"draft": str(state.get("draft", {})), "profile": str(state.get("profile", {})), "research_notes": state.get("research_notes", "")}),
                    timeout=STRUCTURED_SYNTHESIS_TIMEOUT_SECONDS or get_settings().qwen_timeout_seconds,
                )
                data = result.model_dump(mode="json")
                data.update({"status": "ok", "data_source": "structured_llm"})
                return {"result": data}
            except asyncio.TimeoutError:
                logger.warning("Budget structured synthesis timed out; using fallback result.")
                failure_warning = "Budget structured synthesis timed out."
            except Exception as exc:
                logger.warning("Budget structured synthesis failed; using fallback result: %s", exc)
                failure_warning = f"Budget structured synthesis failed: {exc}"
        else:
            failure_warning = "Budget structured synthesis unavailable; using fallback estimate."
        profile = state.get("profile", {})
        draft = state.get("draft", {})
        currency = profile.get("currency", "USD")
        total_budget = profile.get("total_budget")
        adults = max(int(profile.get("adults") or 1), 1)
        budget_level = str(profile.get("budget_level") or "mid_range")
        daily_plan = draft.get("daily_plan", [])
        day_count = max(len(daily_plan), 1)
        nights = max(day_count - 1, 1)
        room_count = max(ceil(adults / 2), 1)
        rate_multiplier = {"budget": 0.7, "mid_range": 1.0, "luxury": 2.0}.get(budget_level, 1.0)
        activity_total = sum(float(activity.get("estimated_cost") or 0) for day in draft.get("daily_plan", []) for activity in day.get("activities", []))
        fallback_activity_total = activity_total or day_count * adults * 40 * rate_multiplier
        accommodation_total = nights * room_count * 140 * rate_multiplier
        food_total = day_count * adults * 60 * rate_multiplier
        transport_total = day_count * adults * 25 * rate_multiplier
        flights_total = adults * 320 if profile.get("origin_city") else 0
        subtotal = fallback_activity_total + accommodation_total + food_total + transport_total + flights_total
        line_items = [
            {"category": "activities", "item": "Planned activity blocks", "estimated_cost": round(fallback_activity_total, 2), "currency": currency, "notes": "Estimated fallback from itinerary activity costs or per-day assumptions."},
            {"category": "accommodation", "item": f"{nights} night(s), {room_count} room(s)", "estimated_cost": round(accommodation_total, 2), "currency": currency, "notes": "Estimated fallback; did not use real-time hotel rates."},
            {"category": "food", "item": f"Meals for {adults} traveler(s)", "estimated_cost": round(food_total, 2), "currency": currency, "notes": "Estimated fallback food allowance."},
            {"category": "transport", "item": "Local transit and transfers", "estimated_cost": round(transport_total, 2), "currency": currency, "notes": "Estimated fallback local transport allowance."},
            {"category": "flights", "item": "Origin-destination transport allowance", "estimated_cost": round(flights_total, 2), "currency": currency, "notes": "Estimated fallback airfare allowance; confirm live fares before booking."},
            {"category": "misc", "item": "Buffer and incidentals", "estimated_cost": round(max(subtotal * 0.15, 50), 2), "currency": currency, "notes": state.get("research_notes", "Fallback budget synthesis.")},
        ]
        total = round(sum(item["estimated_cost"] for item in line_items), 2)
        warnings = ["Budget output fell back because MCP or structured synthesis failed.", "Estimated fallback; did not use real-time data.", failure_warning]
        if total_budget is not None and total > float(total_budget):
            warnings.append("Estimated trip cost exceeds the declared budget cap.")
        return {"result": BudgetExecutionResult(currency=currency, budget_breakdown=line_items, total_estimated_cost=total, warnings=warnings).model_dump(mode="json")}
