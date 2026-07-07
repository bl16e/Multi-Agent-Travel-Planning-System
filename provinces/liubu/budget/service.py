from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from math import ceil
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from provinces.liubu.constrained.state import LiubuWorkerInput
from provinces.liubu.official_tooling import EvidenceToolNode, bind_tools_if_available, invoke_bound_tool_model, load_allowed_liubu_tools, run_tool_node_collect_evidence
from utils.agent_runtime import escape_prompt_template_text, soul_path_for
from utils.schemas import BudgetExecutionResult
from utils.llm_factory import build_qwen_chat
from utils.settings import get_settings

logger = logging.getLogger(__name__)
STRUCTURED_SYNTHESIS_TIMEOUT_SECONDS: float | None = None
BUDGET_ALLOWED_TOOLS = {"search_google_travel", "search_google_hotels", "search_google_flights"}
BUDGET_TOOL_SERVERS = ["serpapi"]


class BudgetState(TypedDict, total=False):
    worker_input: LiubuWorkerInput | dict[str, Any]
    messages: Annotated[list[Any], add_messages]
    draft: dict[str, Any]
    profile: dict[str, Any]
    research_notes: str
    tool_evidence: list[dict[str, Any]]
    tool_step_count: int
    result: dict[str, Any]


class BudgetBureau:
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
        tools = await load_allowed_liubu_tools(BUDGET_TOOL_SERVERS, BUDGET_ALLOWED_TOOLS, agent="BUDGET")
        self.tool_node = EvidenceToolNode(tools, collector=run_tool_node_collect_evidence)
        self.bound_tool_model = bind_tools_if_available(build_qwen_chat(), tools)
        self.graph = self._build_graph()
        self._tooling_ready = True

    def _build_graph(self):
        graph = StateGraph(BudgetState)
        graph.add_node("agent", self.agent)
        graph.add_node("tools", self.tool_node)
        graph.add_node("quality_gate", self.quality_gate)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", self._route_after_agent, {"tools": "tools", "quality_gate": "quality_gate"})
        graph.add_edge("tools", "agent")
        graph.add_edge("quality_gate", END)
        return graph.compile()

    async def agent(self, state: BudgetState) -> dict[str, Any]:
        worker_input = self._worker_input(state)
        draft = dict(worker_input.approved_draft.get("itinerary_draft") or {})
        clean_budget = self._budget_from_live_inputs(draft, worker_input.profile, live_context_available=False)
        if clean_budget is not None:
            return {
                "draft": draft,
                "profile": worker_input.profile,
                "research_notes": "Approved live itinerary inputs available.",
                "tool_evidence": [],
                "result": clean_budget,
            }
        if self.bound_tool_model is not None and int(state.get("tool_step_count") or 0) == 0:
            tool_message = await invoke_bound_tool_model(
                self.bound_tool_model,
                [
                    HumanMessage(
                        content=(
                            "Search live travel cost context. "
                            f"destination={worker_input.destination}; "
                            f"origin={worker_input.profile.get('origin_city')}; "
                            f"start_date={worker_input.constraints.get('start_date')}; "
                            f"end_date={worker_input.constraints.get('end_date')}; "
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
        evidence = list(state.get("tool_evidence", []) or [])
        live_notes = [str(item.get("result")) for item in evidence if item.get("status") == "ok"]
        research_notes = (
            f"No live budget ToolNode evidence was available for {worker_input.destination}; "
            "using structured synthesis or deterministic fallback."
        )
        if live_notes:
            research_notes = "\n".join(live_notes)
        clean_budget = self._budget_from_live_inputs(draft, worker_input.profile, live_context_available=bool(live_notes))
        if clean_budget is not None:
            clean_budget["liubu_evidence"] = evidence
            return {
                "draft": draft,
                "profile": worker_input.profile,
                "research_notes": research_notes,
                "tool_evidence": evidence,
                "result": clean_budget,
            }
        result = await self.synthesize_budget(
            {
                "draft": draft,
                "profile": worker_input.profile,
                "research_notes": research_notes,
            }
        )
        payload = result["result"]
        if live_notes:
            payload["status"] = "ok"
            payload["data_source"] = "live"
        payload["liubu_evidence"] = evidence
        return {
            "draft": draft,
            "profile": worker_input.profile,
            "research_notes": research_notes,
            "tool_evidence": evidence,
            "result": payload,
        }

    async def tools(self, state: BudgetState) -> dict[str, Any]:
        return {
            "tool_evidence": list(state.get("tool_evidence", [])),
            "tool_step_count": int(state.get("tool_step_count") or 0) + 1,
        }

    async def quality_gate(self, state: BudgetState) -> dict[str, Any]:
        return {"result": state["result"]}

    def _route_after_agent(self, state: BudgetState) -> str:
        messages = state.get("messages") or []
        last_message = messages[-1] if messages else None
        return "tools" if getattr(last_message, "tool_calls", None) else "quality_gate"

    def _worker_input(self, state: BudgetState) -> LiubuWorkerInput:
        return LiubuWorkerInput.model_validate(state["worker_input"])

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
        activity_total = sum(float(activity.get("estimated_cost") or 0) for day in draft.get("daily_plan", []) for activity in day.get("activities", []))
        food_daily, transport_daily = _per_diem_rates(currency, budget_level)
        food_total = day_count * adults * food_daily
        transport_total = day_count * adults * transport_daily
        # Hotel & flights = 0 in fallback (no MCP data available)
        accommodation_total = 0.0
        flights_total = 0.0
        subtotal = activity_total + accommodation_total + food_total + transport_total + flights_total
        line_items = [
            {"category": "activities", "item": "Planned activity blocks", "estimated_cost": round(activity_total, 2), "currency": currency, "notes": "From itinerary activity costs; no live MCP data available."},
            {"category": "accommodation", "item": f"{nights} night(s), {room_count} room(s)", "estimated_cost": 0.0, "currency": currency, "notes": "No MCP data available in fallback mode."},
            {"category": "food", "item": f"Meals for {adults} traveler(s)", "estimated_cost": round(food_total, 2), "currency": currency, "notes": f"Per-diem estimate: {food_daily} {currency}/person/day."},
            {"category": "transport", "item": "Local transit and transfers", "estimated_cost": round(transport_total, 2), "currency": currency, "notes": f"Per-diem estimate: {transport_daily} {currency}/person/day."},
            {"category": "flights", "item": "Origin-destination transport", "estimated_cost": 0.0, "currency": currency, "notes": "No MCP data available in fallback mode."},
            {"category": "misc", "item": "Buffer and incidentals", "estimated_cost": round(max(subtotal * 0.15, 50), 2), "currency": currency, "notes": "Fallback contingency estimate."},
        ]
        total = round(sum(item["estimated_cost"] for item in line_items), 2)
        warnings = ["Budget output fell back because MCP or structured synthesis failed.", "Estimated fallback; did not use real-time data.", failure_warning]
        if total_budget is not None and total > float(total_budget):
            warnings.append("Estimated trip cost exceeds the declared budget cap.")
        return {"result": BudgetExecutionResult(currency=currency, budget_breakdown=line_items, total_estimated_cost=total, warnings=warnings).model_dump(mode="json")}

    def _budget_from_live_inputs(self, draft: dict[str, Any], profile: dict[str, Any], *, live_context_available: bool) -> dict[str, Any] | None:
        daily_plan = list(draft.get("daily_plan") or [])
        has_live_activity = any(
            activity.get("map_link") or activity.get("booking_link")
            for day in daily_plan
            for activity in day.get("activities", [])
        )
        if not has_live_activity and not live_context_available:
            return None
        currency = profile.get("currency", "USD")
        adults = max(int(profile.get("adults") or 1), 1)
        day_count = max(len(daily_plan), 1)
        nights = max(day_count - 1, 1)
        room_count = max(ceil(adults / 2), 1)

        # Activity costs: from the draft itinerary (real data, not estimated)
        activity_total = sum(
            float(activity.get("estimated_cost") or 0)
            for day in daily_plan
            for activity in day.get("activities", [])
        )

        # Food + local transport: inherently estimated per diems.
        # These are the ONLY line items the Budget bureau owns — the rest
        # (hotel, flights) are filled by the assemble step with real MCP data.
        budget_level = str(profile.get("budget_level") or "mid_range")
        food_daily, transport_daily = _per_diem_rates(currency, budget_level)
        food_total = day_count * adults * food_daily
        transport_total = day_count * adults * transport_daily

        # Hotel & flights: deliberately set to 0 here — the assemble step
        # replaces them with actual RollingGo / Amap results.
        accommodation_total = 0.0
        intercity_total = 0.0

        subtotal = activity_total + accommodation_total + food_total + transport_total + intercity_total
        misc_total = round(max(subtotal * 0.12, 50), 2)

        line_items = [
            {"category": "activities", "item": "Approved live itinerary activities",
             "estimated_cost": round(activity_total, 2), "currency": currency,
             "notes": "From approved live itinerary; confirm ticket prices before booking."},
            {"category": "accommodation", "item": f"{nights} night(s), {room_count} room(s)",
             "estimated_cost": 0.0, "currency": currency,
             "notes": "PENDING — will be replaced with live hotel rates after accommodation search."},
            {"category": "food", "item": f"Meals for {adults} traveler(s)",
             "estimated_cost": round(food_total, 2), "currency": currency,
             "notes": f"Per-diem estimate: {food_daily} {currency}/person/day ({budget_level})."},
            {"category": "transport", "item": "Local transit and transfers",
             "estimated_cost": round(transport_total, 2), "currency": currency,
             "notes": f"Per-diem estimate: {transport_daily} {currency}/person/day ({budget_level})."},
            {"category": "flights", "item": "Origin-destination transport",
             "estimated_cost": 0.0, "currency": currency,
             "notes": "PENDING — will be replaced with live flight prices after flight search."},
            {"category": "misc", "item": "Buffer and incidentals",
             "estimated_cost": misc_total, "currency": currency,
             "notes": "Contingency for reservations, schedule changes, and incidentals."},
        ]
        total = round(sum(item["estimated_cost"] for item in line_items), 2)
        warnings: list[str] = []
        total_budget = profile.get("total_budget")
        if total_budget is not None and total > float(total_budget):
            warnings.append("Planning allowance exceeds the declared budget cap.")
        return BudgetExecutionResult(
            status="ok",
            data_source="live",
            currency=currency,
            budget_breakdown=line_items,
            total_estimated_cost=total,
            warnings=warnings,
        ).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Per-diem rate table — single source of truth for food + local transport.
# These are *inherently estimates* (no MCP tool can predict meal costs),
# but they're centralized here so every code path uses the same baseline.
# ---------------------------------------------------------------------------


def _per_diem_rates(currency: str, budget_level: str) -> tuple[float, float]:
    """Return (food_daily, transport_daily) in *currency* for *budget_level*.

    The rates are rough daily per-person estimates derived from general travel
    norms.  Hotel and flight costs are NOT handled here — those come from live
    MCP data filled in by the assemble step.
    """
    # Base rates in CNY — domestic China, mid-range
    base = {"food": 100.0, "transport": 50.0}
    multiplier = {"budget": 0.5, "mid_range": 1.0, "luxury": 2.0}.get(budget_level, 1.0)
    food = base["food"] * multiplier
    transport = base["transport"] * multiplier
    if currency != "CNY":
        food = round(food / 7.2, 2)
        transport = round(transport / 7.2, 2)
    return food, transport
