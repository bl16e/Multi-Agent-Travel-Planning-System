from __future__ import annotations

import asyncio
import json
import logging
from datetime import date
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from provinces.liubu.constrained.state import LiubuWorkerInput
from provinces.liubu.official_tooling import EvidenceToolNode, bind_tools_if_available, invoke_bound_tool_model, load_allowed_liubu_tools, run_tool_node_collect_evidence
from utils.agent_runtime import escape_prompt_template_text, soul_path_for
from utils.schemas import WeatherExecutionResult
from utils.llm_factory import build_qwen_chat
from utils.settings import get_settings

logger = logging.getLogger(__name__)
STRUCTURED_SYNTHESIS_TIMEOUT_SECONDS: float | None = None
WEATHER_ALLOWED_TOOLS = {"maps_weather"}
WEATHER_TOOL_SERVERS = ["amap"]


class WeatherState(TypedDict, total=False):
    worker_input: LiubuWorkerInput | dict[str, Any]
    messages: Annotated[list[Any], add_messages]
    destination: str
    daily_plan: list[dict[str, Any]]
    research_notes: str
    tool_evidence: list[dict[str, Any]]
    tool_step_count: int
    result: dict[str, Any]


class WeatherBureau:
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
        tools = await load_allowed_liubu_tools(WEATHER_TOOL_SERVERS, WEATHER_ALLOWED_TOOLS, agent="WEATHER")
        self.available_tool_names = {str(getattr(tool, "name", "")) for tool in tools}
        self.tool_node = EvidenceToolNode(tools, collector=run_tool_node_collect_evidence)
        self.bound_tool_model = bind_tools_if_available(build_qwen_chat(), tools)
        self.graph = self._build_graph()
        self._tooling_ready = True

    def _build_graph(self):
        graph = StateGraph(WeatherState)
        graph.add_node("agent", self.agent)
        graph.add_node("tools", self.tool_node)
        graph.add_node("quality_gate", self.quality_gate)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", self._route_after_agent, {"tools": "tools", "quality_gate": "quality_gate"})
        graph.add_edge("tools", "agent")
        graph.add_edge("quality_gate", END)
        return graph.compile()

    async def agent(self, state: WeatherState) -> dict[str, Any]:
        worker_input = self._worker_input(state)
        if "maps_weather" in self.available_tool_names and int(state.get("tool_step_count") or 0) == 0:
            from langchain_core.messages import AIMessage

            return {
                "worker_input": worker_input,
                "messages": [
                    AIMessage(
                        content="Search Amap weather.",
                        tool_calls=[
                            {
                                "name": "maps_weather",
                                "args": {"city": worker_input.destination},
                                "id": f"{worker_input.request_id}-weather-1",
                            }
                        ],
                    )
                ],
            }
        if self.bound_tool_model is not None and int(state.get("tool_step_count") or 0) == 0:
            tool_message = await invoke_bound_tool_model(
                self.bound_tool_model,
                [HumanMessage(content=f"Search live weather and local context for {worker_input.destination}.")],
                bureau=worker_input.bureau,
                request_id=worker_input.request_id,
                destination=worker_input.destination,
            )
            if getattr(tool_message, "tool_calls", None):
                return {"worker_input": worker_input, "messages": [tool_message]}
        evidence = list(state.get("tool_evidence", []) or [])
        live_notes = [str(item.get("result")) for item in evidence if item.get("status") == "ok"]
        research_notes = (
            f"No live weather ToolNode evidence was available for {worker_input.destination}; "
            "using structured synthesis or deterministic fallback."
        )
        if live_notes:
            research_notes = "\n".join(live_notes)
        live_weather = self._weather_from_live_evidence(worker_input, evidence)
        if live_weather is not None:
            payload = live_weather
        else:
            result = await self.synthesize_weather(
                {
                    "destination": worker_input.destination,
                    "daily_plan": worker_input.daily_plan,
                    "research_notes": research_notes,
                }
            )
            payload = result["result"]
            if live_notes:
                payload["status"] = "ok"
                payload["data_source"] = "live"
        payload["liubu_evidence"] = evidence
        return {
            "destination": worker_input.destination,
            "daily_plan": worker_input.daily_plan,
            "research_notes": research_notes,
            "tool_evidence": evidence,
            "result": payload,
        }

    async def tools(self, state: WeatherState) -> dict[str, Any]:
        return {
            "tool_evidence": list(state.get("tool_evidence", [])),
            "tool_step_count": int(state.get("tool_step_count") or 0) + 1,
        }

    async def quality_gate(self, state: WeatherState) -> dict[str, Any]:
        return {"result": state["result"]}

    def _route_after_agent(self, state: WeatherState) -> str:
        messages = state.get("messages") or []
        last_message = messages[-1] if messages else None
        return "tools" if getattr(last_message, "tool_calls", None) else "quality_gate"

    def _worker_input(self, state: WeatherState) -> LiubuWorkerInput:
        return LiubuWorkerInput.model_validate(state["worker_input"])

    async def synthesize_weather(self, state: WeatherState) -> dict[str, Any]:
        llm = build_qwen_chat()
        if llm:
            try:
                structured = llm.with_structured_output(WeatherExecutionResult)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", escape_prompt_template_text(Path(self.soul_path).read_text(encoding="utf-8"))),
                    ("user", "Destination: {destination}\nDaily plan: {daily_plan}\nResearch notes: {research_notes}\nReturn valid JSON structured weather guidance."),
                ])
                result = await asyncio.wait_for(
                    (prompt | structured).ainvoke({"destination": state["destination"], "daily_plan": str(state.get("daily_plan", [])), "research_notes": state.get("research_notes", "")}),
                    timeout=STRUCTURED_SYNTHESIS_TIMEOUT_SECONDS or get_settings().qwen_timeout_seconds,
                )
                data = result.model_dump(mode="json")
                data.update({"status": "ok", "data_source": "structured_llm"})
                return {"result": data}
            except asyncio.TimeoutError:
                logger.warning("Weather structured synthesis timed out; using fallback result.")
                failure_warning = "Weather structured synthesis timed out."
            except Exception as exc:
                logger.warning("Weather structured synthesis failed; using fallback result: %s", exc)
                failure_warning = f"Weather structured synthesis failed: {exc}"
        else:
            failure_warning = "Weather structured synthesis unavailable; using fallback estimate."
        forecast_dates = self._forecast_dates(state.get("daily_plan", []))
        result = WeatherExecutionResult(
            destination=state["destination"],
            forecast_days=[
                {
                    "date": forecast_date,
                    "condition": "Weather unavailable",
                    "min_temp_c": 18,
                    "max_temp_c": 26,
                    "precipitation_probability": 0.2,
                    "activity_suitability": "Use flexible scheduling.",
                    "clothing_advice": ["Pack light layers."],
                    "warnings": ["MCP research unavailable."],
                    "is_estimated": True,
                }
                for forecast_date in forecast_dates
            ],
            packing_list=["passport", "phone charger", "comfortable walking shoes", "light layers"],
            warnings=["Weather output fell back because MCP or structured synthesis failed.", failure_warning],
            summary=state.get("research_notes", "Fallback weather guidance."),
        )
        return {"result": result.model_dump(mode="json")}

    def _weather_from_live_evidence(self, worker_input: LiubuWorkerInput, evidence: list[dict[str, Any]]) -> dict[str, Any] | None:
        forecast_dates = self._forecast_dates(worker_input.daily_plan)
        for item in evidence:
            if item.get("status") != "ok" or item.get("tool_name") != "maps_weather":
                continue
            result = item.get("result")
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    continue
            if not isinstance(result, dict):
                continue
            forecasts = result.get("forecasts")
            if not isinstance(forecasts, list):
                continue
            casts: list[dict[str, Any]] = []
            for forecast in forecasts:
                if not isinstance(forecast, dict):
                    continue
                if isinstance(forecast.get("casts"), list):
                    casts.extend(cast for cast in forecast["casts"] if isinstance(cast, dict))
                elif forecast.get("date"):
                    casts.append(forecast)
            days: list[dict[str, Any]] = []
            for forecast_date in forecast_dates:
                cast = next((candidate for candidate in casts if str(candidate.get("date")) == forecast_date.isoformat()), None)
                if cast is None and casts:
                    cast = casts[min(len(days), len(casts) - 1)]
                if cast is None:
                    continue
                min_temp = self._float_or_default(cast.get("nighttemp"), 18.0)
                max_temp = self._float_or_default(cast.get("daytemp"), min_temp)
                days.append(
                    {
                        "date": forecast_date,
                        "condition": str(cast.get("dayweather") or cast.get("nightweather") or "Weather available"),
                        "min_temp_c": min_temp,
                        "max_temp_c": max_temp,
                        "precipitation_probability": 0.0,
                        "activity_suitability": "Use the live Amap forecast to schedule outdoor blocks flexibly.",
                        "clothing_advice": ["Pack layers suitable for the live forecast."],
                        "warnings": [],
                        "is_estimated": False,
                    }
                )
            if days:
                return WeatherExecutionResult(
                    status="ok",
                    data_source="live",
                    destination=worker_input.destination,
                    forecast_days=days,
                    packing_list=["phone charger", "comfortable walking shoes", "weather-appropriate layers"],
                    warnings=[],
                    summary="Live Amap weather forecast was used for the trip dates.",
                ).model_dump(mode="json")
        return None

    def _float_or_default(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _forecast_dates(self, daily_plan: list[dict[str, Any]]) -> list[date]:
        dates: list[date] = []
        for day in daily_plan:
            value = day.get("date") if isinstance(day, dict) else None
            parsed = self._parse_date(value)
            if parsed is not None and parsed not in dates:
                dates.append(parsed)
        return dates or [date.today()]

    def _parse_date(self, value: Any) -> date | None:
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError:
                return None
        return None
