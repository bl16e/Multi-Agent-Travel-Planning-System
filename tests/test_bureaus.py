import asyncio

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

import provinces.liubu.accommodation.service as accommodation_service
import provinces.liubu.budget.service as budget_service
import provinces.liubu.calendar.service as calendar_service
import provinces.liubu.flight_transport.service as flight_service
import provinces.liubu.weather.service as weather_service
import provinces.liubu.official_tooling as official_tooling
from provinces.liubu.accommodation.service import AccommodationBureau
from provinces.liubu.weather.service import WeatherBureau
from provinces.liubu.budget.service import BudgetBureau
from provinces.liubu.calendar.service import CalendarBureau
from provinces.liubu.flight_transport.service import FlightTransportBureau
from provinces.liubu.constrained.state import normalize_worker_input
from utils.schemas import AgentRole
from utils.schemas import CalendarEventListModel
from workflow import ProvinceWorkflow


LIUBU_SUBAGENT_TEMPLATE_NODES = {
    "agent",
    "tools",
    "quality_gate",
}
REMOVED_LIUBU_SUBAGENT_NODES = {
    "prepare_context",
    "agent_reasoning",
    "synthesize_result",
}


class FailingLLM:
    def with_structured_output(self, output_model):
        raise RuntimeError("structured synthesis failed")


class HangingStructuredRunnable:
    async def ainvoke(self, variables):
        await asyncio.sleep(10)
        return {}


class HangingPromptChain:
    def __or__(self, other):
        return HangingStructuredRunnable()


class HangingPromptTemplate:
    @classmethod
    def from_messages(cls, messages):
        return HangingPromptChain()


class HangingLLM:
    def with_structured_output(self, output_model):
        return object()


@pytest.mark.parametrize(
    "bureau",
    [
        WeatherBureau(),
        FlightTransportBureau(),
        BudgetBureau(),
        AccommodationBureau(),
        CalendarBureau(),
    ],
)
def test_liubu_subagents_use_official_toolnode_template(bureau):
    graph = bureau.graph.get_graph()
    node_names = set(graph.nodes)

    assert LIUBU_SUBAGENT_TEMPLATE_NODES <= node_names
    assert REMOVED_LIUBU_SUBAGENT_NODES.isdisjoint(node_names)
    assert isinstance(graph.nodes["tools"].data, ToolNode)


def _liubu_subtask(payload: dict, bureau: str) -> dict:
    return {"worker_input": normalize_worker_input(payload, bureau)}


def _liubu_payload(destination: str = "Tokyo") -> dict:
    return {
        "request_id": "liubu_subtask",
        "approved_draft": {
            "destination": destination,
            "itinerary_draft": {
                "destination": destination,
                "daily_plan": [{"date": "2026-05-01", "activities": []}],
            },
        },
        "execution_plan": {
            "user_request": {
                "profile": {
                    "origin_city": "Beijing",
                    "destination_preferences": [destination],
                    "start_date": "2026-05-01",
                    "end_date": "2026-05-01",
                    "adults": 1,
                    "currency": "USD",
                    "total_budget": 1000,
                }
            }
        },
    }


@pytest.mark.asyncio
async def test_workflow_passes_normalized_subtask_to_liubu_bureau():
    workflow = ProvinceWorkflow()
    context = workflow.orchestrator.bootstrap("liubu_subtask", {"request_id": "liubu_subtask"})
    captured = {}

    async def fake_weather_run(subtask):
        captured.update(subtask)
        return {
            "bureau": "WEATHER",
            "destination": subtask["worker_input"].destination,
            "forecast_days": [],
            "packing_list": [],
            "warnings": [],
            "summary": "captured",
        }

    await workflow._run_liubu_node(
        {"payload": _liubu_payload(), "context": context},
        node_name="liubu_weather",
        role=AgentRole.WEATHER,
        running_message="weather bureau running",
        returned_message="weather bureau returned",
        run_bureau=fake_weather_run,
        fallback_result=workflow._fallback_weather_result,
    )

    assert set(captured) == {"worker_input"}
    assert captured["worker_input"].bureau == "WEATHER"
    assert captured["worker_input"].destination == "Tokyo"


@pytest.mark.asyncio
async def test_flight_transport_loads_allowed_mcp_tools_and_binds_model(monkeypatch):
    calls = {"load": [], "bind": 0}

    @tool
    async def maps_geo(address: str = "Beijing", city: str = "Beijing") -> str:
        """Search fake Amap geocoding results."""
        return f"{address}-{city}"

    class FakeModel:
        def bind_tools(self, tools):
            calls["bind"] += 1
            assert [tool.name for tool in tools] == ["maps_geo"]
            return self

    async def fake_load(server_names, allowed_names):
        calls["load"].append((server_names, allowed_names))
        return [maps_geo]

    monkeypatch.setattr(flight_service, "build_qwen_chat", lambda: FakeModel())
    monkeypatch.setattr(flight_service, "load_allowed_liubu_tools", fake_load)
    bureau = FlightTransportBureau()

    await bureau.ensure_live_tooling()

    assert calls["load"] == [(["amap"], {"maps_geo"})]
    assert calls["bind"] == 1
    assert "maps_geo" in bureau.tool_node.tools_by_name


@pytest.mark.asyncio
async def test_accommodation_loads_allowed_mcp_tools_and_binds_model(monkeypatch):
    calls = {"load": [], "bind": 0}

    @tool
    async def maps_text_search(query: str = "上海 酒店") -> str:
        """Search fake Amap hotel POI results."""
        return query

    class FakeModel:
        def bind_tools(self, tools):
            calls["bind"] += 1
            assert [tool.name for tool in tools] == ["maps_text_search"]
            return self

    async def fake_load(server_names, allowed_names):
        calls["load"].append((server_names, allowed_names))
        return [maps_text_search]

    monkeypatch.setattr(accommodation_service, "build_qwen_chat", lambda: FakeModel())
    monkeypatch.setattr(accommodation_service, "load_allowed_liubu_tools", fake_load)
    bureau = AccommodationBureau()

    await bureau.ensure_live_tooling()

    assert calls["load"] == [(["amap"], {"maps_text_search"})]
    assert calls["bind"] == 1
    assert "maps_text_search" in bureau.tool_node.tools_by_name


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module", "bureau", "expected_tools"),
    [
        (weather_service, WeatherBureau(), {"maps_weather"}),
        (budget_service, BudgetBureau(), {"search_google_travel", "search_google_hotels", "search_google_flights"}),
        (calendar_service, CalendarBureau(), {"search_google_maps", "search_google_maps_directions", "search_local_places"}),
    ],
)
async def test_general_liubu_bureaus_load_allowed_mcp_tools_and_bind_model(module, bureau, expected_tools, monkeypatch):
    calls = {"load": [], "bind": 0}

    fake_tools = []
    for name in sorted(expected_tools):
        async def fake_tool(query: str = "", _name=name) -> str:
            """Search fake live data."""
            return _name or query

        fake_tool.__name__ = name
        fake_tools.append(tool(fake_tool))

    class FakeModel:
        def bind_tools(self, tools):
            calls["bind"] += 1
            assert {item.name for item in tools} == expected_tools
            return self

    async def fake_load(server_names, allowed_names):
        calls["load"].append((server_names, allowed_names))
        return fake_tools

    monkeypatch.setattr(module, "build_qwen_chat", lambda: FakeModel())
    monkeypatch.setattr(module, "load_allowed_liubu_tools", fake_load)

    await bureau.ensure_live_tooling()

    expected_servers = ["amap"] if expected_tools == {"maps_weather"} else ["serpapi"]
    assert calls["load"] == [(expected_servers, expected_tools)]
    assert calls["bind"] == 1
    assert set(bureau.tool_node.tools_by_name) == expected_tools


@pytest.mark.asyncio
async def test_weather_bureau():
    bureau = WeatherBureau()
    result = await bureau.run(_liubu_subtask(_liubu_payload(), "WEATHER"))
    assert result["bureau"] == "WEATHER"
    assert "forecast_days" in result


@pytest.mark.asyncio
async def test_weather_fallback_uses_trip_dates_instead_of_current_date(monkeypatch):
    monkeypatch.setattr(weather_service, "build_qwen_chat", lambda: None)
    bureau = WeatherBureau()

    result = await bureau.synthesize_weather(
        {
            "destination": "Tokyo",
            "daily_plan": [
                {"date": "2026-10-10", "activities": []},
                {"date": "2026-10-11", "activities": []},
            ],
            "research_notes": "offline",
        }
    )

    dates = [day["date"] for day in result["result"]["forecast_days"]]
    assert dates == ["2026-10-10", "2026-10-11"]


@pytest.mark.asyncio
async def test_budget_bureau():
    bureau = BudgetBureau()
    result = await bureau.run(_liubu_subtask(_liubu_payload(), "BUDGET"))
    assert result["bureau"] == "BUDGET"
    assert "budget_breakdown" in result


@pytest.mark.asyncio
async def test_budget_fallback_estimates_major_trip_categories():
    bureau = BudgetBureau()
    payload = {
        "approved_draft": {
            "destination": "Tokyo",
            "itinerary_draft": {
                "destination": "Tokyo",
                "daily_plan": [
                    {
                        "date": "2026-05-01",
                        "activities": [
                            {"title": "Museum", "estimated_cost": 40},
                            {"title": "Dinner", "estimated_cost": 60},
                        ],
                    },
                    {"date": "2026-05-02", "activities": []},
                ],
            },
        },
        "execution_plan": {
            "user_request": {
                "profile": {
                    "origin_city": "Beijing",
                    "destination_preferences": ["Tokyo"],
                    "adults": 2,
                    "budget_level": "mid_range",
                    "currency": "USD",
                    "total_budget": 2500,
                }
            }
        },
    }

    result = await bureau.run(_liubu_subtask(payload, "BUDGET"))
    categories = {item["category"] for item in result["budget_breakdown"]}

    assert {"activities", "accommodation", "food", "transport", "flights", "misc"} <= categories
    assert result["total_estimated_cost"] > 1000


@pytest.mark.asyncio
async def test_calendar_bureau_uses_wrapper_model_for_structured_output(monkeypatch, tmp_path):
    seen_models = []

    class FakeStructuredRunnable:
        async def ainvoke(self, variables):
            return CalendarEventListModel(events=[])

    class FakePromptChain:
        def __or__(self, other):
            return FakeStructuredRunnable()

    class FakePromptTemplate:
        @classmethod
        def from_messages(cls, messages):
            return FakePromptChain()

    class FakeLLM:
        def with_structured_output(self, output_model):
            seen_models.append(output_model)
            return object()

    import provinces.liubu.calendar.service as calendar_service

    monkeypatch.setattr(calendar_service, "build_qwen_chat", lambda: FakeLLM())
    monkeypatch.setattr(calendar_service, "ChatPromptTemplate", FakePromptTemplate)
    bureau = CalendarBureau(output_dir=tmp_path)

    result = await bureau.build_events({"daily_plan": [], "research_notes": ""})

    assert seen_models == [CalendarEventListModel]
    assert result == {"events": [], "calendar_status": "ok", "calendar_data_source": "structured_llm", "calendar_warnings": []}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module", "bureau", "method_name", "state"),
    [
        (
            weather_service,
            WeatherBureau(),
            "synthesize_weather",
            {"destination": "Tokyo", "daily_plan": [{"date": "2026-05-01"}], "research_notes": "offline"},
        ),
        (
            flight_service,
            FlightTransportBureau(),
            "synthesize_transport",
            {"origin_city": "Beijing", "destination": "Tokyo", "profile": {"currency": "USD", "start_date": "2026-05-01"}, "daily_plan": [], "research_notes": "offline"},
        ),
        (
            budget_service,
            BudgetBureau(),
            "synthesize_budget",
            {"draft": {"daily_plan": []}, "profile": {"currency": "USD", "total_budget": 1000}, "research_notes": "offline"},
        ),
        (
            accommodation_service,
            AccommodationBureau(),
            "synthesize_accommodation",
            {"destination": "Tokyo", "profile": {"currency": "USD"}, "daily_plan": [{"date": "2026-05-01"}], "research_notes": "offline"},
        ),
    ],
)
async def test_bureau_structured_failures_log_and_return_labeled_fallback(module, bureau, method_name, state, monkeypatch, caplog):
    monkeypatch.setattr(module, "build_qwen_chat", lambda: FailingLLM())

    with caplog.at_level("WARNING"):
        result = await getattr(bureau, method_name)(state)

    payload = result["result"]
    assert payload["status"] == "fallback"
    assert payload["data_source"] == "fallback_estimate"
    assert any("structured synthesis failed" in warning for warning in payload.get("warnings", []) + payload.get("transport_notes", []) + payload.get("search_notes", []))
    assert any("structured synthesis failed" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_calendar_structured_failure_logs_and_returns_labeled_fallback(monkeypatch, caplog):
    monkeypatch.setattr(calendar_service, "build_qwen_chat", lambda: FailingLLM())
    bureau = CalendarBureau()

    with caplog.at_level("WARNING"):
        result = await bureau.build_events(
            {
                "daily_plan": [
                    {
                        "date": "2026-05-01",
                        "activities": [
                            {
                                "title": "Museum",
                                "start_time": "09:00",
                                "end_time": "10:00",
                                "location_name": "Tokyo Museum",
                                "description": "Visit museum.",
                            }
                        ],
                    }
                ],
                "research_notes": "offline",
            }
        )

    assert result["calendar_status"] == "fallback"
    assert result["calendar_data_source"] == "fallback_estimate"
    assert any("structured synthesis failed" in warning for warning in result["calendar_warnings"])
    assert any("structured synthesis failed" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module", "bureau", "method_name", "state", "result_key"),
    [
        (
            weather_service,
            WeatherBureau(),
            "synthesize_weather",
            {"destination": "Tokyo", "daily_plan": [{"date": "2026-05-01"}], "research_notes": "live weather pending"},
            "result",
        ),
        (
            flight_service,
            FlightTransportBureau(),
            "synthesize_transport",
            {"origin_city": "Beijing", "destination": "Tokyo", "profile": {"currency": "USD", "start_date": "2026-05-01"}, "daily_plan": [], "research_notes": "live transport pending"},
            "result",
        ),
        (
            budget_service,
            BudgetBureau(),
            "synthesize_budget",
            {"draft": {"daily_plan": []}, "profile": {"currency": "USD", "total_budget": 1000}, "research_notes": "live budget pending"},
            "result",
        ),
        (
            accommodation_service,
            AccommodationBureau(),
            "synthesize_accommodation",
            {"destination": "Tokyo", "profile": {"currency": "USD"}, "daily_plan": [{"date": "2026-05-01"}], "research_notes": "live hotels pending"},
            "result",
        ),
    ],
)
async def test_bureau_structured_synthesis_timeout_returns_fallback(module, bureau, method_name, state, result_key, monkeypatch):
    monkeypatch.setattr(module, "build_qwen_chat", lambda: HangingLLM())
    monkeypatch.setattr(module, "ChatPromptTemplate", HangingPromptTemplate)
    monkeypatch.setattr(module, "STRUCTURED_SYNTHESIS_TIMEOUT_SECONDS", 0.01, raising=False)

    result = await asyncio.wait_for(getattr(bureau, method_name)(state), timeout=1.0)

    payload = result[result_key]
    assert payload["status"] == "fallback"
    assert payload["data_source"] == "fallback_estimate"
    assert "timed out" in str(payload).lower()


@pytest.mark.asyncio
async def test_calendar_structured_synthesis_timeout_returns_fallback(monkeypatch):
    monkeypatch.setattr(calendar_service, "build_qwen_chat", lambda: HangingLLM())
    monkeypatch.setattr(calendar_service, "ChatPromptTemplate", HangingPromptTemplate)
    monkeypatch.setattr(calendar_service, "STRUCTURED_SYNTHESIS_TIMEOUT_SECONDS", 0.01, raising=False)
    bureau = CalendarBureau()

    result = await asyncio.wait_for(
        bureau.build_events(
            {
                "daily_plan": [
                    {
                        "date": "2026-05-01",
                        "activities": [
                            {
                                "title": "Museum",
                                "start_time": "09:00",
                                "end_time": "10:00",
                                "location_name": "Tokyo Museum",
                                "description": "Visit museum.",
                            }
                        ],
                    }
                ],
                "research_notes": "live calendar pending",
            }
        ),
        timeout=1.0,
    )

    assert result["calendar_status"] == "fallback"
    assert result["calendar_data_source"] == "fallback_estimate"
    assert any("timed out" in warning.lower() for warning in result["calendar_warnings"])


@pytest.mark.asyncio
async def test_calendar_uses_itinerary_dates_and_writes_utc_icalendar(tmp_path):
    bureau = CalendarBureau(output_dir=tmp_path)
    payload = {
        "request_id": "calendar_dates",
        "approved_draft": {
            "destination": "Tokyo",
            "itinerary_draft": {
                "destination": "Tokyo",
                "daily_plan": [
                    {
                        "date": "2026-05-01",
                        "activities": [
                            {
                                "title": "Museum",
                                "start_time": "09:15",
                                "end_time": "10:45",
                                "location_name": "Tokyo Museum",
                                "description": "Visit museum.",
                            }
                        ],
                    }
                ],
            },
        },
    }

    result = await bureau.run(_liubu_subtask(payload, "CALENDAR"))
    content = (tmp_path / "calendar_dates_trip_calendar.ics").read_text(encoding="utf-8")

    assert result["events_created"] == 1
    assert result["status"] == "fallback"
    assert result["data_source"] == "fallback_estimate"
    assert "DTSTART:20260501T091500Z" in content
    assert "DTEND:20260501T104500Z" in content
    assert "X-MA-DATA-SOURCE:fallback_estimate" in content


@pytest.mark.asyncio
async def test_calendar_missing_itinerary_date_returns_error_without_current_time(tmp_path):
    bureau = CalendarBureau(output_dir=tmp_path)
    payload = {
        "request_id": "calendar_missing_date",
        "approved_draft": {
            "destination": "Tokyo",
            "itinerary_draft": {
                "destination": "Tokyo",
                "daily_plan": [
                    {
                        "activities": [
                            {
                                "title": "Museum",
                                "start_time": "09:15",
                                "end_time": "10:45",
                                "location_name": "Tokyo Museum",
                                "description": "Visit museum.",
                            }
                        ],
                    }
                ],
            },
        },
    }

    result = await bureau.run(_liubu_subtask(payload, "CALENDAR"))
    content = (tmp_path / "calendar_missing_date_trip_calendar.ics").read_text(encoding="utf-8")

    assert result["status"] == "error"
    assert result["data_source"] == "unavailable"
    assert result["events_created"] == 0
    assert any("missing itinerary date" in warning.lower() for warning in result["warnings"])
    assert "DTSTART" not in content
    assert "2026" not in content


@pytest.mark.asyncio
async def test_liubu_bureaus_can_fill_current_review_requirements_without_live_llm(monkeypatch, tmp_path):
    monkeypatch.setattr(weather_service, "build_qwen_chat", lambda: FailingLLM())
    monkeypatch.setattr(flight_service, "build_qwen_chat", lambda: FailingLLM())
    monkeypatch.setattr(accommodation_service, "build_qwen_chat", lambda: FailingLLM())
    monkeypatch.setattr(budget_service, "build_qwen_chat", lambda: FailingLLM())
    monkeypatch.setattr(calendar_service, "build_qwen_chat", lambda: FailingLLM())

    daily_plan = [
        {
            "day_index": 1,
            "date": "2026-05-01",
            "city": "Tokyo",
            "theme": "Tokyo culture",
            "summary": "Uses live MCP place results.",
            "activities": [
                {
                    "start_time": "09:00",
                    "end_time": "11:00",
                    "title": "Senso-ji Temple",
                    "location_name": "2 Chome-3-1 Asakusa, Tokyo",
                    "description": "Visit the named temple.",
                    "map_link": "https://www.google.com/maps/search/?api=1&query=Senso-ji+Temple",
                    "status": "pending",
                    "estimated_cost": 0,
                    "transport": {
                        "from_location": "hotel",
                        "to_location": "2 Chome-3-1 Asakusa, Tokyo",
                        "mode": "transit",
                        "duration_text": "Confirm live transit duration before final approval.",
                        "status": "pending",
                    },
                }
            ],
        }
    ]
    approved_draft = {
        "destination": "Tokyo",
        "itinerary_draft": {"destination": "Tokyo", "daily_plan": daily_plan},
    }
    execution_plan = {
        "user_request": {
            "profile": {
                "origin_city": "Beijing",
                "destination_preferences": ["Tokyo"],
                "start_date": "2026-05-01",
                "end_date": "2026-05-01",
                "adults": 1,
                "budget_level": "mid_range",
                "currency": "USD",
                "total_budget": 1800,
            }
        }
    }
    payload = {
        "request_id": "liubu_capability",
        "approved_draft": approved_draft,
        "execution_plan": execution_plan,
    }

    weather = (await WeatherBureau().synthesize_weather({"destination": "Tokyo", "daily_plan": daily_plan, "research_notes": "weather unavailable"}))["result"]
    transport = (await FlightTransportBureau().synthesize_transport({"origin_city": "Beijing", "destination": "Tokyo", "profile": execution_plan["user_request"]["profile"], "daily_plan": daily_plan, "research_notes": "transport unavailable"}))["result"]
    accommodation = (await AccommodationBureau().synthesize_accommodation({"destination": "Tokyo", "profile": execution_plan["user_request"]["profile"], "daily_plan": daily_plan, "research_notes": "hotel unavailable"}))["result"]
    budget = (await BudgetBureau().synthesize_budget({"draft": approved_draft["itinerary_draft"], "profile": execution_plan["user_request"]["profile"], "research_notes": "budget unavailable"}))["result"]
    calendar = await CalendarBureau(output_dir=tmp_path).run(_liubu_subtask(payload, "CALENDAR"))

    assert weather["forecast_days"]
    assert weather["packing_list"]
    assert any("weather" in warning.lower() for warning in weather["warnings"])
    assert transport["flight_options"]
    assert all(option["duration_minutes"] for option in transport["flight_options"])
    assert transport["booking_links"]
    assert accommodation["hotel_options"]
    assert accommodation["booking_links"]
    assert {"activities", "accommodation", "food", "transport", "flights", "misc"} <= {item["category"] for item in budget["budget_breakdown"]}
    assert calendar["events_created"] == 1
    assert calendar["calendar_file"].endswith("_trip_calendar.ics")


@pytest.mark.asyncio
async def test_flight_transport_successful_tool_messages_become_live_evidence(monkeypatch):
    async def fake_run_tool_node(tool_node, state):
        return {
            "messages": [
                ToolMessage(
                    content='{"price": 260, "currency": "USD", "route": "PEK-HND"}',
                    name="search_flights",
                    tool_call_id="flight-1",
                )
            ]
        }

    monkeypatch.setattr(flight_service, "build_qwen_chat", lambda: None)
    monkeypatch.setattr(flight_service, "run_tool_node_collect_evidence", fake_run_tool_node)
    bureau = FlightTransportBureau()
    async def fake_agent_reasoning(state):
        return {
            "worker_input": state["worker_input"],
            "messages": [
                AIMessage(
                    content="search flights",
                    tool_calls=[{"name": "search_flights", "args": {}, "id": "flight-1"}],
                )
            ],
        }

    bureau._agent_reasoning = fake_agent_reasoning
    payload = {
        "request_id": "flight_tool_placeholder",
        "approved_draft": {"destination": "Tokyo", "itinerary_draft": {"destination": "Tokyo", "daily_plan": [{"date": "2026-10-01", "activities": []}]}},
        "execution_plan": {"user_request": {"profile": {"origin_city": "Beijing", "origin_airport_code": "PEK", "destination_airport_code": "HND", "start_date": "2026-10-01", "end_date": "2026-10-01", "adults": 2, "currency": "USD"}}},
    }

    result = await bureau.run(_liubu_subtask(payload, "FLIGHT_TRANSPORT"))

    assert result["bureau"] == "FLIGHT_TRANSPORT"
    assert result["data_source"] == "live"
    assert result["liubu_evidence"][0]["tool_name"] == "search_flights"
    assert result["liubu_evidence"][0]["status"] == "ok"


@pytest.mark.asyncio
async def test_accommodation_successful_tool_messages_become_live_evidence(monkeypatch):
    async def fake_run_tool_node(tool_node, state):
        return {
            "messages": [
                ToolMessage(
                    content='{"hotel": "Tokyo Central Hotel", "check_in_date": "2026-10-01"}',
                    name="search_hotels",
                    tool_call_id="hotel-1",
                )
            ]
        }

    monkeypatch.setattr(accommodation_service, "build_qwen_chat", lambda: None)
    monkeypatch.setattr(accommodation_service, "run_tool_node_collect_evidence", fake_run_tool_node)
    bureau = AccommodationBureau()
    async def fake_agent_reasoning(state):
        return {
            "worker_input": state["worker_input"],
            "messages": [
                AIMessage(
                    content="search hotels",
                    tool_calls=[{"name": "search_hotels", "args": {}, "id": "hotel-1"}],
                )
            ],
        }

    bureau._agent_reasoning = fake_agent_reasoning
    payload = {
        "request_id": "hotel_tool_placeholder",
        "approved_draft": {"destination": "Tokyo", "itinerary_draft": {"destination": "Tokyo", "daily_plan": [{"date": "2026-10-01", "activities": []}, {"date": "2026-10-02", "activities": []}]}},
        "execution_plan": {"user_request": {"profile": {"start_date": "2026-10-01", "end_date": "2026-10-02", "adults": 2, "currency": "USD"}}},
    }

    result = await bureau.run(_liubu_subtask(payload, "ACCOMMODATION"))

    assert result["bureau"] == "ACCOMMODATION"
    assert result["data_source"] == "live"
    assert result["liubu_evidence"][0]["tool_name"] == "search_hotels"
    assert result["liubu_evidence"][0]["status"] == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module", "bureau_cls", "bureau_name", "tool_name", "payload"),
    [
        (
            weather_service,
            WeatherBureau,
            "WEATHER",
            "search_weather_context",
            {
                "request_id": "weather_tool_live",
                "approved_draft": {"destination": "Tokyo", "itinerary_draft": {"destination": "Tokyo", "daily_plan": [{"date": "2026-10-01", "activities": []}]}},
                "execution_plan": {"user_request": {"profile": {"origin_city": "Beijing", "start_date": "2026-10-01", "end_date": "2026-10-01", "currency": "USD"}}},
            },
        ),
        (
            budget_service,
            BudgetBureau,
            "BUDGET",
            "search_budget_context",
            {
                "request_id": "budget_tool_live",
                "approved_draft": {"destination": "Tokyo", "itinerary_draft": {"destination": "Tokyo", "daily_plan": [{"date": "2026-10-01", "activities": []}]}},
                "execution_plan": {"user_request": {"profile": {"origin_city": "Beijing", "start_date": "2026-10-01", "end_date": "2026-10-01", "adults": 2, "currency": "USD", "total_budget": 2500}}},
            },
        ),
        (
            calendar_service,
            CalendarBureau,
            "CALENDAR",
            "search_calendar_context",
            {
                "request_id": "calendar_tool_live",
                "approved_draft": {
                    "destination": "Tokyo",
                    "itinerary_draft": {
                        "destination": "Tokyo",
                        "daily_plan": [
                            {
                                "date": "2026-10-01",
                                "activities": [
                                    {
                                        "title": "Museum",
                                        "start_time": "09:00",
                                        "end_time": "10:00",
                                        "location_name": "Tokyo Museum",
                                        "description": "Visit museum.",
                                    }
                                ],
                            }
                        ],
                    },
                },
                "execution_plan": {"user_request": {"profile": {"origin_city": "Beijing", "start_date": "2026-10-01", "end_date": "2026-10-01", "currency": "USD"}}},
            },
        ),
    ],
)
async def test_general_liubu_successful_tool_messages_become_live_evidence(module, bureau_cls, bureau_name, tool_name, payload, monkeypatch):
    async def fake_run_tool_node(tool_node, state):
        return {
            "messages": [
                ToolMessage(
                    content='{"source": "live", "detail": "official ToolNode output"}',
                    name=tool_name,
                    tool_call_id="tool-1",
                )
            ]
        }

    class FakeBoundToolModel:
        async def ainvoke(self, messages):
            return AIMessage(
                content="search live context",
                tool_calls=[{"name": tool_name, "args": {}, "id": "tool-1"}],
            )

    monkeypatch.setattr(module, "build_qwen_chat", lambda: None)
    monkeypatch.setattr(module, "run_tool_node_collect_evidence", fake_run_tool_node)
    bureau = bureau_cls()
    bureau.bound_tool_model = FakeBoundToolModel()
    bureau._tooling_ready = True
    bureau.graph = bureau._build_graph()

    result = await bureau.run(_liubu_subtask(payload, bureau_name))

    assert result["bureau"] == bureau_name
    assert result["data_source"] == "live"
    assert result["liubu_evidence"][0]["tool_name"] == tool_name
    assert result["liubu_evidence"][0]["status"] == "ok"


def test_tool_messages_with_tool_exception_are_error_evidence():
    evidence = official_tooling.tool_messages_to_evidence(
        [
            ToolMessage(
                content='Error: ToolException("Error calling tool search_google_flights")\n Please fix your mistakes.',
                name="search_google_flights",
                tool_call_id="tool-error",
            )
        ]
    )

    assert evidence[0].status == "error"
    assert evidence[0].result is None
    assert "ToolException" in evidence[0].error


@pytest.mark.asyncio
async def test_weather_amap_tool_message_builds_live_forecast(monkeypatch):
    async def fake_run_tool_node(tool_node, state):
        return {
            "messages": [
                ToolMessage(
                    content=(
                        '{"status":"1","forecasts":[{"city":"\u4e0a\u6d77\u5e02","casts":['
                        '{"date":"2026-10-24","dayweather":"\u591a\u4e91","nightweather":"\u6674",'
                        '"daytemp":"24","nighttemp":"18","daypower":"3","nightpower":"3"}]}]}'
                    ),
                    name="maps_weather",
                    tool_call_id="weather-1",
                )
            ]
        }

    class FakeBoundToolModel:
        async def ainvoke(self, messages):
            return AIMessage(
                content="search amap weather",
                tool_calls=[{"name": "maps_weather", "args": {"city": "\u4e0a\u6d77"}, "id": "weather-1"}],
            )

    monkeypatch.setattr(weather_service, "build_qwen_chat", lambda: None)
    monkeypatch.setattr(weather_service, "run_tool_node_collect_evidence", fake_run_tool_node)
    bureau = WeatherBureau()
    bureau.bound_tool_model = FakeBoundToolModel()
    bureau._tooling_ready = True
    bureau.graph = bureau._build_graph()
    payload = {
        "request_id": "weather_amap_live",
        "approved_draft": {
            "destination": "\u4e0a\u6d77",
            "itinerary_draft": {
                "destination": "\u4e0a\u6d77",
                "daily_plan": [{"date": "2026-10-24", "activities": []}],
            },
        },
        "execution_plan": {"user_request": {"profile": {"currency": "CNY"}}},
    }

    result = await bureau.run(_liubu_subtask(payload, "WEATHER"))

    assert result["status"] == "ok"
    assert result["data_source"] == "live"
    assert result["forecast_days"][0]["condition"] == "\u591a\u4e91"
    assert result["forecast_days"][0]["is_estimated"] is False
    assert not any("fallback" in warning.lower() for warning in result["warnings"])


@pytest.mark.asyncio
async def test_weather_amap_flat_forecast_builds_live_forecast(monkeypatch):
    async def fake_run_tool_node(tool_node, state):
        return {
            "messages": [
                ToolMessage(
                    content=(
                        '{"city":"\u4e0a\u6d77\u5e02","forecasts":['
                        '{"date":"2026-10-24","dayweather":"\u591a\u4e91","nightweather":"\u6674",'
                        '"daytemp":"24","nighttemp":"18","daypower":"3","nightpower":"3"}]}'
                    ),
                    name="maps_weather",
                    tool_call_id="weather-1",
                )
            ]
        }

    class FakeBoundToolModel:
        async def ainvoke(self, messages):
            return AIMessage(
                content="search amap weather",
                tool_calls=[{"name": "maps_weather", "args": {"city": "\u4e0a\u6d77"}, "id": "weather-1"}],
            )

    monkeypatch.setattr(weather_service, "build_qwen_chat", lambda: None)
    monkeypatch.setattr(weather_service, "run_tool_node_collect_evidence", fake_run_tool_node)
    bureau = WeatherBureau()
    bureau.bound_tool_model = FakeBoundToolModel()
    bureau._tooling_ready = True
    bureau.available_tool_names = {"maps_weather"}
    bureau.graph = bureau._build_graph()
    payload = {
        "request_id": "weather_amap_flat_live",
        "approved_draft": {
            "destination": "\u4e0a\u6d77",
            "itinerary_draft": {
                "destination": "\u4e0a\u6d77",
                "daily_plan": [{"date": "2026-10-24", "activities": []}],
            },
        },
        "execution_plan": {"user_request": {"profile": {"currency": "CNY"}}},
    }

    result = await bureau.run(_liubu_subtask(payload, "WEATHER"))

    assert result["status"] == "ok"
    assert result["data_source"] == "live"
    assert result["forecast_days"][0]["condition"] == "\u591a\u4e91"
    assert result["forecast_days"][0]["is_estimated"] is False
    assert "fallback" not in str(result).lower()


@pytest.mark.asyncio
async def test_budget_from_live_itinerary_has_no_fallback_language(monkeypatch):
    monkeypatch.setattr(budget_service, "build_qwen_chat", lambda: None)
    bureau = BudgetBureau()
    bureau._tooling_ready = True
    payload = {
        "request_id": "budget_clean_live_inputs",
        "approved_draft": {
            "destination": "\u4e0a\u6d77",
            "itinerary_draft": {
                "destination": "\u4e0a\u6d77",
                "daily_plan": [
                    {
                        "date": "2026-10-24",
                        "activities": [
                            {
                                "title": "\u4e0a\u6d77\u535a\u7269\u9986",
                                "estimated_cost": 30,
                                "map_link": "https://ditu.amap.com/search?query=shanghai-museum",
                            }
                        ],
                    }
                ],
            },
        },
        "execution_plan": {
            "user_request": {
                "profile": {
                    "origin_city": "\u5317\u4eac",
                    "start_date": "2026-10-24",
                    "end_date": "2026-10-24",
                    "adults": 1,
                    "currency": "CNY",
                    "total_budget": 1800,
                }
            }
        },
    }

    result = await bureau.run(_liubu_subtask(payload, "BUDGET"))

    assert result["status"] == "ok"
    assert result["data_source"] == "live"
    assert "fallback" not in str(result).lower()


@pytest.mark.asyncio
async def test_accommodation_amap_poi_builds_clean_live_options(monkeypatch):
    async def fake_run_tool_node(tool_node, state):
        return {
            "messages": [
                ToolMessage(
                    content='{"pois":[{"name":"\u4e0a\u6d77\u5916\u6ee9\u9152\u5e97","address":"\u9ec4\u6d66\u533a","type":"\u4f4f\u5bbf\u670d\u52a1;\u5bbe\u9986\u9152\u5e97","location":"121.49,31.24"}]}',
                    name="maps_text_search",
                    tool_call_id="hotel-1",
                )
            ]
        }

    class FakeBoundToolModel:
        async def ainvoke(self, messages):
            return AIMessage(
                content="search amap hotels",
                tool_calls=[{"name": "maps_text_search", "args": {"keywords": "\u4e0a\u6d77 \u9152\u5e97"}, "id": "hotel-1"}],
            )

    monkeypatch.setattr(accommodation_service, "build_qwen_chat", lambda: None)
    monkeypatch.setattr(accommodation_service, "run_tool_node_collect_evidence", fake_run_tool_node)
    bureau = AccommodationBureau()
    bureau.bound_tool_model = FakeBoundToolModel()
    bureau._tooling_ready = True
    bureau.graph = bureau._build_graph()
    payload = {
        "request_id": "hotel_amap_live",
        "approved_draft": {"destination": "\u4e0a\u6d77", "itinerary_draft": {"destination": "\u4e0a\u6d77", "daily_plan": [{"date": "2026-10-24", "activities": []}]}},
        "execution_plan": {"user_request": {"profile": {"start_date": "2026-10-24", "end_date": "2026-10-25", "adults": 1, "currency": "CNY"}}},
    }

    result = await bureau.run(_liubu_subtask(payload, "ACCOMMODATION"))

    assert result["status"] == "ok"
    assert result["data_source"] == "live"
    assert result["hotel_options"][0]["name"] == "\u4e0a\u6d77\u5916\u6ee9\u9152\u5e97"
    assert "fallback" not in str(result).lower()


@pytest.mark.asyncio
async def test_flight_transport_amap_geo_builds_clean_live_options(monkeypatch):
    async def fake_run_tool_node(tool_node, state):
        return {
            "messages": [
                ToolMessage(
                    content='{"geocodes":[{"location":"116.4074,39.9042"}]}',
                    name="maps_geo",
                    tool_call_id="geo-1",
                ),
                ToolMessage(
                    content='{"geocodes":[{"location":"121.4737,31.2304"}]}',
                    name="maps_geo",
                    tool_call_id="geo-2",
                )
            ]
        }

    class FakeBoundToolModel:
        async def ainvoke(self, messages):
            return AIMessage(
                content="search amap geo",
                tool_calls=[
                    {"name": "maps_geo", "args": {"address": "\u5317\u4eac"}, "id": "geo-1"},
                    {"name": "maps_geo", "args": {"address": "\u4e0a\u6d77"}, "id": "geo-2"},
                ],
            )

    monkeypatch.setattr(flight_service, "build_qwen_chat", lambda: None)
    monkeypatch.setattr(flight_service, "run_tool_node_collect_evidence", fake_run_tool_node)
    bureau = FlightTransportBureau()
    bureau.bound_tool_model = FakeBoundToolModel()
    bureau._tooling_ready = True
    bureau.graph = bureau._build_graph()
    payload = {
        "request_id": "transport_amap_live",
        "approved_draft": {"destination": "\u4e0a\u6d77", "itinerary_draft": {"destination": "\u4e0a\u6d77", "daily_plan": [{"date": "2026-10-24", "activities": []}]}},
        "execution_plan": {"user_request": {"profile": {"origin_city": "\u5317\u4eac", "origin_airport_code": "PEK", "destination_airport_code": "PVG", "start_date": "2026-10-24", "end_date": "2026-10-24", "adults": 1, "currency": "CNY"}}},
    }

    result = await bureau.run(_liubu_subtask(payload, "FLIGHT_TRANSPORT"))

    assert result["status"] == "ok"
    assert result["data_source"] == "live"
    assert result["flight_options"][0]["departure_airport"] == "PEK"
    assert result["flight_options"][0]["arrival_airport"] == "PVG"
    assert "fallback" not in str(result).lower()


@pytest.mark.asyncio
async def test_flight_transport_amap_results_geo_builds_clean_live_options(monkeypatch):
    async def fake_run_tool_node(tool_node, state):
        return {
            "messages": [
                ToolMessage(
                    content='{"results":[{"location":"116.407387,39.904179"}]}',
                    name="maps_geo",
                    tool_call_id="geo-1",
                ),
                ToolMessage(
                    content='{"results":[{"location":"121.473667,31.230525"}]}',
                    name="maps_geo",
                    tool_call_id="geo-2",
                ),
            ]
        }

    class FakeBoundToolModel:
        async def ainvoke(self, messages):
            return AIMessage(
                content="search amap geo",
                tool_calls=[
                    {"name": "maps_geo", "args": {"address": "\u5317\u4eac"}, "id": "geo-1"},
                    {"name": "maps_geo", "args": {"address": "\u4e0a\u6d77"}, "id": "geo-2"},
                ],
            )

    monkeypatch.setattr(flight_service, "build_qwen_chat", lambda: None)
    monkeypatch.setattr(flight_service, "run_tool_node_collect_evidence", fake_run_tool_node)
    bureau = FlightTransportBureau()
    bureau.bound_tool_model = FakeBoundToolModel()
    bureau._tooling_ready = True
    bureau.graph = bureau._build_graph()
    payload = {
        "request_id": "transport_amap_results_live",
        "approved_draft": {"destination": "\u4e0a\u6d77", "itinerary_draft": {"destination": "\u4e0a\u6d77", "daily_plan": [{"date": "2026-10-24", "activities": []}]}},
        "execution_plan": {"user_request": {"profile": {"origin_city": "\u5317\u4eac", "origin_airport_code": "PEK", "destination_airport_code": "PVG", "start_date": "2026-10-24", "end_date": "2026-10-24", "adults": 1, "currency": "CNY"}}},
    }

    result = await bureau.run(_liubu_subtask(payload, "FLIGHT_TRANSPORT"))

    assert result["status"] == "ok"
    assert result["data_source"] == "live"
    assert "fallback" not in str(result).lower()


@pytest.mark.asyncio
async def test_calendar_itinerary_derived_complete_events_are_ok_not_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(calendar_service, "build_qwen_chat", lambda: None)
    bureau = CalendarBureau(output_dir=tmp_path)
    bureau._tooling_ready = True
    payload = {
        "request_id": "calendar_live_itinerary",
        "approved_draft": {
            "destination": "\u4e0a\u6d77",
            "itinerary_draft": {
                "destination": "\u4e0a\u6d77",
                "daily_plan": [
                    {
                        "date": "2026-10-24",
                        "activities": [
                            {
                                "title": "\u4e0a\u6d77\u535a\u7269\u9986",
                                "start_time": "09:00",
                                "end_time": "11:00",
                                "location_name": "\u4e0a\u6d77\u535a\u7269\u9986",
                                "description": "Visit a concrete approved itinerary place.",
                                "map_link": "https://ditu.amap.com/search?query=shanghai-museum",
                            }
                        ],
                    }
                ],
            },
        },
        "execution_plan": {"user_request": {"profile": {"currency": "CNY"}}},
    }

    result = await bureau.run(_liubu_subtask(payload, "CALENDAR"))

    assert result["status"] == "ok"
    assert result["data_source"] == "live"
    assert result["events_created"] == 1
    assert not any("fallback" in warning.lower() for warning in result["warnings"])
