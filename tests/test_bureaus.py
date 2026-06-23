import asyncio

import pytest

import provinces.liubu.accommodation.service as accommodation_service
import provinces.liubu.budget.service as budget_service
import provinces.liubu.calendar.service as calendar_service
import provinces.liubu.flight_transport.service as flight_service
import provinces.liubu.weather.service as weather_service
from provinces.liubu.accommodation.service import AccommodationBureau
from provinces.liubu.weather.service import WeatherBureau
from provinces.liubu.budget.service import BudgetBureau
from provinces.liubu.calendar.service import CalendarBureau
from provinces.liubu.flight_transport.service import FlightTransportBureau
from utils.schemas import CalendarEventListModel


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


@pytest.mark.asyncio
async def test_weather_bureau():
    bureau = WeatherBureau()
    payload = {
        "destination": "Tokyo",
        "daily_plan": [{"date": "2026-05-01"}]
    }
    result = await bureau.run(payload)
    assert result["bureau"] == "WEATHER"
    assert "forecast_days" in result


@pytest.mark.asyncio
async def test_budget_bureau():
    bureau = BudgetBureau()
    payload = {
        "daily_plan": [],
        "currency": "USD",
        "total_budget": 5000
    }
    result = await bureau.run(payload)
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

    result = await bureau.run(payload)
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

    result = await bureau.run(payload)
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

    result = await bureau.run(payload)
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
    calendar = await CalendarBureau(output_dir=tmp_path).run(payload)

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
async def test_flight_transport_blocks_agent_tool_call_with_wrong_trip_facts(monkeypatch):
    class FakeAgent:
        async def ainvoke(self, state):
            return {
                "tool_requests": [
                    {
                        "tool": "google_flights",
                        "args": {
                            "departure_id": "SHA",
                            "arrival_id": "NRT",
                            "outbound_date": "2023-10-01",
                            "adults": 1,
                            "currency": "JPY",
                        },
                    }
                ]
            }

    async def fake_tool_map(server_names, allowed_tool_names):
        return {"google_flights": object()}

    monkeypatch.setattr(flight_service, "build_qwen_chat", lambda: None)
    monkeypatch.setattr(flight_service, "load_allowed_tool_map", fake_tool_map)
    bureau = FlightTransportBureau()
    bureau._agent_reasoning = FakeAgent().ainvoke
    payload = {
        "request_id": "flight_wrong_args",
        "approved_draft": {"destination": "Tokyo", "itinerary_draft": {"destination": "Tokyo", "daily_plan": [{"date": "2026-10-01", "activities": []}]}},
        "execution_plan": {"user_request": {"profile": {"origin_city": "Beijing", "origin_airport_code": "PEK", "destination_airport_code": "HND", "start_date": "2026-10-01", "end_date": "2026-10-01", "adults": 2, "currency": "USD"}}},
    }

    result = await bureau.run(payload)

    assert result["bureau"] == "FLIGHT_TRANSPORT"
    assert result["status"] == "fallback"
    assert result["data_source"] == "fallback_estimate"
    assert any("departure_id must match PEK" in note for note in result["transport_notes"])
    assert result["liubu_quality"]["passed"] is False
    assert result["liubu_evidence"][0]["status"] == "blocked"
