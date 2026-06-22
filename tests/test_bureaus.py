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
