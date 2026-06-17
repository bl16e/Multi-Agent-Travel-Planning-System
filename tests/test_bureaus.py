import pytest
from provinces.liubu.weather.service import WeatherBureau
from provinces.liubu.budget.service import BudgetBureau
from provinces.liubu.calendar.service import CalendarBureau
from utils.schemas import CalendarEventListModel


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
    assert result == {"events": []}
