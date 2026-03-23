import pytest
from provinces.liubu.weather.service import WeatherBureau
from provinces.liubu.budget.service import BudgetBureau


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
