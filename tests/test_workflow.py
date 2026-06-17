import pytest
from workflow import ProvinceWorkflow
from utils.schemas import PlanningRequest, TravelerProfile


class RaisingBureau:
    async def run(self, payload):
        raise KeyError("boom")


@pytest.mark.asyncio
async def test_preflight():
    workflow = ProvinceWorkflow()
    request = PlanningRequest(
        request_id="test_002",
        user_message="Test trip to Tokyo",
        profile=TravelerProfile(
            destination_preferences=["Tokyo"],
            origin_city="Beijing",
            start_date="2026-05-01",
            end_date="2026-05-05",
            total_budget=5000
        )
    )
    state = {"request": request.model_dump(mode="json")}
    result = await workflow._node_preflight(state)
    assert "context" in result
    assert result["status"] == "RUNNING"


@pytest.mark.asyncio
async def test_liubu_node_returns_fallback_result_when_bureau_raises():
    workflow = ProvinceWorkflow()
    workflow.weather = RaisingBureau()
    request = PlanningRequest(
        request_id="test_liubu_failure",
        user_message="Test trip to Tokyo",
        profile=TravelerProfile(
            destination_preferences=["Tokyo"],
            origin_city="Beijing",
            start_date="2026-05-01",
            end_date="2026-05-03",
            total_budget=5000,
        ),
    )
    context = workflow.orchestrator.bootstrap(request.request_id, request.model_dump(mode="json"))
    state = {
        "context": context,
        "payload": {
            "approved_draft": {
                "destination": "Tokyo",
                "itinerary_draft": {"destination": "Tokyo", "daily_plan": []},
            }
        },
    }

    result = await workflow._node_liubu_weather(state)

    weather = result["execution_results"]["WEATHER"]
    assert weather["bureau"] == "WEATHER"
    assert weather["destination"] == "Tokyo"
    assert weather["forecast_days"] == []
    assert any("failed" in warning.lower() for warning in weather["warnings"])
