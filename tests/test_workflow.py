import pytest
from workflow import ProvinceWorkflow
from utils.schemas import PlanningRequest, TravelerProfile


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
