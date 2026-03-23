import pytest
from workflow import ProvinceWorkflow
from utils.schemas import PlanningRequest, TravelerProfile


@pytest.mark.asyncio
async def test_full_workflow_integration():
    """完整工作流集成测试（包含LLM调用）"""
    workflow = ProvinceWorkflow()
    request = PlanningRequest(
        request_id="test_integration_001",
        user_message="计划东京3日游",
        profile=TravelerProfile(
            destination_preferences=["Tokyo"],
            origin_city="Beijing",
            start_date="2026-05-01",
            end_date="2026-05-03",
            total_budget=3000,
            currency="USD",
            adults=2,
            interests=["culture"]
        )
    )
    result = await workflow.run(request)
    assert result["status"] in ["DONE", "HUMAN_INTERVENE", "REJECTED"]
    if result["status"] == "DONE":
        assert "final_package" in result
