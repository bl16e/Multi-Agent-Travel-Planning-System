import pytest
from langgraph.types import Send

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


def test_route_to_liubu_rejects_empty_task_list():
    workflow = ProvinceWorkflow()

    with pytest.raises(ValueError, match="No Liubu bureau tasks"):
        workflow._route_to_liubu({"liubu_tasks": []})


def test_route_to_liubu_keeps_each_task_payload_isolated():
    workflow = ProvinceWorkflow()
    state = {
        "context": object(),
        "liubu_tasks": [
            {"node": "liubu_weather", "payload": {"target_bureau": "WEATHER"}},
            {"node": "liubu_budget", "payload": {"target_bureau": "BUDGET"}},
        ],
    }

    sends = workflow._route_to_liubu(state)

    assert all(isinstance(send, Send) for send in sends)
    assert [send.node for send in sends] == ["liubu_weather", "liubu_budget"]
    assert sends[0].arg["payload"] == {"target_bureau": "WEATHER"}
    assert sends[1].arg["payload"] == {"target_bureau": "BUDGET"}


@pytest.mark.asyncio
async def test_finish_human_creates_boundary_resume_state():
    workflow = ProvinceWorkflow()
    request = PlanningRequest(
        request_id="resume_boundary",
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
    context.pending_user_inputs.append("Need budget")

    result = await workflow._node_finish_human(
        {
            "request": request.model_dump(mode="json"),
            "context": context,
            "status": "HUMAN_INTERVENE",
            "question": "Need budget",
        }
    )

    assert result["status"] == "HUMAN_INTERVENE"
    assert result["resume_mode"] == "boundary"
    assert result["resume_state"]["mode"] == "boundary"
    assert result["resume_state"]["next_node"] == "zhongshu_itinerary"
    assert result["resume_state"]["question"] == "Need budget"
    assert result["resume_state"]["state"]["request"]["request_id"] == "resume_boundary"
    assert result["resume_state"]["state"]["context"]["request_id"] == "resume_boundary"
