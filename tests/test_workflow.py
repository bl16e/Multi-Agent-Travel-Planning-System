import pytest
from langgraph.types import Send

from workflow import ProvinceWorkflow, merge_dicts
from utils.schemas import AgentRole
from utils.schemas import PlanningRequest, TravelerProfile


class RaisingBureau:
    async def run(self, payload):
        raise KeyError("boom")


@pytest.mark.asyncio
async def test_workflow_stream_uses_langgraph_astream(monkeypatch):
    workflow = ProvinceWorkflow()
    request = PlanningRequest(
        request_id="stream_uses_astream",
        user_message="Test trip to Tokyo",
        profile=TravelerProfile(
            destination_preferences=["Tokyo"],
            origin_city="Beijing",
            start_date="2026-05-01",
            end_date="2026-05-03",
            total_budget=5000,
        ),
    )
    called = {"astream": False}

    class FakeGraph:
        async def astream(self, graph_input, config=None, stream_mode=None):
            called["astream"] = True
            assert stream_mode == ["updates", "messages"]
            yield ("updates", {"shangshu_preflight": {"status": "RUNNING"}})
            yield ("updates", {"finish_rejected": {"status": "REJECTED", "rejected_payload": {"status": "REJECTED", "request_id": request.request_id}}})

        async def aget_state(self, config):
            return type("Snapshot", (), {"values": {"status": "REJECTED", "rejected_payload": {"status": "REJECTED", "request_id": request.request_id}}, "next": (), "interrupts": ()})()

    async def fake_ensure():
        return FakeGraph()

    async def fake_close():
        return None

    monkeypatch.setattr(workflow, "_ensure_persistent_graph", fake_ensure)
    monkeypatch.setattr(workflow, "_close_checkpointer", fake_close)

    events = [event async for event in workflow.stream_run(request)]

    assert called["astream"] is True
    assert any(event["event"] == "progress" for event in events)
    assert events[-1]["event"] == "result"
    assert events[-1]["data"]["status"] == "REJECTED"


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


def test_workflow_instances_own_independent_orchestrators_and_runtime_state():
    first = ProvinceWorkflow()
    second = ProvinceWorkflow()
    first_request = PlanningRequest(
        request_id="workflow_isolation_first",
        user_message="Test trip to Tokyo",
        profile=TravelerProfile(
            destination_preferences=["Tokyo"],
            origin_city="Beijing",
            start_date="2026-05-01",
            end_date="2026-05-03",
            total_budget=5000,
        ),
    )
    second_request = PlanningRequest(
        request_id="workflow_isolation_second",
        user_message="Test trip to Seoul",
        profile=TravelerProfile(
            destination_preferences=["Seoul"],
            origin_city="Beijing",
            start_date="2026-05-01",
            end_date="2026-05-03",
            total_budget=5000,
        ),
    )
    first_context = first.orchestrator.bootstrap(
        first_request.request_id,
        first_request.model_dump(mode="json"),
    )
    second_context = second.orchestrator.bootstrap(
        second_request.request_id,
        second_request.model_dump(mode="json"),
    )

    first.orchestrator.register_execution_result(first_context, AgentRole.WEATHER, {"bureau": "WEATHER"})

    assert first.orchestrator is not second.orchestrator
    assert first.weather is not second.weather
    assert first_context.execution_results == {"WEATHER": {"bureau": "WEATHER"}}
    assert second_context.execution_results == {}


def test_liubu_fanout_uses_independent_context_snapshots_and_merges_only_results():
    workflow = ProvinceWorkflow()
    request = PlanningRequest(
        request_id="liubu_context_isolation",
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
        "liubu_tasks": [
            {"node": "liubu_weather", "payload": {"target_bureau": "WEATHER"}},
            {"node": "liubu_budget", "payload": {"target_bureau": "BUDGET"}},
        ],
    }

    sends = workflow._route_to_liubu(state)

    first_context = sends[0].arg["context"]
    second_context = sends[1].arg["context"]
    assert first_context is not context
    assert second_context is not context
    assert first_context is not second_context

    first_context.execution_results["WEATHER"] = {"bureau": "WEATHER"}

    assert second_context.execution_results == {}
    assert context.execution_results == {}
    assert merge_dicts({"WEATHER": {"bureau": "WEATHER"}}, {"BUDGET": {"bureau": "BUDGET"}}) == {
        "WEATHER": {"bureau": "WEATHER"},
        "BUDGET": {"bureau": "BUDGET"},
    }


@pytest.mark.asyncio
async def test_finish_human_creates_boundary_resume_metadata():
    workflow = ProvinceWorkflow()
    request = PlanningRequest(
        request_id="resume_placeholder",
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
    assert result["resume_state"]["thread_id"] == "resume_placeholder"
    assert result["resume_state"]["question"] == "Need budget"


@pytest.mark.asyncio
async def test_assemble_rejects_fallback_outputs_before_final_delivery(tmp_path):
    workflow = ProvinceWorkflow(artifact_dir=tmp_path)
    request = PlanningRequest(
        request_id="fallback_delivery_gate",
        user_message="Test trip to Tokyo",
        profile=TravelerProfile(
            destination_preferences=["Tokyo"],
            origin_city="Beijing",
            start_date="2026-05-01",
            end_date="2026-05-01",
            total_budget=5000,
        ),
    )
    context = workflow.orchestrator.bootstrap(request.request_id, request.model_dump(mode="json"))
    draft_packet = {
        "request_id": request.request_id,
        "destination": "Tokyo",
        "itinerary_draft": {
            "destination": "Tokyo",
            "overview": "Specific Tokyo plan",
            "trip_style": "structured",
            "daily_plan": [
                {
                    "day_index": 1,
                    "date": "2026-05-01",
                    "city": "Tokyo",
                    "theme": "Tokyo culture",
                    "summary": "Visit Senso-ji and Ueno Park with named places.",
                    "activities": [
                        {
                            "start_time": "09:00",
                            "end_time": "11:00",
                            "title": "Senso-ji Temple visit",
                            "location_name": "Senso-ji, Asakusa",
                            "description": "Visit the named temple complex.",
                        }
                    ],
                }
            ],
            "planning_notes": [],
            "pending_confirmations": [],
            "risk_flags": [],
        },
        "required_bureaus": ["WEATHER", "BUDGET"],
        "bureau_tasks": [],
        "governance": {"producer": "ZHONGSHU", "revision_round": 0},
    }
    review_packet = {
        "request_id": request.request_id,
        "verdict": "APPROVED",
        "summary": "Offline structural review approved the plan.",
        "blocking_issues": [],
        "revision_requests": [],
        "human_questions": [],
        "approved_bureaus": ["WEATHER", "BUDGET"],
        "governance": {
            "reviewer": "MENXIA",
            "source_producer": "ZHONGSHU",
            "next_hop": "SHANGSHU",
            "verdict_state": "APPROVED",
            "veto_enabled": True,
            "rejection_round": 0,
            "max_rejection_rounds": 2,
        },
        "review_notes": ["Offline review only."],
        "data_source": "fallback_estimate",
        "warnings": ["Offline structural review only."],
    }
    execution_results = {
        "WEATHER": {
            "bureau": "WEATHER",
            "status": "fallback",
            "data_source": "fallback_estimate",
            "destination": "Tokyo",
            "forecast_days": [],
            "packing_list": [],
            "warnings": ["offline weather"],
            "summary": "Fallback weather",
        },
        "BUDGET": {
            "bureau": "BUDGET",
            "status": "fallback",
            "data_source": "fallback_estimate",
            "currency": "USD",
            "budget_breakdown": [],
            "total_estimated_cost": 0,
            "warnings": ["offline budget"],
        },
    }

    result = await workflow._node_assemble(
        {
            "request": request.model_dump(mode="json"),
            "context": context,
            "draft_packet": draft_packet,
            "review_packet": review_packet,
            "execution_results": execution_results,
        }
    )

    assert result["status"] == "REJECTED"
    assert "final_package" not in result
    assert result["rejected_payload"]["reason"] == "fallback_outputs_not_deliverable"
    assert result["rejected_payload"]["fallback_sources"]


@pytest.mark.asyncio
async def test_assemble_rejects_fallback_content_even_when_marked_live(tmp_path):
    workflow = ProvinceWorkflow(artifact_dir=tmp_path)
    request = PlanningRequest(
        request_id="fallback_content_gate",
        user_message="Test domestic trip to Shanghai",
        profile=TravelerProfile(
            destination_preferences=["\u4e0a\u6d77"],
            origin_city="\u5317\u4eac",
            start_date="2026-10-24",
            end_date="2026-10-24",
            total_budget=1800,
            currency="CNY",
        ),
    )
    context = workflow.orchestrator.bootstrap(request.request_id, request.model_dump(mode="json"))
    draft_packet = {
        "request_id": request.request_id,
        "destination": "\u4e0a\u6d77",
        "itinerary_draft": {
            "destination": "\u4e0a\u6d77",
            "overview": "Live researched Shanghai plan",
            "trip_style": "structured",
            "daily_plan": [
                {
                    "day_index": 1,
                    "date": "2026-10-24",
                    "city": "\u4e0a\u6d77",
                    "theme": "\u4e0a\u6d77 \u6587\u5316",
                    "summary": "Visit Shanghai Museum with named places.",
                    "activities": [
                        {
                            "start_time": "09:00",
                            "end_time": "11:00",
                            "title": "\u4e0a\u6d77\u535a\u7269\u9986",
                            "location_name": "\u4e0a\u6d77\u535a\u7269\u9986",
                            "description": "Concrete approved activity.",
                        }
                    ],
                }
            ],
            "planning_notes": [],
            "pending_confirmations": [],
            "risk_flags": [],
        },
        "required_bureaus": ["BUDGET"],
        "bureau_tasks": [],
        "governance": {"producer": "ZHONGSHU", "revision_round": 0},
    }
    review_packet = {
        "request_id": request.request_id,
        "verdict": "APPROVED",
        "summary": "Live review approved the plan.",
        "blocking_issues": [],
        "revision_requests": [],
        "human_questions": [],
        "approved_bureaus": ["BUDGET"],
        "governance": {"reviewer": "MENXIA", "verdict_state": "APPROVED"},
        "review_notes": [],
        "data_source": "live",
        "warnings": [],
    }
    execution_results = {
        "BUDGET": {
            "bureau": "BUDGET",
            "status": "ok",
            "data_source": "live",
            "currency": "CNY",
            "budget_breakdown": [
                {
                    "category": "activities",
                    "item": "Planned activity blocks",
                    "estimated_cost": 40,
                    "currency": "CNY",
                    "notes": "Estimated fallback from itinerary activity costs.",
                }
            ],
            "total_estimated_cost": 40,
            "warnings": ["Budget output fell back because structured synthesis failed."],
        },
    }

    result = await workflow._node_assemble(
        {
            "request": request.model_dump(mode="json"),
            "context": context,
            "draft_packet": draft_packet,
            "review_packet": review_packet,
            "execution_results": execution_results,
        }
    )

    assert result["status"] == "REJECTED"
    assert result["rejected_payload"]["fallback_sources"] == [
        {"component": "BUDGET", "status": "ok", "data_source": "live", "reason": "fallback_content"}
    ]


@pytest.mark.asyncio
async def test_assemble_rejects_zhongshu_fallback_content_before_delivery(tmp_path):
    workflow = ProvinceWorkflow(artifact_dir=tmp_path)
    request = PlanningRequest(
        request_id="draft_fallback_content_gate",
        user_message="Test domestic trip to Shanghai",
        profile=TravelerProfile(
            destination_preferences=["\u4e0a\u6d77"],
            origin_city="\u5317\u4eac",
            start_date="2026-10-24",
            end_date="2026-10-24",
            total_budget=1800,
            currency="CNY",
        ),
    )
    context = workflow.orchestrator.bootstrap(request.request_id, request.model_dump(mode="json"))
    draft_packet = {
        "request_id": request.request_id,
        "destination": "\u4e0a\u6d77",
        "itinerary_draft": {
            "destination": "\u4e0a\u6d77",
            "overview": "Live MCP research was used to build this draft.",
            "trip_style": "live_research_fallback",
            "daily_plan": [],
            "planning_notes": ["data_source=live_mcp_research; synthesis=fallback_from_compact_tool_results."],
            "pending_confirmations": [],
            "risk_flags": [],
        },
        "required_bureaus": [],
        "bureau_tasks": [],
        "governance": {"producer": "ZHONGSHU", "revision_round": 0},
    }
    review_packet = {
        "request_id": request.request_id,
        "verdict": "APPROVED",
        "summary": "Live review approved the plan.",
        "blocking_issues": [],
        "revision_requests": [],
        "human_questions": [],
        "approved_bureaus": [],
        "governance": {"reviewer": "MENXIA", "verdict_state": "APPROVED"},
        "review_notes": [],
        "data_source": "live",
        "warnings": [],
    }

    result = await workflow._node_assemble(
        {
            "request": request.model_dump(mode="json"),
            "context": context,
            "draft_packet": draft_packet,
            "review_packet": review_packet,
            "execution_results": {},
        }
    )

    assert result["status"] == "REJECTED"
    assert result["rejected_payload"]["fallback_sources"] == [
        {"component": "ZHONGSHU", "status": "approved", "data_source": "live", "reason": "fallback_content"}
    ]
