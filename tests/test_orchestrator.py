import pytest

from provinces.shangshu_orchestrator.orchestrator import ShangshuOrchestrator
from utils.permission_matrix import AgentRole


def test_resolve_liubu_targets_defaults_when_field_missing():
    orchestrator = ShangshuOrchestrator()

    targets = orchestrator._resolve_liubu_targets({})

    assert targets == [
        AgentRole.WEATHER,
        AgentRole.CALENDAR,
        AgentRole.BUDGET,
        AgentRole.ACCOMMODATION,
        AgentRole.FLIGHT_TRANSPORT,
    ]


def test_resolve_liubu_targets_rejects_explicit_empty_list():
    orchestrator = ShangshuOrchestrator()

    with pytest.raises(ValueError, match="required_bureaus cannot be empty"):
        orchestrator._resolve_liubu_targets({"required_bureaus": []})


def test_resolve_liubu_targets_rejects_invalid_name():
    orchestrator = ShangshuOrchestrator()

    with pytest.raises(ValueError, match="Unsupported Liubu target"):
        orchestrator._resolve_liubu_targets({"required_bureaus": ["WEATHER", "NOT_A_BUREAU"]})


def test_resolve_liubu_targets_deduplicates_preserving_order():
    orchestrator = ShangshuOrchestrator()

    targets = orchestrator._resolve_liubu_targets(
        {"required_bureaus": ["WEATHER", "WEATHER", "BUDGET", "CALENDAR", "BUDGET"]}
    )

    assert targets == [AgentRole.WEATHER, AgentRole.BUDGET, AgentRole.CALENDAR]


def test_register_execution_result_records_liubu_quality_metadata():
    orchestrator = ShangshuOrchestrator()
    context = orchestrator.bootstrap("quality_meta", {"request_id": "quality_meta"})

    orchestrator.register_execution_result(
        context,
        AgentRole.FLIGHT_TRANSPORT,
        {
            "bureau": "FLIGHT_TRANSPORT",
            "status": "fallback",
            "data_source": "fallback_estimate",
            "liubu_quality": {
                "passed": False,
                "findings": [{"severity": "error", "code": "wrong_departure_date", "message": "date mismatch"}],
            },
        },
    )

    assert context.execution_results["FLIGHT_TRANSPORT"]["liubu_quality"]["passed"] is False
    assert context.quality_gate_results["FLIGHT_TRANSPORT"]["passed"] is False
    assert context.progress_events[-1]["stage"] == "execution_quality_gate"
    assert "wrong_departure_date" in context.progress_events[-1]["message"]
