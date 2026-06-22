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
