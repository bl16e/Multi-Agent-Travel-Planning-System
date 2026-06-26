from pathlib import Path


FEATURE_DIR = Path("specs/001-resolve-report-issues")
EVIDENCE_FILE = FEATURE_DIR / "verification-evidence.md"
ENV_KEYS = {
    "SESSION_CACHE_MAX_ENTRIES",
    "SESSION_CACHE_TTL_SECONDS",
    "LANGGRAPH_CHECKPOINT_DB",
    "ENABLE_LANGGRAPH_INTERRUPTS",
    "QWEN_TIMEOUT_SECONDS",
    "RUN_LIVE_TESTS",
}
TARGETED_TEST_FILES = {
    "tests/test_session_cache.py",
    "tests/test_resume_flow.py",
    "tests/test_main_api.py",
    "tests/test_workflow.py",
    "tests/test_orchestrator.py",
    "tests/test_zhongshu.py",
    "tests/test_menxia.py",
    "tests/test_agent_runtime.py",
    "tests/test_bureaus.py",
    "tests/test_offline_fallbacks.py",
    "tests/test_live_config.py",
    "tests/test_docs_examples.py",
    "tests/test_official_cleanup.py",
}
LIVE_TEST_FILES = {
    "tests/test_live_config.py",
    "tests/test_live_e2e.py",
}
STATUS_VALUES = {"fixed", "already_fixed", "accepted_current_behavior", "deferred"}
LIVE_VALUES = {"passed", "deferred_no_credentials", "not_applicable"}


def _read(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _evidence_rows() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for line in _read(EVIDENCE_FILE).splitlines():
        if not line.startswith("| R-"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        issue, status, reason, evidence, live_validation = cells[:5]
        rows[issue] = {
            "status": status,
            "reason": reason,
            "evidence": evidence,
            "live_validation": live_validation,
        }
    return rows


def test_official_rewrite_cleanup_log_records_removed_surfaces():
    cleanup = Path("docs/official-rewrite-cleanup.md").read_text(encoding="utf-8")

    for phrase in (
        "utils/state_machine.py",
        "utils/permission_matrix.py",
        "utils/mcp_tools.py",
        "LangGraph checkpointer",
        "Command(resume=...)",
        "MultiServerMCPClient",
    ):
        assert phrase in cleanup


def test_new_runtime_environment_variables_are_documented():
    env_example = _read(".env.example")
    operations = _read("docs/operations.md")
    quickstart = _read(FEATURE_DIR / "quickstart.md")

    for key in ENV_KEYS:
        assert f"{key}=" in env_example
        assert key in operations
        assert key in quickstart


def test_quickstart_targeted_command_lists_focused_test_files():
    quickstart = _read(FEATURE_DIR / "quickstart.md")

    for test_file in TARGETED_TEST_FILES:
        assert test_file in quickstart


def test_live_e2e_command_documents_complete_planning_flow():
    quickstart = _read(FEATURE_DIR / "quickstart.md")
    operations = _read("docs/operations.md")
    evidence = _read(EVIDENCE_FILE)

    for test_file in LIVE_TEST_FILES:
        assert test_file in quickstart
    assert "tests/test_live_e2e.py" in operations
    assert "tests/test_live_e2e.py" in evidence
    assert "complete live planning flow" in quickstart.lower()


def test_verification_evidence_has_one_entry_for_every_report_issue():
    rows = _evidence_rows()

    assert set(rows) == {f"R-{index:03d}" for index in range(1, 28)}
    for issue, row in rows.items():
        assert row["status"] in STATUS_VALUES, issue
        assert row["reason"], issue
        assert row["evidence"], issue
        assert row["live_validation"] in LIVE_VALUES, issue


def test_unresolved_or_accepted_evidence_has_command_or_deferral_reason():
    rows = _evidence_rows()

    for issue, row in rows.items():
        if row["status"] not in {"accepted_current_behavior", "deferred"}:
            continue
        evidence = row["evidence"].lower()
        reason = row["reason"].lower()
        assert "python -m pytest" in evidence or "git diff --check" in evidence or "deferr" in reason, issue


def test_api_contract_covers_actual_response_fields():
    contract = _read(FEATURE_DIR / "contracts/api-contract.yaml")

    for field in (
        "booking_links",
        "packing_list",
        "progress_events",
        "generated_at",
        "resume_state",
    ):
        assert field in contract


def test_data_model_covers_persisted_session_fields():
    data_model = _read(FEATURE_DIR / "data-model.md")
    session_store = _read("utils/session_store.py")

    for field in (
        "schema_version",
        "request_id",
        "request",
        "status",
        "context_snapshot",
        "result",
        "package",
        "resume_state",
        "resume_mode",
    ):
        assert field in data_model
        assert field in session_store
