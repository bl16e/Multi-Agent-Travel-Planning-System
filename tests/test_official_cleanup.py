from pathlib import Path


HANDWRITTEN_LAYERS = (
    "utils/state_machine.py",
    "utils/permission_matrix.py",
    "utils/mcp_tools.py",
    "interactive_demo.py",
    "provinces/liubu/constrained/tools.py",
    "provinces/liubu/constrained/tool_node.py",
)


def test_handwritten_framework_replacements_are_removed():
    for path in HANDWRITTEN_LAYERS:
        assert not Path(path).exists(), f"{path} should be removed or replaced by an official integration"


def test_official_cleanup_log_records_rewrite_paths():
    log = Path("docs/official-rewrite-cleanup.md").read_text(encoding="utf-8")

    for phrase in (
        "LangGraph checkpointer",
        "Command(resume=...)",
        "LangGraph event streaming",
        "ToolNode",
        "MultiServerMCPClient",
        "Constrained Liubu tool executor",
    ):
        assert phrase in log


def test_official_cleanup_log_has_no_stale_liubu_placeholder_followups():
    log = Path("docs/official-rewrite-cleanup.md").read_text(encoding="utf-8")

    assert "Broaden live MCP coverage for Weather, Budget, and Calendar" not in log
    assert "remaining live research placeholders for Weather, Budget, and Calendar" not in log
