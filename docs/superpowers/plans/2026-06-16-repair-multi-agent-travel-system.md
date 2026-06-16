# Multi-Agent Travel Planning System Repair Plan

## Goal

Repair the current demonstrable multi-agent prototype into an installable, testable MVP that runs deterministically offline by default and can optionally use real LLM/MCP services.

## Scope

- Fix dependency and startup issues around LangChain/LangGraph agent creation.
- Keep default tests offline and deterministic.
- Add JSON-backed session persistence.
- Sanitize request IDs for artifact and session file paths.
- Make bureau fallbacks explicit estimates, not fake live data.
- Preserve the existing HTTP API surface.
- Update README, operations docs, environment example, and CI.

## Verification

- `python -m py_compile main.py workflow.py utils/*.py provinces/*/*/*.py provinces/*/*.py`
- `python -m pytest -q`
- `RUN_LIVE_TESTS=1 python -m pytest -q -m live` when live credentials are present.
