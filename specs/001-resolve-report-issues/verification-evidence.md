# Verification Evidence: Resolve Reported Workflow Issues

Last updated: 2026-06-22

This file records closure evidence for report issues R-001 through R-027. The
matrix uses the status values from `data-model.md`:

- `fixed`: implementation changed during this feature and has verification evidence.
- `already_fixed`: behavior was already fixed before this feature phase and now has recorded evidence.
- `accepted_current_behavior`: current behavior is intentionally accepted with a reason.
- `deferred`: remaining work is intentionally deferred with a reason.

Live validation status uses `passed`, `deferred_no_credentials`, or
`not_applicable`. Live provider validation was run with credentials from `.env`
on 2026-06-22.

## Closure Matrix

| Issue | Status | Decision Reason | Evidence | Live Validation |
|-------|--------|-----------------|----------|-----------------|
| R-001 | already_fixed | Shared bureau error boundaries preserve inspectable workflow output when one bureau fails. | Command: python -m pytest tests/test_bureaus.py tests/test_workflow.py -q. Covered by bureau failure fallback and Liubu node fallback tests. | not_applicable |
| R-002 | fixed | Planning route failures now return structured API errors instead of raw internal exceptions. | Command: python -m pytest tests/test_main_api.py -q. Covered by structured 500 and validation mapping tests. | not_applicable |
| R-003 | fixed | Request-scoped planner instances and shared bounded cache prevent cross-request mutable state leakage. | Command: python -m pytest tests/test_main_api.py tests/test_workflow.py tests/test_session_cache.py -q. Covered by concurrent request isolation and workflow instance isolation tests. | not_applicable |
| R-004 | fixed | New human-intervention sessions persist boundary resume metadata and legacy replay is labeled. | Command: python -m pytest tests/test_resume_flow.py tests/test_workflow.py tests/test_main_api.py -q. Covered by boundary resume and replay labeling tests. | not_applicable |
| R-005 | fixed | Runtime sessions use configurable capacity and TTL while persisted sessions remain the recovery path. | Command: python -m pytest tests/test_session_cache.py tests/test_main_api.py -q. Covered by cache eviction, TTL, and API-created cache tests. | not_applicable |
| R-006 | fixed | Default HTTP mode returns structured human-intervention responses instead of waiting indefinitely. | Command: python -m pytest tests/test_resume_flow.py tests/test_main_api.py -q. Covered by human-intervention and resume metadata responses. | not_applicable |
| R-007 | fixed | Empty Liubu dispatch is rejected before fanout. | Command: python -m pytest tests/test_workflow.py tests/test_orchestrator.py -q. Covered by empty task and invalid target tests. | not_applicable |
| R-008 | fixed | Targeted and end-to-end tests now cover concurrency, fanout merging, resume, provider fallback, resource pressure, and a complete live planning flow. | Command: python -m pytest tests/test_session_cache.py tests/test_resume_flow.py tests/test_main_api.py tests/test_workflow.py tests/test_orchestrator.py tests/test_zhongshu.py tests/test_menxia.py tests/test_agent_runtime.py tests/test_bureaus.py tests/test_offline_fallbacks.py tests/test_live_config.py tests/test_docs_examples.py -q. Live commands: python -m pytest tests/test_live_e2e.py -q; python -m pytest -q -m live. | passed |
| R-009 | fixed | Generic offline itinerary placeholders are rejected before final package approval. | Command: python -m pytest tests/test_menxia.py tests/test_offline_fallbacks.py tests/test_integration.py -q. Covered by generic placeholder rejection and offline label tests. | not_applicable |
| R-010 | fixed | Calendar and travel artifacts use itinerary dates or structured fallback/error evidence. | Command: python -m pytest tests/test_bureaus.py tests/test_integration.py -q. Covered by calendar date and artifact label assertions. | not_applicable |
| R-011 | already_fixed | Accommodation fallback duration derives from itinerary duration rather than a fixed stay length. | Command: python -m pytest tests/test_bureaus.py -q. Covered by accommodation fallback duration behavior. | not_applicable |
| R-012 | fixed | Calendar output records explicit timezone/date behavior and avoids naive current-time fallback. | Command: python -m pytest tests/test_bureaus.py -q. Covered by iCalendar date and timezone assertions. | not_applicable |
| R-013 | already_fixed | Calendar structured output uses a stable event-list wrapper. | Command: python -m pytest tests/test_bureaus.py tests/test_integration.py -q. Covered by calendar artifact generation and model-shape tests. | not_applicable |
| R-014 | fixed | Missing destination preferences are rejected before drafting. | Command: python -m pytest tests/test_zhongshu.py tests/test_main_api.py -q. Covered by missing destination and request validation tests. | not_applicable |
| R-015 | fixed | Menxia review rejects structurally complete but generic itinerary drafts. | Command: python -m pytest tests/test_menxia.py tests/test_integration.py -q. Covered by placeholder-quality review tests. | not_applicable |
| R-016 | accepted_current_behavior | Acceptance reason: configured rejection limits remain product-visible through structured rejection fields and no product owner requirement expanded retry count in this feature. | Command: python -m pytest tests/test_menxia.py tests/test_integration.py -q. Structured rejection and retry metadata remain visible. | not_applicable |
| R-017 | fixed | Liubu fanout uses independent context snapshots so bureau-local mutations cannot affect siblings or unrelated sessions. | Command: python -m pytest tests/test_workflow.py -q. Covered by fanout context isolation and merge tests. | not_applicable |
| R-018 | fixed | Structured generation fallback failures preserve inspectable exception cause information. | Command: python -m pytest tests/test_agent_runtime.py tests/test_offline_fallbacks.py -q. Covered by exception-cause and fallback tests. Live command includes tests/test_live_e2e.py through python -m pytest -q -m live. | passed |
| R-019 | fixed | Provider failure causes are preserved when fallback handling also fails. | Command: python -m pytest tests/test_agent_runtime.py -q. Covered by structured synthesis exception-cause tests. Live command includes tests/test_live_e2e.py through python -m pytest -q -m live. | passed |
| R-020 | fixed | Streaming and non-streaming planning paths share execution helpers and equivalent final status semantics. | Command: python -m pytest tests/test_main_api.py -q. Covered by streaming equivalence and labeled resume stream tests. | not_applicable |
| R-021 | fixed | Unsupported bureau role values are validated before target resolution can crash. | Command: python -m pytest tests/test_orchestrator.py tests/test_workflow.py -q. Covered by invalid Liubu target tests. | not_applicable |
| R-022 | fixed | Bureau enrichment failures are logged or surfaced through structured status and warnings. | Command: python -m pytest tests/test_bureaus.py tests/test_offline_fallbacks.py -q. Covered by bureau status and warning tests. Live command includes tests/test_live_e2e.py through python -m pytest -q -m live. | passed |
| R-023 | deferred | Deferral reason: duplicated package role labels are low-risk maintainability work and no user-visible drift remains after data-source rendering tests. | Command: python -m pytest tests/test_integration.py tests/test_main_api.py -q. Phase 8 keeps a cross-cutting consistency check open. | not_applicable |
| R-024 | deferred | Deferral reason: remaining direct workflow state accesses are low-priority hardening candidates and current high-risk paths have structured tests. | Command: python -m pytest tests/test_workflow.py tests/test_main_api.py -q. Phase 8 keeps consistency review open. | not_applicable |
| R-025 | deferred | Deferral reason: unused itinerary behavior cleanup is maintainability-only and unrelated to the report closure behavior delivered in Phases 1-7. | Command: python -m pytest tests/test_zhongshu.py tests/test_integration.py -q. Phase 8 keeps placeholder search and consistency checks open. | not_applicable |
| R-026 | already_fixed | Impossible trip date ranges are rejected by shared request validation before workflow execution. | Command: python -m pytest tests/test_main_api.py -q. Covered by schema and API invalid date range tests. | not_applicable |
| R-027 | fixed | Qwen/OpenAI-compatible structured generation has configurable timeout through settings and documented environment keys. | Command: python -m pytest tests/test_live_config.py tests/test_agent_runtime.py -q. Live command includes tests/test_live_config.py and tests/test_live_e2e.py through python -m pytest -q -m live. | passed |

## Verification Runs

Fresh Phase 7 verification results are recorded here after T092 through T096
run in this workspace.

| Command | Observed Result |
|---------|-----------------|
| python -m pytest tests/test_session_cache.py tests/test_resume_flow.py tests/test_main_api.py tests/test_workflow.py tests/test_orchestrator.py tests/test_zhongshu.py tests/test_menxia.py tests/test_agent_runtime.py tests/test_bureaus.py tests/test_offline_fallbacks.py tests/test_live_config.py tests/test_docs_examples.py -q | 82 passed, 1 skipped in 2.10s on 2026-06-22 with `RUN_LIVE_TESTS=0` for offline verification |
| python -m pytest -q | 94 passed, 2 skipped in 3.40s on 2026-06-22 with `RUN_LIVE_TESTS=0` for offline verification |
| python -m py_compile main.py workflow.py utils/agent_runtime.py utils/llm_factory.py utils/schemas.py utils/session_store.py utils/settings.py | Exit code 0 on 2026-06-22 |
| git diff --check | Exit code 0 with no whitespace errors on 2026-06-22 |
| python -m pytest tests/test_live_e2e.py -q | 1 passed in 134.29s on 2026-06-22 using credentials loaded from `.env`; exercised a complete live planning flow through `POST /plan` |
| python -m pytest -q -m live | 2 passed, 94 deselected in 235.72s on 2026-06-22 using credentials loaded from `.env`; covered `tests/test_live_config.py` and `tests/test_live_e2e.py` |
| python -m pytest tests/test_docs_examples.py -q | 8 passed on 2026-06-22 after live E2E, contract, data-model, and evidence consistency checks |
| rg -n "Placeholder Air\|verified against historical avg\|interest-focused visit\|estimated attraction\|explore local area\|free time" provinces utils workflow.py main.py | Only matched Menxia rejection text for `estimated attraction slot`; no final itinerary output path contains generic placeholder content |
| rg -n "request_id\s*==\|request\.request_id\s*==\|magic\|canned\|lookup table\|test-only\|test only\|equiv_\|cache_evict_\|stream_iso_" main.py workflow.py utils provinces | No production matches on 2026-06-22 |
| python -m pytest tests/test_main_api.py::test_download_markdown_artifact_preserves_data_source_labels tests/test_main_api.py::test_download_calendar_artifact_preserves_data_source_labels tests/test_integration.py::test_final_package_and_artifacts_expose_data_source_labels -q | 3 passed on 2026-06-22; verified `/download/{request_id}` content types and generated Markdown/iCalendar labels |
| python -m pytest tests/test_main_api.py::test_plan_stream_matches_non_streaming_for_success_human_and_error tests/test_main_api.py::test_resume_stream_emits_labeled_resume_events -q | 2 passed on 2026-06-22; verified streaming and non-streaming final status semantics |
| python -m pytest tests/test_integration.py tests/test_main_api.py -q | 23 passed on 2026-06-22; final real offline E2E validation for Phase 8 |
