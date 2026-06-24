# Tasks: Resolve Open Report Issues

**Input**: Design documents from `specs/001-resolve-report-issues/`
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/api-contract.yaml`, `quickstart.md`

**Tests**: Required by constitution and feature spec. Each story includes tests before implementation tasks. Do not hardcode outputs only to satisfy tests; implementation must follow official documentation or local production patterns.

**Organization**: Tasks are grouped by user story so each story can be implemented and verified independently where practical.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel because it touches a different file and has no dependency on another open task
- **[Story]**: User story identifier such as US1, US2, US3, US4, US5
- Every task names the exact file path to edit or verify

## Phase 1: Setup

**Purpose**: Confirm planning inputs and prepare configuration surfaces before implementation.

- [X] T001 Review source decisions and official documentation links in `specs/001-resolve-report-issues/research.md` before changing runtime behavior.
- [X] T002 Review API response expectations in `specs/001-resolve-report-issues/contracts/api-contract.yaml` before editing endpoint serializers.
- [X] T003 Review data ownership fields in `specs/001-resolve-report-issues/data-model.md` before changing persisted session format.
- [X] T004 Add new runtime environment keys to `.env.example` for session cache size, session cache TTL, LangGraph interrupt behavior, and Qwen request timeout.
- [X] T005 Update configuration guidance in `docs/operations.md` for the new runtime environment keys from `.env.example`.

---

## Phase 2: Foundational Infrastructure

**Purpose**: Shared models, settings, and cache primitives needed by all stories.

- [X] T006 [P] Add unit tests for bounded cache insertion, lookup, expiry, eviction, and deletion in `tests/test_session_cache.py`.
- [X] T007 [P] Add settings tests for cache defaults, cache overrides, interrupt flag defaults, and Qwen timeout parsing in `tests/test_live_config.py`.
- [X] T008 [P] Add session persistence tests for `resume_state` and `resume_mode` round-tripping in `tests/test_resume_flow.py`.
- [X] T009 Implement bounded TTL cache primitives in `utils/session_cache.py` using standard library data structures.
- [X] T010 Wire cache configuration fields into `utils/settings.py` with defaults of 500 entries and 86,400 seconds TTL.
- [X] T011 Wire `enable_langgraph_interrupts` and `qwen_timeout_seconds` into `utils/settings.py` with documented defaults.
- [X] T012 Extend `StoredSession` in `utils/session_store.py` with `resume_state` and `resume_mode` while preserving older JSON compatibility.
- [X] T013 Add shared execution metadata fields to relevant result models in `utils/schemas.py`, including `status`, `data_source`, and warning collections where needed.
- [X] T014 Update serialization helpers in `utils/session_store.py` so new model fields persist without losing existing `context_snapshot`, `result`, or `package` data.
- [X] T015 Run foundational tests in `tests/test_session_cache.py`, `tests/test_live_config.py`, and `tests/test_resume_flow.py`.

**Checkpoint**: Runtime settings, cache behavior, and persisted session shape are available for story work.

---

## Phase 3: User Story 1 - Reliable Planning Requests (Priority: P1)

**Goal**: A planning request should either produce a structurally valid itinerary package or fail with clear errors, without leaking shared workflow state or silently skipping invalid work.

**Independent Test**: Submit a valid `/plan` request and verify the response contains itinerary, review, bureau outputs, calendar data, session metadata, and no generic placeholder itinerary content.

### Tests for User Story 1

- [X] T016 Add API test for per-request planner instance isolation in `tests/test_main_api.py`.
- [X] T017 Add API test for structured planning failure responses in `tests/test_main_api.py`.
- [X] T018 [P] Add workflow test for empty `liubu_tasks` routing failure in `tests/test_workflow.py`.
- [X] T019 [P] Add orchestrator tests for invalid, empty, and duplicate Liubu targets in `tests/test_orchestrator.py`.
- [X] T020 [P] Add itinerary tests for missing destination rejection and non-placeholder offline output in `tests/test_zhongshu.py`.
- [X] T021 [P] Add integration test assertions for full response structure in `tests/test_integration.py`.
- [X] T022 Add invalid trip date range API test that proves validation happens before workflow execution in `tests/test_main_api.py`.
- [X] T023 Add streaming vs non-streaming equivalence tests for success, human-intervention, and error outcomes in `tests/test_main_api.py`.

### Implementation for User Story 1

- [X] T024 Replace global mutable planner execution in `main.py` with request-scoped `ThreeProvinceTravelSystem` creation for `/plan`.
- [X] T025 Refactor `/plan`, `/plan/stream`, `/resume/{request_id}`, and `/resume/{request_id}/stream` in `main.py` to share one endpoint-safe execution helper.
- [X] T026 Replace unbounded `_shared_sessions` usage in `main.py` with `utils/session_cache.py`.
- [X] T027 Add explicit API error mapping in `main.py` for validation failures, missing sessions, resume conflicts, and internal planning failures.
- [X] T028 Add an empty-task guard in `_route_to_liubu` in `workflow.py` so no Liubu fanout silently succeeds with no bureaus.
- [X] T029 Harden `_resolve_liubu_targets` in `provinces/shangshu_orchestrator/orchestrator.py` for invalid names, duplicate values, and empty requested lists.
- [X] T030 Reject missing destination preferences in `provinces/zhongshu_itinerary/graph.py` instead of defaulting to a hardcoded city.
- [X] T031 Run story tests in `tests/test_main_api.py`, `tests/test_workflow.py`, `tests/test_orchestrator.py`, `tests/test_zhongshu.py`, and `tests/test_integration.py`.

**Checkpoint**: User Story 1 can be demonstrated through API and integration tests without depending on resume, live credentials, or documentation changes.

---

## Phase 4: User Story 2 - Transparent Resume Behavior (Priority: P1)

**Goal**: A traveler can resume an interrupted planning session from a saved boundary when possible, and the system labels replay behavior when boundary resume is unavailable.

**Independent Test**: Create a session that reaches human intervention, resume it with user input, and verify the saved boundary is used instead of restarting the complete workflow.

### Tests for User Story 2

- [X] T032 [P] Add workflow test for `resume_state` creation at human intervention in `tests/test_workflow.py`.
- [X] T033 Add system test for boundary resume path in `tests/test_resume_flow.py`.
- [X] T034 Add system test for legacy replay fallback labeling in `tests/test_resume_flow.py`.
- [X] T035 Add dashboard test for `resume_mode` and `resume_state` fields in `tests/test_main_api.py`.
- [X] T036 Add streaming resume test for labeled resume events in `tests/test_main_api.py`.
- [X] T037 Add contract coverage for resume response fields in `tests/test_main_api.py`.

### Implementation for User Story 2

- [X] T038 Extend `SystemState` in `workflow.py` with resume boundary fields needed by `data-model.md`.
- [X] T039 Update `_node_finish_human` in `workflow.py` to persist the next-node boundary, human question, and resumable state snapshot.
- [X] T040 Add a `resume` method in `workflow.py` that applies user payloads to a stored boundary before continuing execution.
- [X] T041 Update `ThreeProvinceTravelSystem.resume_trip` in `main.py` to use boundary resume when `resume_state` exists.
- [X] T042 Update `ThreeProvinceTravelSystem.resume_trip` in `main.py` to label legacy replay with `resume_mode` and `replay_reason` when boundary data is unavailable.
- [X] T043 Persist `resume_state` and `resume_mode` through `_persist_session` and `_load_session` in `main.py`.
- [X] T044 Add `resume_mode`, `resume_state`, and replay reason fields to dashboard snapshots in `main.py`.
- [X] T045 Run story tests in `tests/test_resume_flow.py`, `tests/test_workflow.py`, and `tests/test_main_api.py`.

**Checkpoint**: User Story 2 can be demonstrated without changing output labels or live credential behavior.

---

## Phase 5: User Story 3 - Honest Offline and Live Output Labels (Priority: P2)

**Goal**: Offline fallback, live API results, and failed optional enrichments are labeled clearly, and generic placeholder itineraries are rejected before final output.

**Independent Test**: Run with live credentials disabled and verify final output labels offline data sources, records skipped enrichments, and rejects generic placeholder itinerary content.

### Tests for User Story 3

- [X] T046 [P] Add fallback-label tests for offline Zhongshu, Shangshu, and Liubu outputs in `tests/test_offline_fallbacks.py`.
- [X] T047 [P] Add Menxia tests that reject generic itinerary items such as orientation walks and placeholder recommendations in `tests/test_menxia.py`.
- [X] T048 [P] Add structured synthesis exception-cause tests in `tests/test_agent_runtime.py`.
- [X] T049 Add bureau logging and status tests for weather, flight, budget, accommodation, and calendar failures in `tests/test_bureaus.py`.
- [X] T050 [P] Add Qwen timeout configuration tests in `tests/test_live_config.py`.
- [X] T051 Add final package data-source assertions in `tests/test_integration.py`.
- [X] T052 Add Markdown and iCalendar artifact data-source label assertions in `tests/test_integration.py`.
- [X] T053 [P] Add `/download/{request_id}` artifact label and content-type assertions in `tests/test_main_api.py`.
- [X] T054 Add calendar and iCalendar date/timezone assertions that prove itinerary dates are used and missing dates do not silently use current time in `tests/test_bureaus.py`.

### Implementation for User Story 3

- [X] T055 Update offline synthesis helpers in `utils/agent_runtime.py` so generated drafts contain explicit offline provenance and avoid generic placeholder activities.
- [X] T056 Preserve the original structured synthesis exception as the raised cause in `utils/agent_runtime.py`.
- [X] T057 Add Qwen request timeout wiring to `utils/llm_factory.py` using `qwen_timeout_seconds` from `utils/settings.py`.
- [X] T058 Add generic-placeholder detection to `provinces/menxia_review/graph.py` before approval decisions.
- [X] T059 Label review decisions in `provinces/menxia_review/graph.py` with data source and escalation warnings.
- [X] T060 Replace silent bureau exception swallowing with logged warnings and structured status fields in `provinces/liubu/weather/service.py`.
- [X] T061 Replace silent bureau exception swallowing with logged warnings and structured status fields in `provinces/liubu/flight_transport/service.py`.
- [X] T062 Replace silent bureau exception swallowing with logged warnings and structured status fields in `provinces/liubu/budget/service.py`.
- [X] T063 Replace silent bureau exception swallowing with logged warnings and structured status fields in `provinces/liubu/accommodation/service.py`.
- [X] T064 Replace silent bureau exception swallowing with logged warnings and structured status fields in `provinces/liubu/calendar/service.py`.
- [X] T065 Update calendar generation in `provinces/liubu/calendar/service.py` so iCalendar dates come from itinerary dates with explicit timezone behavior or a structured fallback/error status.
- [X] T066 Add data-source and warning labels to final package rendering in `workflow.py`.
- [X] T067 Add data-source labels to Markdown and iCalendar artifact generation in `workflow.py`.
- [X] T068 Ensure `/download/{request_id}` in `main.py` returns the labeled Markdown or iCalendar artifact that matches the final package status.
- [X] T069 Run story tests in `tests/test_offline_fallbacks.py`, `tests/test_menxia.py`, `tests/test_agent_runtime.py`, `tests/test_bureaus.py`, `tests/test_live_config.py`, `tests/test_integration.py`, and `tests/test_main_api.py`.

**Checkpoint**: User Story 3 can be demonstrated offline and does not depend on concurrent load tests.

---

## Phase 6: User Story 4 - Concurrent and Long-Running Request Safety (Priority: P2)

**Goal**: Multiple users can submit or stream planning requests without sharing mutable state, and stale sessions are bounded by configured cache policy.

**Independent Test**: Run two planning requests with different profiles and verify their sessions, streamed events, and cached state remain isolated.

### Tests for User Story 4

- [X] T070 Add concurrent request isolation test in `tests/test_main_api.py`.
- [X] T071 Add streaming session isolation test in `tests/test_main_api.py`.
- [X] T072 Add cache eviction integration test through API-created sessions in `tests/test_main_api.py`.
- [X] T073 Add workflow instance isolation test in `tests/test_workflow.py`.
- [X] T074 Add Liubu fanout context isolation and merge test proving bureau-local context mutations cannot affect sibling bureaus or unrelated sessions in `tests/test_workflow.py`.

### Implementation for User Story 4

- [X] T075 Ensure `ThreeProvinceTravelSystem` in `main.py` owns per-instance workflow and session state only.
- [X] T076 Ensure streamed planning in `main.py` records events against the correct request id in `utils/session_cache.py`.
- [X] T077 Ensure streamed resume in `main.py` records events against the correct request id in `utils/session_cache.py`.
- [X] T078 Add cache cleanup calls in `main.py` for expired entries before dashboard and resume reads.
- [X] T079 Isolate Liubu fanout context snapshots in `workflow.py` so each `Send` receives an independent context and merge logic preserves only intended bureau results.
- [X] T080 Run story tests in `tests/test_main_api.py`, `tests/test_workflow.py`, and `tests/test_session_cache.py`.

**Checkpoint**: User Story 4 can be demonstrated without live credentials.

---

## Phase 7: User Story 5 - Verification Evidence and Operations Clarity (Priority: P3)

**Goal**: Operators can verify fixes through documented offline and live commands and understand any skipped live checks.

**Independent Test**: Follow `quickstart.md` from a clean checkout and verify the documented commands produce the expected offline evidence; live commands are run only when credentials are available.

### Tests and Documentation for User Story 5

- [X] T081 Add documentation-example coverage for new environment variables in `tests/test_docs_examples.py`.
- [X] T082 Add quickstart command consistency checks for new test files in `tests/test_docs_examples.py`.
- [X] T083 Update verification commands in `specs/001-resolve-report-issues/quickstart.md` to include all new focused test files.
- [X] T084 Update runtime behavior notes in `docs/operations.md` for resume modes, offline labels, cache limits, and live test gating.
- [X] T085 Create verification evidence log template in `specs/001-resolve-report-issues/verification-evidence.md`.
- [X] T086 Add R-001 through R-027 from `specs/001-resolve-report-issues/spec.md` to the closure matrix in `specs/001-resolve-report-issues/verification-evidence.md`.
- [X] T087 Record already-fixed evidence for R-001, R-011, R-013, and R-026 in `specs/001-resolve-report-issues/verification-evidence.md`.
- [X] T088 Record `accepted_current_behavior` or deferred decisions for any unresolved report issue that remains intentionally unchanged in `specs/001-resolve-report-issues/verification-evidence.md`.
- [X] T089 Add documentation checks that every R-001 through R-027 entry has a status, evidence field, and decision reason in `tests/test_docs_examples.py`.
- [X] T090 Add documentation checks that every unresolved R-001 through R-027 issue links to at least one verification command or explicit deferral reason in `tests/test_docs_examples.py`.
- [X] T091 Run documentation tests in `tests/test_docs_examples.py`.

### Final Verification for User Story 5

- [X] T092 Run focused offline suite from `specs/001-resolve-report-issues/quickstart.md`.
- [X] T093 Run full offline suite with `python -m pytest -q` for repository root `D:\code\Multi-Agent-Travel-Planning-System`.
- [X] T094 Run bytecode compilation check with `python -m py_compile main.py workflow.py utils/agent_runtime.py utils/llm_factory.py utils/schemas.py utils/session_store.py utils/settings.py`.
- [X] T095 Run diff whitespace check with `git diff --check` for repository root `D:\code\Multi-Agent-Travel-Planning-System`.
- [X] T096 Run live E2E suite with `RUN_LIVE_TESTS=1 python -m pytest -q -m live` when credentials are present and record the result in `specs/001-resolve-report-issues/verification-evidence.md`.
- [X] T097 Record offline E2E results, targeted suite results, and any live-test deferral reason in `specs/001-resolve-report-issues/verification-evidence.md`.

**Checkpoint**: User Story 5 provides repeatable evidence for implementation acceptance.

---

## Phase 8: Polish and Cross-Cutting Consistency

**Purpose**: Ensure contracts, docs, and implementation remain aligned after all story work.

- [X] T098 Verify `specs/001-resolve-report-issues/contracts/api-contract.yaml` matches actual API response fields in `main.py`.
- [X] T099 Verify `specs/001-resolve-report-issues/data-model.md` matches persisted JSON fields from `utils/session_store.py`.
- [X] T100 Verify `specs/001-resolve-report-issues/spec.md` acceptance scenarios are covered by tests in `tests/test_integration.py` and `tests/test_main_api.py`.
- [X] T101 Verify no implementation contains generic placeholder final itinerary content by searching `provinces/`, `utils/`, and `workflow.py`.
- [X] T102 Verify no task-only hardcoded behavior was introduced by reviewing changed tests and implementation files together.
- [X] T103 Verify every R-001 through R-027 entry in `specs/001-resolve-report-issues/verification-evidence.md` has final evidence or a documented decision.
- [X] T104 Verify artifact download behavior by inspecting `/download/{request_id}` output and generated files under `OUTPUT_DIR`.
- [X] T105 Verify streaming and non-streaming final statuses match for success, human-intervention, and error cases in `tests/test_main_api.py`.
- [X] T106 Run final real E2E offline validation command from `specs/001-resolve-report-issues/quickstart.md`.
- [X] T107 Add and run a complete live planning-flow E2E through `POST /plan` in `tests/test_live_e2e.py`, then record the live result in `specs/001-resolve-report-issues/verification-evidence.md`.

---

## Dependencies and Execution Order

### Phase Dependencies

- Setup (Phase 1) must finish before Foundation (Phase 2).
- Foundation (Phase 2) must finish before all user stories because it defines shared settings, cache, and session fields.
- User Story 1 (Phase 3) and User Story 2 (Phase 4) are both P1 and should complete before P2 stories.
- User Story 3 (Phase 5) depends on User Story 1 output structure.
- User Story 4 (Phase 6) depends on Foundation and the endpoint refactor from User Story 1.
- User Story 5 (Phase 7) depends on completed implementation stories.
- Polish (Phase 8) runs after all stories.

### Story Dependencies

- **US1**: Requires Foundation only.
- **US2**: Requires Foundation; can run after or alongside US1 if endpoint refactor conflicts are coordinated in `main.py`.
- **US3**: Requires US1 response structure and Foundation metadata fields.
- **US4**: Requires Foundation cache and US1 endpoint helper refactor.
- **US5**: Requires all implementation stories.

### Parallel Examples

- After Phase 1, T006, T007, and T008 can run in parallel because they add or update different test files.
- During US1, T018, T019, T020, and T021 can run in parallel after the same-file `tests/test_main_api.py` tasks are coordinated.
- During US2, T032 can run in parallel with either the `tests/test_resume_flow.py` group or the `tests/test_main_api.py` group.
- During US3, T046, T047, T048, T050, and T053 can run in parallel; coordinate T049/T054 in `tests/test_bureaus.py` and T051/T052 in `tests/test_integration.py`.
- During US4, coordinate the `tests/test_main_api.py` group before running T073/T074 in `tests/test_workflow.py`.
- During US5, coordinate all `tests/test_docs_examples.py` tasks together before T091.

---

## Implementation Strategy

### MVP First

1. Complete Phase 1 and Phase 2.
2. Complete US1 and US2 because they address the highest-risk planning and resume behavior.
3. Run T031 and T045 to verify the MVP.
4. Confirm `/plan` and `/resume/{request_id}` behavior manually or through E2E tests before moving to P2 stories.

### Incremental Delivery

1. Deliver US1 for reliable request execution and explicit API errors.
2. Deliver US2 for transparent resume behavior.
3. Deliver US3 for honest offline/live labeling and placeholder rejection.
4. Deliver US4 for concurrency, cache safety, and Liubu fanout context isolation.
5. Deliver US5 for documented evidence and operational clarity.

### Quality Gates

- Tests must fail for the intended behavior before implementation when practical.
- Do not add lookup tables or hardcoded outputs solely to pass tests.
- Use official documentation decisions already captured in `research.md` when implementing framework-specific behavior.
- Run real offline E2E from `quickstart.md` after implementation changes.
- Run live E2E only when credentials are configured; otherwise record the concrete deferral reason.
