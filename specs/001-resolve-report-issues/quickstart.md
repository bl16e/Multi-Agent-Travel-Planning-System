# Quickstart: Resolve Reported Workflow Issues

This guide defines validation scenarios for the report-closure feature. It is a
run guide, not an implementation task list.

## Prerequisites

1. Install project dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Ensure default offline mode unless intentionally validating live providers:

   ```bash
   copy .env.example .env
   ```

   The default offline verification profile uses:

   - `SESSION_CACHE_MAX_ENTRIES=500`
   - `SESSION_CACHE_TTL_SECONDS=86400`
   - `ENABLE_LANGGRAPH_INTERRUPTS=0`
   - `QWEN_TIMEOUT_SECONDS=60`
   - `RUN_LIVE_TESTS=0`

3. For live validation, set provider credentials and enable live tests:

   ```bash
   set RUN_LIVE_TESTS=1
   set QWEN_API_KEY=<your key>
   ```

## Scenario 1: Offline E2E Workflow/API Evidence

Run:

```bash
python -m pytest tests/test_integration.py tests/test_main_api.py -q
```

Expected outcome:

- A representative planning workflow completes as `DONE`, `HUMAN_INTERVENE`,
  or `REJECTED` without an unhandled exception.
- API routes return structured statuses/errors.
- Streaming endpoints emit `progress`, `result` or `error`, and `done`.
- Streaming and non-streaming routes report equivalent final status semantics
  for successful, human-intervention, and error outcomes.
- Invalid trip date ranges are rejected before workflow execution with a clear
  structured error.

## Scenario 2: Targeted Report Closure Tests

Run the targeted suite after implementation:

```bash
python -m pytest tests/test_session_cache.py tests/test_resume_flow.py tests/test_main_api.py tests/test_workflow.py tests/test_orchestrator.py tests/test_zhongshu.py tests/test_menxia.py tests/test_agent_runtime.py tests/test_bureaus.py tests/test_offline_fallbacks.py tests/test_live_config.py tests/test_docs_examples.py -q
```

Expected outcome:

- Runtime sessions evict after 500 entries or 24-hour TTL while persisted
  sessions remain available.
- New human-intervention sessions resume from boundary state.
- Legacy/incompatible sessions label replay behavior.
- Generic placeholder itinerary drafts are rejected before final package
  approval.
- Invalid Liubu dispatch, missing destination, invalid dates, corrupt sessions,
  and provider failures produce inspectable outcomes.
- Fallback/live data-source labels are visible in API/dashboard/artifact data.
- `/download/{request_id}` returns generated Markdown or iCalendar artifacts
  whose content-type and data-source labels match the final package.
- The closure evidence file maps every R-001 through R-027 report issue to
  fixed, already-fixed, accepted_current_behavior, or deferred status.
- Liubu fanout context isolation is verified so bureau-local mutations cannot
  affect sibling bureaus or unrelated sessions.

## Scenario 3: Full Offline Suite

Run:

```bash
python -m pytest -q
```

Expected outcome:

- All non-live tests pass.
- Live tests are skipped unless `RUN_LIVE_TESTS=1`.
- No production behavior depends on magic request IDs or canned test-only data.

## Scenario 4: Compile And Hygiene

Run:

```powershell
Get-ChildItem -Recurse -Filter *.py | Where-Object { $_.FullName -notmatch '\\.venv|\\__pycache__|\\.pytest_cache' } | ForEach-Object { python -m py_compile $_.FullName }
git diff --check
```

Expected outcome:

- Python compilation exits with code 0.
- Diff hygiene has no whitespace errors.

## Scenario 5: Live Provider Evidence

When credentials are available, run:

```bash
RUN_LIVE_TESTS=1 python -m pytest -q -m live
```

With `RUN_LIVE_TESTS=1` and provider keys already present in `.env`, the same
suite can also be run directly:

```bash
python -m pytest -q -m live
```

The live suite includes:

- `tests/test_live_config.py` for Qwen/OpenAI-compatible client configuration.
- `tests/test_live_e2e.py` for a complete live planning flow through
  `POST /plan`.

Expected outcome:

- Live Qwen/OpenAI-compatible client configuration is usable.
- The complete live planning flow reaches a final `DONE`, `REJECTED`, or
  `HUMAN_INTERVENE` response without unhandled exceptions.
- A `DONE` live response includes real structured LLM evidence on the review
  and bureau outputs, plus itinerary, calendar, and progress data.
- Provider-related fixes have live evidence from both configuration and
  product-path `/plan` execution.

If credentials are unavailable, the completion report must explicitly say live
validation was deferred and include the offline E2E evidence from Scenario 1.

## Scenario 6: Manual Artifact Inspection

After a successful full workflow run, inspect:

- Dashboard snapshot for `resume_mode`, status, pending inputs, and progress.
- Markdown output under `OUTPUT_DIR`.
- iCalendar output under `OUTPUT_DIR`.
- API download response from `/download/{request_id}` for both Markdown and
  iCalendar variants when available.
- Persisted JSON session under `SESSION_STORE_DIR`.
- Report issue closure matrix in
  `specs/001-resolve-report-issues/verification-evidence.md`.

Expected outcome:

- Fallback and estimated content is clearly labeled.
- Generic placeholder drafts do not appear in approved final packages.
- Calendar dates match itinerary dates.
- iCalendar output uses itinerary-derived dates with explicit timezone behavior,
  and missing dates produce structured fallback/error evidence rather than
  silently using the current time.
- Downloaded artifacts match generated files and preserve the same labels as
  API/dashboard data.
- Session status and resume mode match the observed workflow.
- Each R-001 through R-027 row has status, verification command or explicit
  deferral reason, and observed result.
