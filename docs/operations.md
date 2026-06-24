# Operations

## Modes

Default mode is offline-safe. If `QWEN_API_KEY` is unset, structured agent calls return deterministic fallback data and bureau outputs are marked as estimates that did not use real-time data. If live structured calls are configured but fail, itinerary synthesis falls back to deterministic output where a local fallback exists.

Live mode is opt-in. Set `QWEN_API_KEY` and any tool keys you need, then set `RUN_LIVE_TESTS=1` only when you intentionally want tests marked `live` to call real services.

Human-intervention mode is API-safe by default. `ENABLE_LANGGRAPH_INTERRUPTS=0`
means the HTTP service returns a `HUMAN_INTERVENE` response with `resume_mode`
and `resume_state` instead of waiting indefinitely for an interactive
checkpoint. New sessions should resume from `resume_mode=boundary`; legacy or
incompatible sessions may report `resume_mode=replay` with a replay reason.

## Environment

- `QWEN_API_KEY`: Qwen compatible API key. Optional for offline mode.
- `QWEN_MODEL`: Chat model name. Defaults to `qwen-plus`.
- `QWEN_BASE_URL`: OpenAI-compatible Qwen endpoint.
- `AMAP_API_KEY`: Optional AMap MCP key.
- `SERPAPI_API_KEY`: Optional SerpAPI key.
- `OUTPUT_DIR`: artifact output directory. Defaults to `artifacts`.
- `SESSION_STORE_DIR`: JSON session directory. Defaults to `artifacts/sessions`.
- `SESSION_CACHE_MAX_ENTRIES`: maximum in-memory runtime session cache entries. Defaults to `500`.
- `SESSION_CACHE_TTL_SECONDS`: runtime session cache entry lifetime in seconds. Defaults to `86400` (24 hours).
- `ENABLE_LANGGRAPH_INTERRUPTS`: set to `1` only for an explicitly configured interrupt/checkpoint runtime. Defaults to `0` so the HTTP service returns resumable human-intervention responses instead of waiting indefinitely.
- `QWEN_TIMEOUT_SECONDS`: timeout in seconds for Qwen/OpenAI-compatible structured generation and ReAct MCP agent calls. Defaults to `60`.
- `RUN_LIVE_TESTS`: set to `1` to enable live tests.

## Tests

Default verification does not require API keys:

```bash
python -m py_compile main.py workflow.py utils/*.py provinces/*/*/*.py provinces/*/*.py
python -m pytest -q
```

Live tests are skipped unless explicitly enabled:

```bash
RUN_LIVE_TESTS=1 python -m pytest -q -m live
```

If `.env` already contains `RUN_LIVE_TESTS=1` and the provider keys, pytest
loads it before marker gating, so the same live suite can be run directly with
`python -m pytest -q -m live`. The live suite includes
`tests/test_live_config.py` for provider configuration and
`tests/test_live_e2e.py` for a complete `/plan` planning request against real
LLM/provider calls.

## Development Verification

Implementation work must follow the project constitution. Do not add production
branches, lookup tables, canned responses, or magic IDs solely to satisfy tests.
Deterministic fallbacks are acceptable only when they are documented product
behavior, depend on the request input, and are visible outside the test suite.

When changing framework, provider, or protocol behavior, use official
documentation, official examples, or upstream source as the implementation
reference. This applies to FastAPI, LangGraph, LangChain, Pydantic, Qwen or
OpenAI-compatible clients, MCP integrations, iCalendar output, and pytest
features.

After a new feature or behavior change, run a real end-to-end workflow/API
verification and inspect the observed result. The default offline check is:

```bash
python -m pytest tests/test_integration.py tests/test_main_api.py -q
```

If the change affects live LLM/MCP/provider behavior, also run the live suite
with credentials:

```bash
RUN_LIVE_TESTS=1 python -m pytest -q -m live
```

## Artifacts And Sessions

Generated Markdown and iCalendar files are written under `OUTPUT_DIR` or the configured artifact directory. New planning requests require request IDs to match `[A-Za-z0-9_.-]{1,100}`. Resume, dashboard, download, and session lookups reject unsafe IDs instead of mapping them to sanitized aliases.

Offline budget fallback estimates activities, accommodation, food, local transport, origin-destination transport, and incidentals from trip length, travelers, and budget level. These are planning estimates, not live prices.

Sessions are stored as JSON files under `SESSION_STORE_DIR`. Writes use a temporary file and atomic replacement. If a stored session cannot be parsed or validated, resume/dashboard APIs return HTTP 409. Runtime session cache entries are bounded by `SESSION_CACHE_MAX_ENTRIES` and `SESSION_CACHE_TTL_SECONDS`; cache eviction must not delete the persisted JSON session. The default service mode keeps LangGraph interrupts disabled for public HTTP requests so human intervention is represented as structured, resumable state.

Runtime cache cleanup happens before resume and dashboard reads. This keeps
stale in-memory sessions bounded while preserving persisted JSON sessions as
the durable lookup path.

## Verification Evidence

Report closure evidence is maintained in
`specs/001-resolve-report-issues/verification-evidence.md`. Each R-001 through
R-027 row records one of `fixed`, `already_fixed`,
`accepted_current_behavior`, or `deferred`, plus the command or rationale that
supports the decision. When live credentials are unavailable, provider-related
rows must record `deferred_no_credentials` and cite the offline E2E evidence
that was collected.
