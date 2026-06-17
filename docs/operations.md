# Operations

## Modes

Default mode is offline-safe. If `QWEN_API_KEY` is unset, structured agent calls return deterministic fallback data and bureau outputs are marked as estimates that did not use real-time data. If live structured calls are configured but fail, itinerary synthesis falls back to deterministic output where a local fallback exists.

Live mode is opt-in. Set `QWEN_API_KEY` and any tool keys you need, then set `RUN_LIVE_TESTS=1` only when you intentionally want tests marked `live` to call real services.

## Environment

- `QWEN_API_KEY`: Qwen compatible API key. Optional for offline mode.
- `QWEN_MODEL`: Chat model name. Defaults to `qwen-plus`.
- `QWEN_BASE_URL`: OpenAI-compatible Qwen endpoint.
- `AMAP_API_KEY`: Optional AMap MCP key.
- `SERPAPI_API_KEY`: Optional SerpAPI key.
- `OUTPUT_DIR`: artifact output directory. Defaults to `artifacts`.
- `SESSION_STORE_DIR`: JSON session directory. Defaults to `artifacts/sessions`.
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

## Artifacts And Sessions

Generated Markdown and iCalendar files are written under `OUTPUT_DIR` or the configured artifact directory. New planning requests require request IDs to match `[A-Za-z0-9_.-]{1,100}`. Resume, dashboard, download, and session lookups reject unsafe IDs instead of mapping them to sanitized aliases.

Offline budget fallback estimates activities, accommodation, food, local transport, origin-destination transport, and incidentals from trip length, travelers, and budget level. These are planning estimates, not live prices.

Sessions are stored as JSON files under `SESSION_STORE_DIR`. Writes use a temporary file and atomic replacement. If a stored session cannot be parsed or validated, resume/dashboard APIs return HTTP 409. A resumed request can recover the original request payload and current status, but in-memory LangGraph execution state is not checkpointed.
