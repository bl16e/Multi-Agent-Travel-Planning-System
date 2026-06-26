# Multi-Agent Travel Planning System

LangGraph-based travel planning MVP using the "Three Provinces, Six Bureaus" workflow:

- Shangshu orchestrates state, review gates, user intervention, and final assembly.
- Zhongshu drafts an itinerary.
- Menxia reviews the draft before execution.
- Liubu bureaus handle weather, budget, accommodation, transport, and calendar output.

The project runs offline by default. If no Qwen or MCP credentials are configured, agents return deterministic fallback results and clearly mark estimates as not real-time data.

## Quick Start

```bash
python -m pip install -r requirements.txt
copy .env.example .env
python -m uvicorn main:app --reload
```

Open http://127.0.0.1:8000.

## Configuration

The main environment variables are:

- `QWEN_API_KEY`
- `QWEN_MODEL`
- `QWEN_BASE_URL`
- `AMAP_API_KEY`
- `SERPAPI_API_KEY`
- `OUTPUT_DIR`
- `SESSION_STORE_DIR`
- `LANGGRAPH_CHECKPOINT_DB`
- `QWEN_TIMEOUT_SECONDS`
- `PLAN_REQUEST_TIMEOUT_SECONDS`
- `MCP_TOOLING_TIMEOUT_SECONDS`
- `LIUBU_TOOL_TIMEOUT_SECONDS`
- `RUN_LIVE_TESTS`

See [docs/operations.md](docs/operations.md) for operational details.

## API

- `GET /health`
- `POST /plan`
- `POST /plan/stream` (`text/event-stream`)
- `POST /resume/{request_id}`
- `POST /resume/{request_id}/stream` (`text/event-stream`)
- `GET /dashboard/{request_id}`
- `GET /download/{request_id}`

## Testing

Default tests are deterministic and do not require API keys:

```bash
python -m py_compile main.py workflow.py utils/*.py provinces/*/*/*.py provinces/*/*.py
python -m pytest -q
```

Live tests are opt-in:

```bash
RUN_LIVE_TESTS=1 python -m pytest -q -m live
```

## Outputs

The workflow writes Markdown and iCalendar artifacts under `OUTPUT_DIR`, stores JSON session metadata under `SESSION_STORE_DIR`, and stores durable LangGraph checkpoints in `LANGGRAPH_CHECKPOINT_DB`. New planning requests require `request_id` values to match `[A-Za-z0-9_.-]{1,100}` so API routes, sessions, checkpoints, and artifact filenames stay consistent.
