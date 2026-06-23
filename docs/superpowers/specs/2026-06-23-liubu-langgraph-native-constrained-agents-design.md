# Liubu LangGraph-Native Constrained Agents Design

## Objective

Refactor the Liubu execution layer from loosely wrapped ReAct calls into LangGraph-native constrained specialist agents while preserving the existing public API, session format compatibility, final package model, and offline-safe behavior.

## Current Problem

The current Liubu implementation dispatches five bureau nodes through LangGraph, but each bureau internally runs a fixed mini-workflow:

1. ingest payload
2. run broad MCP research through `run_react_mcp_task`
3. synthesize a Pydantic output with `with_structured_output`
4. fall back to estimates on failure

This leaves the outer graph blind to the specialist's internal tool calls, tool arguments, evidence, retries, and quality failures. It also lets the model invent key travel parameters during tool use, such as hotel dates or traveler counts. The result is agent-like flexibility without enough graph-visible control.

## Target Architecture

Liubu remains an orchestrator-worker stage. Shangshu still dispatches Weather, Budget, Accommodation, Flight Transport, and Calendar workers in parallel with LangGraph `Send`.

Each Liubu bureau becomes a LangGraph-native worker subgraph with explicit state:

1. `prepare_context`
2. `agent_reasoning`
3. `tool_execution`
4. `agent_reasoning` loop until no more tool calls or a bounded step limit is reached
5. `evidence_validation`
6. `structured_result`
7. `quality_gate`

The worker graph owns a typed state object containing normalized request data, the approved draft, review warnings, bureau task requirements, allowed tools, tool evidence, validation findings, fallback status, and final result.

## Agent Autonomy Boundary

The agent keeps autonomy over research strategy inside its bureau:

- It may decide which allowed tools to call.
- It may decide whether it needs follow-up searches.
- It may compare alternatives and choose recommendations.
- It may synthesize explanations and warnings.

The graph constrains non-negotiable facts:

- Trip dates, traveler counts, currency, origin, destination, airport codes, budget cap, and user constraints are injected from normalized state.
- Tool calls that conflict with these facts are rejected before execution.
- A bureau cannot call tools outside its allowlist.
- A bureau cannot produce `status=ok` unless its quality gate passes.
- If live evidence is unavailable, the result must be labeled `fallback_estimate` or `unavailable`.

This is constrained agency, not deterministic scripting.

## Worker State Contracts

Create a shared Liubu state module with these concepts:

- `LiubuWorkerInput`: request id, target bureau, approved draft, profile, review payload, bureau task, and constraints.
- `LiubuToolEvidence`: tool name, sanitized args, status, compact result, error, and data source.
- `LiubuValidationFinding`: severity, code, message, field path, and evidence reference.
- `LiubuWorkerState`: input, messages, tool evidence, validation findings, result, status, data source, and warnings.

Each bureau may extend the shared state with bureau-specific fields only when necessary.

## Tool Execution Design

Use LangGraph `ToolNode` or a local equivalent node that exposes tool calls as graph state.

The tool node must:

- Filter tools by bureau allowlist.
- Validate tool arguments against profile and draft constraints before execution.
- Sanitize API keys and long payloads before adding evidence to state.
- Preserve tool errors as evidence instead of throwing broad exceptions.
- Support timeouts and bounded max tool steps.

Initial allowlists:

- Weather: Amap weather and geocode tools.
- Accommodation: SerpApi hotels, maps, local places, Amap POI tools.
- Flight Transport: SerpApi flights, maps directions, Amap directions/geocode tools.
- Budget: cost context tools, maps/local places, and prices from sibling bureau evidence when available.
- Calendar: no external booking tools required; uses approved itinerary and location normalization tools only.

## Quality Gates

Each bureau gets a gate that validates output before Shangshu receives it.

Shared gate rules:

- Output validates against the existing Pydantic execution result model.
- `status=ok` requires at least one successful evidence item for live-dependent bureaus.
- Dates in output and tool arguments must match trip dates or be explicitly marked as estimates.
- Traveler count and currency must match profile.
- Warnings must be present when any evidence item failed or timed out.
- Fallback outputs must include `data_source=fallback_estimate` or `data_source=unavailable`.

Bureau-specific gate rules:

- Flight Transport: live flight result must use profile airport codes and trip date; otherwise fallback or human intervention.
- Accommodation: live hotel result must use future check-in/check-out derived from profile dates.
- Budget: total estimate must include currency and line items for activities, accommodation, food, transport, flights, and buffer.
- Weather: forecast days must correspond to itinerary dates, or warnings must explain unavailable future weather.
- Calendar: every generated event must derive from an approved itinerary activity and refuse current-time fallback.

## Shangshu Assembly Policy

Shangshu should distinguish completion states:

- `DONE`: all required Liubu results passed their quality gates.
- `DONE_WITH_WARNINGS`: non-critical bureau outputs fell back but are transparently labeled.
- `HUMAN_INTERVENE`: a required live-dependent bureau cannot produce acceptable output and user/operator input is required.
- `REJECTED`: Liubu result violates non-negotiable constraints and cannot be safely assembled.

For compatibility, this migration keeps the existing public API status values and records Liubu gate outcomes as internal metadata and warnings. Exposing `DONE_WITH_WARNINGS` as a public status is out of scope for this migration and requires a separate API contract change.

## Migration Strategy

Implement incrementally:

1. Add shared Liubu worker state, evidence, and validation types.
2. Add a constrained tool execution helper with argument validation and evidence capture.
3. Convert Flight Transport first because it has the clearest hard constraints: airport codes, dates, adults, currency.
4. Convert Accommodation second because it has the known date hallucination failure.
5. Convert Calendar, Weather, and Budget after the pattern is proven.
6. Add a Shangshu Liubu quality aggregation step that records gate metadata and refuses silent success for invalid required bureau outputs.

During migration, keep existing bureau result schemas and final package fields unchanged.

## Testing Strategy

Use TDD for each slice.

Required tests:

- Worker state normalization preserves profile dates, airport codes, traveler counts, currency, interests, and constraints.
- Tool argument validation rejects past hotel dates, wrong traveler count, wrong currency, wrong airport code, and unsupported tools.
- Evidence capture records successful tool calls, errors, timeouts, and sanitized args.
- Flight Transport converted worker can pass with valid fake tool evidence and falls back with explicit warnings when evidence fails.
- Accommodation converted worker rejects agent-generated dates that conflict with profile dates.
- Shangshu aggregation records quality gate failures and does not silently treat invalid required bureau results as clean success.
- Existing offline tests continue passing.
- Live E2E continues reaching a terminal state and includes evidence/fallback labels.

## Non-Goals

- Do not replace FastAPI routes.
- Do not change the final package schema in the first migration.
- Do not remove offline deterministic fallback.
- Do not require live credentials for default tests.
- Do not rewrite Zhongshu or Menxia except where Liubu gate metadata needs to be consumed.
- Do not migrate the project from LangGraph to OpenAI Agents SDK in this phase.

## Risks

- ToolNode integration may require adapting MCP LangChain tools to the exact message/tool-call format expected by the graph.
- Live provider behavior can still be slow or unavailable, so quality gates must preserve transparent fallback paths.
- Introducing stricter gates can change successful `DONE` cases into warning or intervention cases. The first implementation should record metadata before changing public status semantics.

## Acceptance Criteria

- Liubu specialist execution exposes graph-visible tool evidence for at least Flight Transport and Accommodation.
- Agent-generated tool arguments are validated before external calls.
- Known bad hotel dates such as `2023-10-01` cannot reach SerpApi in live flow.
- Required fallback warnings explain whether the cause was timeout, provider error, validation failure, or unavailable credentials.
- Existing public API response fields remain compatible.
- Full offline pytest suite passes.
- Live E2E passes or records a precise provider failure without leaking secrets or throwing raw tool tracebacks.
