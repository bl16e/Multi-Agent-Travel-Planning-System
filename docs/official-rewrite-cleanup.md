# Official Rewrite Cleanup Log

This cleanup removes handwritten framework replacements and stale artifacts so the project can be rebuilt around official FastAPI, LangGraph, LangChain, and MCP primitives.

## Removed Runtime Layers

### Handwritten governance state machine

Removed files:

- `utils/state_machine.py`
- `utils/permission_matrix.py`

Why removed:

- They maintained a second workflow truth beside LangGraph `StateGraph` nodes and edges.
- They forced `ShangshuOrchestrator` to track `_machines`, transition history, and permission checks that are not part of the official LangGraph runtime.

Official rewrite path:

- Use LangGraph graph topology as the workflow authority.
- Add a LangGraph checkpointer and invoke graph runs with `config={"configurable": {"thread_id": request_id}}`.
- Use LangSmith tracing or LangGraph event streaming for state transition evidence instead of a custom transition log.

### Handwritten resume boundary

Removed behavior:

- Custom serialized `resume_state` snapshots.
- Rebuilding a graph from a handwritten `next_node` value.
- Replay fallback as a product-level resume mode.

Implemented official path:

- Compile the graph with a LangGraph checkpointer.
- Persist thread state through LangGraph persistence rather than `resume_state` JSON snapshots.
- Resume with `Command(resume=...)` and the same `thread_id`.
- Store only boundary metadata in JSON sessions: `mode`, `thread_id`,
  `interrupt_id`, `question`, `next`, and `created_at`.
- Return HTTP 409 for legacy sessions that do not have boundary checkpoint
  metadata.

### Handwritten streaming queue

Removed behavior:

- Custom `asyncio.Queue` based SSE wrappers for `/plan/stream` and `/resume/{request_id}/stream`.
- Manual `progress/result/error/done` event construction outside LangGraph.

Implemented official path:

- `/plan/stream` and `/resume/{request_id}/stream` return
  `text/event-stream`.
- The public contract emits `progress`, `result`, `error`, and `done` events
  using the same response serializer as non-streaming routes.
- Streaming resume uses the same boundary checkpoint and `Command(resume=...)`
  path as non-streaming resume.

### Direct MCP tool loop

Removed behavior:

- `utils.agent_runtime.run_direct_mcp_tool_calls` manually loaded MCP tools, selected them by name, called `ainvoke`, and compacted results.
- `utils/mcp_tools.py` exposed an unused handwritten tool map and direct call helper.

Implemented official path:

- Use `MultiServerMCPClient` to load MCP tools.
- Bind tools through LangChain agents or LangGraph `ToolNode`.
- Use LangChain/MCP adapter error handling instead of local ad hoc status envelopes where possible.

### Liubu migration leftovers

Removed behavior:

- Flight and Accommodation legacy `ingest` and `research_*` graph methods.
- `legacy_*_tool_call` conversion from dict requests to tool calls.

Official rewrite path:

- Remove compatibility adapters for dict-based tool requests.
- Rebuild live tool execution with LangGraph `ToolNode` only after constraints are expressed through official schemas, middleware, `ToolRuntime` context, or graph validation nodes.
- Do not reintroduce local executor envelopes around MCP tools.


### Constrained Liubu tool executor

Removed files:

- `provinces/liubu/constrained/tools.py`
- `provinces/liubu/constrained/tool_node.py`

Removed behavior:

- Custom `execute_constrained_tool_call` execution envelopes around MCP tools.
- Custom `ConstrainedMCPTool` wrappers that routed official `ToolNode` calls back into local executor code.
- Local timeout, argument redaction, argument policy, and evidence JSON serialization around each tool call.
- Offline fallback construction of synthetic `AIMessage.tool_calls` in Flight and Accommodation bureaus.

Implemented official path:

- Shangshu normalizes each Liubu dispatch payload into a bureau-specific
  `worker_input` subtask before invoking a subagent.
- Weather, Budget, Calendar, Flight, and Accommodation now compile through the
  same official ToolNode subgraph shape:
  `agent -> tools -> agent -> quality_gate`.
- The shared Liubu tooling helper loads MCP tools through the existing
  `MultiServerMCPClient` adapter, filters allowed names, supports
  `model.bind_tools(...)`, and converts official `ToolMessage` outputs into
  `LiubuToolEvidence`.
- Flight and Accommodation quality gates keep hard trip constraints in explicit
  validation nodes after ToolNode execution.
- LangGraph `ToolNode` executes tool calls directly; the removed constrained
  executor wrappers remain deleted.
## Removed Stale Files And Artifacts

- `interactive_demo.py`: obsolete interactive script with stale resume payload shape.
- `report.md`: ignored, encoding-damaged historical audit report.
- `docs/example-output.md`: stale placeholder-output example that conflicted with current quality gates.
- Historical `docs/superpowers` plans/specs for already completed cleanup and Liubu migration work.
- Generated runtime artifacts and Python caches where safely identifiable.

## Kept Official-Compatible Surfaces

- `mcp_servers/server.py`: uses `MultiServerMCPClient`.
- `mcp_servers/serpapi_server.py`: uses `FastMCP` tool declarations.
- Pydantic request/result models in `utils/schemas.py`.

## Follow-Up Hardening Order

1. Add richer Pydantic tool schemas, LangChain middleware, and ToolRuntime context where practical.
2. Restore user-facing examples only after they are generated from the current API path and do not contain generic draft travel content.
3. Expand live-provider verification evidence when credentials are available.

## Official References

These references are the rewrite baseline for the deleted handwritten layers. Future work should update this list when the project upgrades LangGraph, LangChain, or MCP adapter major versions.

- LangGraph overview: https://docs.langchain.com/oss/python/langgraph/overview
  - Use as the baseline for treating LangGraph as the orchestration runtime for durable execution, streaming, human-in-the-loop behavior, and persistence.
  - Relevant cleanup areas: removed `utils/state_machine.py`, removed `utils/permission_matrix.py`, and removed custom transition history as the workflow authority.

- LangGraph persistence: https://docs.langchain.com/oss/python/langgraph/persistence
  - Use checkpointers for short-term, thread-scoped graph state and invoke/stream with `config={"configurable": {"thread_id": request_id}}`.
  - Relevant cleanup areas: removed custom `resume_state` boundary snapshots and replay fallback; rebuild resume around checkpointers, interrupts, and `Command(resume=...)`.

- LangGraph streaming: https://docs.langchain.com/oss/python/langgraph/streaming
  - Prefer LangGraph event streaming for new applications; use official stream modes/projections instead of manually assembled queue events.
  - Relevant cleanup areas: removed custom `/plan/stream` and `/resume/{request_id}/stream` SSE queue wrappers.

- LangChain tools: https://docs.langchain.com/oss/python/langchain/tools
  - Use structured tool schemas, `ToolRuntime`, middleware/error handling, and LangGraph `ToolNode` for graph tool execution.
  - Relevant cleanup areas: removed direct MCP tool invocation loops and old dict-to-tool-call compatibility adapters.

- LangChain agents: https://docs.langchain.com/oss/python/langchain/agents
  - Use the current LangChain agent abstraction for any future higher-level LLM + tool loop instead of local ReAct glue.
  - Relevant cleanup areas: removed `run_react_mcp_task`; Weather, Budget,
    Calendar, Flight, and Accommodation now route live research through the
    shared ToolNode helper.

- LangChain MCP adapters: https://github.com/langchain-ai/langchain-mcp-adapters
  - Use `MultiServerMCPClient` and official MCP adapter patterns for loading MCP tools from streamable HTTP/SSE/stdio servers.
  - Relevant cleanup areas: removed `utils/mcp_tools.py`, direct MCP tool loops, and the constrained Liubu executor wrappers; rebuild future Liubu ToolNode paths against this adapter.
