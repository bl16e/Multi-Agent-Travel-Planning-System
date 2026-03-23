from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent

from utils.llm_factory import build_qwen_chat
from utils.mcp_client import load_mcp_tools


FALLBACK_MESSAGE = "MCP or LLM unavailable; falling back to heuristic synthesis."
DEFAULT_TOOL_LOAD_TIMEOUT_SECONDS = 6.0
DEFAULT_AGENT_INVOKE_TIMEOUT_SECONDS = 18.0
DEFAULT_STRUCTURED_SYNTHESIS_TIMEOUT_SECONDS = 20.0


def load_soul_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


async def run_react_mcp_task(
    *,
    soul_path: str | Path,
    server_names: list[str],
    user_task: str,
    tool_load_timeout_seconds: float = DEFAULT_TOOL_LOAD_TIMEOUT_SECONDS,
    invoke_timeout_seconds: float = DEFAULT_AGENT_INVOKE_TIMEOUT_SECONDS,
) -> str:
    llm = build_qwen_chat()
    try:
        tools = await asyncio.wait_for(load_mcp_tools(server_names), timeout=tool_load_timeout_seconds)
    except asyncio.TimeoutError:
        return f"{FALLBACK_MESSAGE} MCP tool loading timed out after {tool_load_timeout_seconds:.0f}s."
    except Exception as exc:
        return f"{FALLBACK_MESSAGE} MCP tool loading failed: {exc}"
    prompt = load_soul_prompt(soul_path)

    if llm is None or not tools:
        return FALLBACK_MESSAGE

    agent = create_agent(model=llm, tools=tools, system_prompt=prompt)
    try:
        result = await asyncio.wait_for(agent.ainvoke({"messages": [("user", user_task)]}), timeout=invoke_timeout_seconds)
    except asyncio.TimeoutError:
        return f"{FALLBACK_MESSAGE} Agent execution timed out after {invoke_timeout_seconds:.0f}s."
    except Exception as exc:
        return f"{FALLBACK_MESSAGE} Agent tool execution failed: {exc}"
    messages = result.get("messages", [])
    if not messages:
        return "No MCP research output returned."
    return str(messages[-1].content)


async def run_structured_synthesis(
    *,
    soul_path: str | Path,
    output_model: Any,
    user_prompt: str,
    variables: dict[str, Any],
    timeout_seconds: float = DEFAULT_STRUCTURED_SYNTHESIS_TIMEOUT_SECONDS,
) -> Any:
    llm = build_qwen_chat()
    if llm is None:
        raise RuntimeError("LLM unavailable for structured synthesis.")
    structured = llm.with_structured_output(output_model)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", load_soul_prompt(soul_path)),
            ("user", user_prompt),
        ]
    )
    try:
        return await asyncio.wait_for((prompt | structured).ainvoke(variables), timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"Structured synthesis timed out after {timeout_seconds:.0f}s.") from exc


def soul_path_for(file_path: str | Path) -> Path:
    return Path(file_path).with_name("SOUL.md")