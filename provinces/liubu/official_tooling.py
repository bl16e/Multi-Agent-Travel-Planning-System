from __future__ import annotations

import json
import asyncio
import inspect
import logging
from typing import Any, Callable, Sequence

from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode

from provinces.liubu.constrained.state import LiubuToolEvidence
from utils.settings import get_settings
from utils.mcp_client import load_mcp_tools
from utils.mcp_tool_registry import load_agent_tools_for_names

logger = logging.getLogger(__name__)


async def load_allowed_liubu_tools(server_names: list[str], allowed_names: set[str], *, agent: str | None = None) -> list[Any]:
    if agent:
        return await load_agent_tools_for_names(agent, allowed_names)
    tools = await load_mcp_tools(server_names)
    if not allowed_names:
        return tools
    return [tool for tool in tools if getattr(tool, "name", None) in allowed_names]


def bind_tools_if_available(model: Any, tools: Sequence[Any]) -> Any:
    if model is not None and tools and hasattr(model, "bind_tools"):
        return model.bind_tools(list(tools))
    return None


async def invoke_bound_tool_model(
    model: Any,
    messages: Sequence[Any],
    *,
    bureau: str,
    request_id: str,
    destination: str,
) -> Any | None:
    timeout_seconds = get_settings().liubu_tool_timeout_seconds
    try:
        return await asyncio.wait_for(model.ainvoke(list(messages)), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(
            "Bound tool model call timed out request_id=%s bureau=%s destination=%s timeout_seconds=%s",
            request_id,
            bureau,
            destination,
            timeout_seconds,
        )
        return None
    except Exception as exc:
        logger.warning(
            "Bound tool model call failed request_id=%s bureau=%s destination=%s error_type=%s error=%s",
            request_id,
            bureau,
            destination,
            type(exc).__name__,
            exc,
        )
        return None


def tool_messages_to_evidence(messages: Sequence[Any]) -> list[LiubuToolEvidence]:
    evidence: list[LiubuToolEvidence] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        decoded = _decode_tool_content(message.content)
        if _tool_content_is_error(decoded):
            evidence.append(
                LiubuToolEvidence(
                    tool_name=str(message.name or "unknown_tool"),
                    args={},
                    status="error",
                    error=str(decoded),
                    data_source="live",
                )
            )
            continue
        evidence.append(
            LiubuToolEvidence(
                tool_name=str(message.name or "unknown_tool"),
                args={},
                status="ok",
                result=decoded,
                data_source="live",
            )
        )
    return evidence


async def run_tool_node_collect_evidence(
    tool_node: ToolNode,
    state: dict[str, Any],
    *,
    bureau: str = "",
    request_id: str = "",
) -> dict[str, Any]:
    timeout_seconds = get_settings().liubu_tool_timeout_seconds
    try:
        output = await asyncio.wait_for(ToolNode.ainvoke(tool_node, state), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(
            "ToolNode execution timed out request_id=%s bureau=%s timeout_seconds=%s",
            request_id,
            bureau,
            timeout_seconds,
        )
        evidence = [item.model_dump(mode="json") for item in _timeout_evidence_from_state(state)]
        return {
            "messages": list(state.get("messages", [])),
            "tool_evidence": evidence,
            "tool_step_count": int(state.get("tool_step_count") or 0) + 1,
        }
    messages = list(output.get("messages", [])) if isinstance(output, dict) else []
    evidence = [item.model_dump(mode="json") for item in tool_messages_to_evidence(messages)]
    return {
        "messages": messages,
        "tool_evidence": evidence,
        "tool_step_count": int(state.get("tool_step_count") or 0) + 1,
    }


class EvidenceToolNode(ToolNode):
    def __init__(
        self,
        tools: Sequence[Any],
        *,
        collector: Callable[[ToolNode, dict[str, Any]], Any] = run_tool_node_collect_evidence,
    ) -> None:
        super().__init__(list(tools))
        self._collector = collector

    async def ainvoke(self, input: Any, config: Any | None = None, **kwargs: Any) -> Any:
        state = dict(input or {})
        bureau, request_id = _worker_context(state)
        output = await _call_collector(self._collector, self, state, bureau=bureau, request_id=request_id)
        if not isinstance(output, dict):
            return output
        messages = list(output.get("messages", []))
        evidence = list(output.get("tool_evidence") or [])
        if messages and not evidence:
            evidence = [item.model_dump(mode="json") for item in tool_messages_to_evidence(messages)]
        output["tool_evidence"] = evidence
        output["tool_step_count"] = int(state.get("tool_step_count") or 0) + 1
        return output


def _decode_tool_content(content: Any) -> Any:
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    return content


def _tool_content_is_error(content: Any) -> bool:
    if isinstance(content, str):
        lowered = content.lower()
        return "toolexception" in lowered or lowered.startswith("error:")
    if isinstance(content, dict):
        status = str(content.get("status") or "").lower()
        if status in {"error", "failed", "failure"}:
            return True
        message = str(content.get("error") or content.get("message") or "").lower()
        return "toolexception" in message or message.startswith("error:")
    return False


async def _call_collector(
    collector: Callable[[ToolNode, dict[str, Any]], Any],
    tool_node: ToolNode,
    state: dict[str, Any],
    *,
    bureau: str,
    request_id: str,
) -> Any:
    signature = inspect.signature(collector)
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if accepts_kwargs or {"bureau", "request_id"} <= set(signature.parameters):
        return await collector(tool_node, state, bureau=bureau, request_id=request_id)
    return await collector(tool_node, state)


def _worker_context(state: dict[str, Any]) -> tuple[str, str]:
    worker_input = state.get("worker_input")
    bureau = getattr(worker_input, "bureau", None)
    request_id = getattr(worker_input, "request_id", None)
    if isinstance(worker_input, dict):
        bureau = bureau or worker_input.get("bureau")
        request_id = request_id or worker_input.get("request_id")
    return str(bureau or ""), str(request_id or "")


def _timeout_evidence_from_state(state: dict[str, Any]) -> list[LiubuToolEvidence]:
    messages = list(state.get("messages", []) or [])
    last_message = messages[-1] if messages else None
    tool_calls = list(getattr(last_message, "tool_calls", []) or [])
    if not tool_calls:
        return [
            LiubuToolEvidence(
                tool_name="unknown_tool",
                args={},
                status="timeout",
                error="ToolNode execution timed out before returning tool messages.",
            )
        ]
    evidence: list[LiubuToolEvidence] = []
    for call in tool_calls:
        evidence.append(
            LiubuToolEvidence(
                tool_name=str(call.get("name") or "unknown_tool"),
                args=dict(call.get("args") or {}),
                status="timeout",
                error="ToolNode execution timed out before returning tool messages.",
            )
        )
    return evidence
