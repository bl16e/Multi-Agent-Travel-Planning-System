from __future__ import annotations

import json
from typing import Any, Sequence

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode
from pydantic import PrivateAttr

from provinces.liubu.constrained.state import LiubuToolEvidence, LiubuWorkerInput
from provinces.liubu.constrained.tools import execute_constrained_tool_call
from utils.mcp_client import load_mcp_tools


class ConstrainedMCPTool(BaseTool):
    name: str
    description: str = ""
    args_schema: Any = None
    _original_tool: Any = PrivateAttr()
    _worker_input: LiubuWorkerInput = PrivateAttr()

    def __init__(self, *, original_tool: Any, worker_input: LiubuWorkerInput) -> None:
        super().__init__(
            name=str(getattr(original_tool, "name", "")),
            description=str(getattr(original_tool, "description", "") or ""),
            args_schema=getattr(original_tool, "args_schema", None),
        )
        self._original_tool = original_tool
        self._worker_input = worker_input

    def _run(self, **kwargs: Any) -> str:
        raise RuntimeError("ConstrainedMCPTool only supports async execution.")

    async def _arun(self, **kwargs: Any) -> str:
        evidence = await execute_constrained_tool_call(
            worker_input=self._worker_input,
            tool_map={self.name: self._original_tool},
            allowed_tool_names={self.name},
            tool_name=self.name,
            args=kwargs,
        )
        return json.dumps(evidence.model_dump(mode="json"), ensure_ascii=False, default=str)


def wrap_constrained_tools(worker_input: LiubuWorkerInput, tools: Sequence[Any], allowed_tool_names: set[str]) -> list[BaseTool]:
    return [
        ConstrainedMCPTool(original_tool=tool, worker_input=worker_input)
        for tool in tools
        if getattr(tool, "name", "") in allowed_tool_names
    ]


async def load_constrained_mcp_tools(worker_input: LiubuWorkerInput, server_names: list[str], allowed_tool_names: set[str]) -> list[BaseTool]:
    tools = await load_mcp_tools(server_names)
    return wrap_constrained_tools(worker_input, tools, allowed_tool_names)


async def run_constrained_tool_node(*, messages: list[BaseMessage], tools: Sequence[BaseTool]) -> tuple[list[ToolMessage], list[LiubuToolEvidence]]:
    output = await ToolNode(list(tools)).ainvoke({"messages": messages})
    tool_messages = list(output.get("messages") or [])
    evidence: list[LiubuToolEvidence] = []
    for message in tool_messages:
        if isinstance(message, ToolMessage):
            parsed = _parse_evidence_message(message)
            if parsed is not None:
                evidence.append(parsed)
    return tool_messages, evidence


def _parse_evidence_message(message: ToolMessage) -> LiubuToolEvidence | None:
    try:
        payload = json.loads(str(message.content))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return LiubuToolEvidence.model_validate(payload)
