from __future__ import annotations

from typing import Any

from utils.mcp_client import load_mcp_tools


async def load_named_tools(server_names: list[str]) -> dict[str, Any]:
    tools = await load_mcp_tools(server_names)
    return {getattr(tool, "name", ""): tool for tool in tools}


async def call_tool(tool_map: dict[str, Any], tool_name: str, payload: dict[str, Any]) -> Any:
    tool = tool_map.get(tool_name)
    if tool is None:
        raise KeyError(f"MCP tool not available: {tool_name}")
    return await tool.ainvoke(payload)
