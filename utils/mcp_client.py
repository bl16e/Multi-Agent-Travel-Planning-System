from __future__ import annotations

import asyncio
from typing import Any

_CLIENT_CACHE: dict[tuple[str, ...], Any] = {}
_TOOLS_CACHE: dict[tuple[str, ...], list[Any]] = {}
_CACHE_LOCK = asyncio.Lock()


async def load_mcp_tools(server_names: list[str]) -> list[Any]:
    try:
        from mcp_servers.server import build_mcp_client
    except ImportError:
        return []

    cache_key = tuple(sorted(server_names))
    cached_tools = _TOOLS_CACHE.get(cache_key)
    if cached_tools is not None:
        return cached_tools

    async with _CACHE_LOCK:
        cached_tools = _TOOLS_CACHE.get(cache_key)
        if cached_tools is not None:
            return cached_tools

        client = _CLIENT_CACHE.get(cache_key)
        if client is None:
            client = build_mcp_client(server_names)
            if client is None:
                _TOOLS_CACHE[cache_key] = []
                return []
            _CLIENT_CACHE[cache_key] = client

        tools = await client.get_tools()
        _TOOLS_CACHE[cache_key] = tools
        return tools
