from __future__ import annotations

import asyncio
import logging
from typing import Any

from utils.settings import get_settings

_CLIENT_CACHE: dict[tuple[str, ...], Any] = {}
_TOOLS_CACHE: dict[tuple[str, ...], list[Any]] = {}
_CACHE_LOCK = asyncio.Lock()
logger = logging.getLogger(__name__)


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

        timeout_seconds = get_settings().mcp_tooling_timeout_seconds
        try:
            tools = await asyncio.wait_for(client.get_tools(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logger.warning(
                "MCP tool loading timed out server_names=%s timeout_seconds=%s",
                ",".join(cache_key),
                timeout_seconds,
            )
            tools = []
        except Exception as exc:
            logger.warning(
                "MCP tool loading failed server_names=%s error_type=%s error=%s",
                ",".join(cache_key),
                type(exc).__name__,
                exc,
            )
            tools = []
        _TOOLS_CACHE[cache_key] = tools
        return tools
