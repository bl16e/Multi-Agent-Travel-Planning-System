from __future__ import annotations

from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

from utils.settings import get_settings

SERPAPI_SERVER_PATH = str(Path(__file__).resolve().with_name("serpapi_server.py"))
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])


def build_mcp_client(server_names: list[str] | None = None) -> MultiServerMCPClient | None:
    settings = get_settings()
    requested = set(server_names or [])
    config: dict[str, dict[str, object]] = {}

    if (not requested or "serpapi" in requested) and settings.serpapi_api_key:
        config["serpapi"] = {
            "command": "python",
            "args": [SERPAPI_SERVER_PATH],
            "transport": "stdio",
            "cwd": PROJECT_ROOT,
            "env": {"PYTHONUTF8": "1", "SERPAPI_API_KEY": settings.serpapi_api_key},
        }

    if (not requested or "amap" in requested) and settings.amap_mcp_url:
        config["amap"] = {
            "url": settings.amap_mcp_url,
            "transport": "http",
        }

    if not config:
        return None
    return MultiServerMCPClient(config)
