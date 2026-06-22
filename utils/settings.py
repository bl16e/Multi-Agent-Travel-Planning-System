from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _append_query_param(url: str, key: str, value: str) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.setdefault(key, value)
    return urlunsplit(parts._replace(query=urlencode(query)))


def _default_amap_mcp_url() -> str | None:
    explicit_url = os.getenv("AMAP_MCP_URL")
    amap_api_key = os.getenv("AMAP_API_KEY")
    if explicit_url:
        if amap_api_key:
            return _append_query_param(explicit_url, "key", amap_api_key)
        return explicit_url
    if amap_api_key:
        return _append_query_param("https://mcp.amap.com/mcp", "key", amap_api_key)
    return None


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


class AppSettings(BaseModel):
    qwen_api_key: str | None = Field(default_factory=lambda: os.getenv("QWEN_API_KEY"))
    qwen_model: str = Field(default_factory=lambda: os.getenv("QWEN_MODEL", "qwen-plus"))
    qwen_base_url: str = Field(
        default_factory=lambda: os.getenv(
            "QWEN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    )
    output_dir: str = Field(default_factory=lambda: os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "artifacts")))
    default_currency: str = Field(default_factory=lambda: os.getenv("DEFAULT_CURRENCY", "USD"))
    amap_api_key: str | None = Field(default_factory=lambda: os.getenv("AMAP_API_KEY"))
    amap_mcp_url: str | None = Field(default_factory=_default_amap_mcp_url)
    serpapi_api_key: str | None = Field(default_factory=lambda: os.getenv("SERPAPI_API_KEY"))
    run_live_tests: bool = Field(default_factory=lambda: os.getenv("RUN_LIVE_TESTS") == "1")
    session_store_dir: str = Field(default_factory=lambda: os.getenv("SESSION_STORE_DIR", str(PROJECT_ROOT / "artifacts" / "sessions")))
    session_cache_max_entries: int = Field(default_factory=lambda: _env_int("SESSION_CACHE_MAX_ENTRIES", 500))
    session_cache_ttl_seconds: int = Field(default_factory=lambda: _env_int("SESSION_CACHE_TTL_SECONDS", 86400))
    enable_langgraph_interrupts: bool = Field(default_factory=lambda: _env_bool("ENABLE_LANGGRAPH_INTERRUPTS"))
    qwen_timeout_seconds: int = Field(default_factory=lambda: _env_int("QWEN_TIMEOUT_SECONDS", 60))


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
