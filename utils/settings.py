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


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()