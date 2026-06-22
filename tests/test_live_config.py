import os

import pytest

from utils.settings import get_settings
from utils.llm_factory import build_qwen_chat


@pytest.mark.live
def test_live_qwen_configuration_is_available_when_enabled():
    if not os.getenv("QWEN_API_KEY"):
        pytest.skip("QWEN_API_KEY is required for live Qwen tests")
    build_qwen_chat.cache_clear()
    assert build_qwen_chat() is not None


def test_runtime_settings_defaults(monkeypatch):
    for name in (
        "SESSION_CACHE_MAX_ENTRIES",
        "SESSION_CACHE_TTL_SECONDS",
        "ENABLE_LANGGRAPH_INTERRUPTS",
        "QWEN_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.session_cache_max_entries == 500
    assert settings.session_cache_ttl_seconds == 86400
    assert settings.enable_langgraph_interrupts is False
    assert settings.qwen_timeout_seconds == 60


def test_runtime_settings_env_overrides(monkeypatch):
    monkeypatch.setenv("SESSION_CACHE_MAX_ENTRIES", "7")
    monkeypatch.setenv("SESSION_CACHE_TTL_SECONDS", "33")
    monkeypatch.setenv("ENABLE_LANGGRAPH_INTERRUPTS", "1")
    monkeypatch.setenv("QWEN_TIMEOUT_SECONDS", "12")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.session_cache_max_entries == 7
    assert settings.session_cache_ttl_seconds == 33
    assert settings.enable_langgraph_interrupts is True
    assert settings.qwen_timeout_seconds == 12
