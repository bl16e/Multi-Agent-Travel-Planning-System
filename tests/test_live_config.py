import os

import pytest

from utils.llm_factory import build_qwen_chat


@pytest.mark.live
def test_live_qwen_configuration_is_available_when_enabled():
    if not os.getenv("QWEN_API_KEY"):
        pytest.skip("QWEN_API_KEY is required for live Qwen tests")
    build_qwen_chat.cache_clear()
    assert build_qwen_chat() is not None
