import sys
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

pytest_plugins = ["pytester"]

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))
load_dotenv(project_root / ".env")

if os.getenv("RUN_LIVE_TESTS") != "1":
    os.environ["QWEN_API_KEY"] = ""
    os.environ["AMAP_API_KEY"] = ""
    os.environ["SERPAPI_API_KEY"] = ""


@pytest.fixture(autouse=True)
def clear_cached_app_configuration():
    try:
        from utils.llm_factory import build_qwen_chat
        from utils.settings import get_settings
    except ModuleNotFoundError:
        yield
        return

    get_settings.cache_clear()
    build_qwen_chat.cache_clear()
    yield
    get_settings.cache_clear()
    build_qwen_chat.cache_clear()


def pytest_configure(config):
    config.addinivalue_line("markers", "live: tests that call real LLM or MCP services")


def pytest_collection_modifyitems(config, items):
    if os.getenv("RUN_LIVE_TESTS") == "1":
        return
    skip_live = pytest.mark.skip(reason="set RUN_LIVE_TESTS=1 to run live LLM/MCP tests")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
