import sys
import os
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

if os.getenv("RUN_LIVE_TESTS") != "1":
    os.environ.pop("QWEN_API_KEY", None)
    os.environ.pop("AMAP_API_KEY", None)
    os.environ.pop("SERPAPI_API_KEY", None)


def pytest_configure(config):
    config.addinivalue_line("markers", "live: tests that call real LLM or MCP services")


def pytest_collection_modifyitems(config, items):
    if os.getenv("RUN_LIVE_TESTS") == "1":
        return
    skip_live = pytest.mark.skip(reason="set RUN_LIVE_TESTS=1 to run live LLM/MCP tests")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
