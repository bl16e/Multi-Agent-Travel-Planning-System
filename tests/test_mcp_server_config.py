from utils.settings import get_settings


def test_amap_mcp_client_uses_streamable_http_transport(monkeypatch):
    import mcp_servers.server as mcp_server

    captured = {}

    class FakeMultiServerMCPClient:
        def __init__(self, config):
            captured["config"] = config

    monkeypatch.setenv("AMAP_API_KEY", "test-amap-key")
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    get_settings.cache_clear()
    monkeypatch.setattr(mcp_server, "MultiServerMCPClient", FakeMultiServerMCPClient)

    assert mcp_server.build_mcp_client(["amap"]) is not None

    amap_config = captured["config"]["amap"]
    assert amap_config["transport"] == "streamable_http"
    assert amap_config["url"].startswith("https://mcp.amap.com/mcp")
    assert "key=test-amap-key" in amap_config["url"]


def test_serpapi_mcp_client_can_use_prestarted_streamable_http_server(monkeypatch):
    import mcp_servers.server as mcp_server

    captured = {}

    class FakeMultiServerMCPClient:
        def __init__(self, config):
            captured["config"] = config

    monkeypatch.setenv("SERPAPI_API_KEY", "test-serpapi-key")
    monkeypatch.setenv("SERPAPI_MCP_URL", "http://127.0.0.1:8765/mcp")
    get_settings.cache_clear()
    monkeypatch.setattr(mcp_server, "MultiServerMCPClient", FakeMultiServerMCPClient)

    assert mcp_server.build_mcp_client(["serpapi"]) is not None

    serpapi_config = captured["config"]["serpapi"]
    assert serpapi_config == {
        "url": "http://127.0.0.1:8765/mcp",
        "transport": "streamable_http",
    }


def test_serpapi_mcp_client_defaults_to_stdio_autostart(monkeypatch):
    import mcp_servers.server as mcp_server

    captured = {}

    class FakeMultiServerMCPClient:
        def __init__(self, config):
            captured["config"] = config

    monkeypatch.setenv("SERPAPI_API_KEY", "test-serpapi-key")
    monkeypatch.delenv("SERPAPI_MCP_URL", raising=False)
    get_settings.cache_clear()
    monkeypatch.setattr(mcp_server, "MultiServerMCPClient", FakeMultiServerMCPClient)

    assert mcp_server.build_mcp_client(["serpapi"]) is not None

    serpapi_config = captured["config"]["serpapi"]
    assert serpapi_config["transport"] == "stdio"
    assert serpapi_config["args"] == [mcp_server.SERPAPI_SERVER_PATH]


def test_search_mcp_client_autostarts_with_bocha_key_only(monkeypatch):
    import mcp_servers.server as mcp_server

    captured = {}

    class FakeMultiServerMCPClient:
        def __init__(self, config):
            captured["config"] = config

    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    monkeypatch.delenv("SERPAPI_MCP_URL", raising=False)
    monkeypatch.setenv("BOCHA_API_KEY", "test-bocha-key")
    get_settings.cache_clear()
    monkeypatch.setattr(mcp_server, "MultiServerMCPClient", FakeMultiServerMCPClient)

    assert mcp_server.build_mcp_client(["serpapi"]) is not None

    serpapi_config = captured["config"]["serpapi"]
    assert serpapi_config["transport"] == "stdio"
    assert serpapi_config["args"] == [mcp_server.SERPAPI_SERVER_PATH]
    assert serpapi_config["env"]["BOCHA_API_KEY"] == "test-bocha-key"
    assert "SERPAPI_API_KEY" not in serpapi_config["env"]
