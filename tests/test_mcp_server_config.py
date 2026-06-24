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
