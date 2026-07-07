from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_serpapi_search_returns_sanitized_http_error(monkeypatch):
    import mcp_servers.serpapi_server as serpapi_server

    class FakeResponse:
        status_code = 400
        text = '{"error": "`check_in_date` cannot be in the past.", "api_key": "secret-key"}'

        def json(self):
            return {"error": "`check_in_date` cannot be in the past.", "api_key": "secret-key"}

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, params):
            return FakeResponse()

    monkeypatch.setattr(serpapi_server, "settings", SimpleNamespace(serpapi_api_key="secret-key"))
    monkeypatch.setattr(serpapi_server.httpx, "AsyncClient", FakeAsyncClient)

    result = await serpapi_server._search({"engine": "google_hotels", "api_key": "secret-key"})

    assert result["status"] == "error"
    assert result["http_status"] == 400
    assert "check_in_date" in result["error"]
    assert "secret-key" not in str(result)


def test_serpapi_server_main_defaults_to_stdio(monkeypatch):
    import mcp_servers.serpapi_server as serpapi_server

    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(serpapi_server.mcp, "run", fake_run)

    serpapi_server.main([])

    assert captured == {"transport": "stdio"}


def test_serpapi_server_main_can_start_streamable_http(monkeypatch):
    import mcp_servers.serpapi_server as serpapi_server

    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(serpapi_server.mcp, "run", fake_run)

    serpapi_server.main([
        "--transport",
        "streamable-http",
        "--host",
        "127.0.0.1",
        "--port",
        "8765",
        "--path",
        "/mcp",
    ])

    assert captured == {
        "transport": "streamable-http",
        "host": "127.0.0.1",
        "port": 8765,
        "path": "/mcp",
    }
