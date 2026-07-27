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


@pytest.mark.asyncio
async def test_search_google_web_uses_bocha_web_search(monkeypatch):
    import mcp_servers.serpapi_server as serpapi_server

    captured = {}

    class FakeResponse:
        status_code = 200
        text = "{}"

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": {
                    "webPages": {
                        "value": [
                            {
                                "name": "Shanghai Museum Guide",
                                "url": "https://example.test/shanghai-museum",
                                "summary": "A concise guide to Shanghai Museum.",
                                "datePublished": "2026-01-01",
                                "siteName": "Example Travel",
                            }
                        ]
                    }
                }
            }

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers, json):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            captured["timeout"] = self.timeout
            return FakeResponse()

    monkeypatch.setattr(
        serpapi_server,
        "settings",
        SimpleNamespace(serpapi_api_key="serpapi-key", bocha_api_key="bocha-key"),
    )
    monkeypatch.setattr(serpapi_server.httpx, "AsyncClient", FakeAsyncClient)

    result = await serpapi_server.search_google_web.fn("上海必去景点", num=5)

    assert captured["url"].startswith("https://api.bochaai.com/v1/web-search")
    assert captured["headers"]["Authorization"] == "Bearer bocha-key"
    assert captured["json"] == {
        "query": "上海必去景点",
        "summary": True,
        "freshness": "noLimit",
        "count": 5,
    }
    assert result["search_metadata"]["provider"] == "bocha"
    assert result["organic_results"][0]["title"] == "Shanghai Museum Guide"
    assert result["organic_results"][0]["link"] == "https://example.test/shanghai-museum"


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
