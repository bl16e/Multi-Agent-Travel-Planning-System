import pytest
import json
import utils.agent_runtime as agent_runtime
import provinces.zhongshu_itinerary.graph as zhongshu_graph
from provinces.zhongshu_itinerary.graph import ZhongshuItineraryAgent
from utils.schemas import ItineraryDraftModel, SerpApiCandidateSelectionModel, SerpApiQueryPlanModel


@pytest.fixture(autouse=True)
def disable_live_amap_tools(monkeypatch):
    async def no_tools(server_names):
        return []

    monkeypatch.setattr(zhongshu_graph, "load_mcp_tools", no_tools)
    monkeypatch.setattr(zhongshu_graph, "load_agent_tools", lambda agent, categories: no_tools([]))


@pytest.mark.asyncio
async def test_ingest_request():
    agent = ZhongshuItineraryAgent()
    state = {
        "user_request": {
            "profile": {
                "destination_preferences": ["Tokyo"],
                "origin_city": "Beijing",
                "start_date": "2026-05-01",
                "end_date": "2026-05-05",
                "total_budget": 5000,
                "currency": "USD",
                "interests": ["culture", "food"]
            }
        },
        "governance": {"rejection_count": 0}
    }
    result = await agent.ingest_request(state)
    assert "normalized_request" in result
    assert result["normalized_request"]["destination"] == "Tokyo"


@pytest.mark.asyncio
async def test_ingest_request_rejects_missing_destination_preferences():
    agent = ZhongshuItineraryAgent()
    state = {
        "user_request": {
            "profile": {
                "destination_preferences": [],
                "origin_city": "Beijing",
                "start_date": "2026-05-01",
                "end_date": "2026-05-05",
                "total_budget": 5000,
            }
        },
        "governance": {"rejection_count": 0},
    }

    with pytest.raises(ValueError, match="destination preference"):
        await agent.ingest_request(state)


@pytest.mark.asyncio
async def test_draft_itinerary_offline_output_uses_request_destination(monkeypatch):
    monkeypatch.setattr(agent_runtime, "build_qwen_chat", lambda: None)

    agent = ZhongshuItineraryAgent()
    result = await agent.draft_itinerary(
        {
            "request_id": "offline_kyoto",
            "normalized_request": {
                "destination": "Kyoto",
                "origin_city": "Beijing",
                "start_date": "2026-05-01",
                "end_date": "2026-05-03",
                "total_budget": 3000,
                "currency": "USD",
                "adults": 2,
                "interests": ["culture"],
                "constraints": [],
                "revision_round": 0,
                "rejection_reasons": [],
                "revision_requests": [],
            },
        }
    )

    draft = result["draft"]
    assert draft["destination"] == "Kyoto"
    assert draft["daily_plan"][0]["date"] == "2026-05-01"
    assert draft["daily_plan"][0]["city"] == "Kyoto"
    assert "Kyoto" in draft["daily_plan"][0]["activities"][0]["title"]
    assert "Tokyo" not in str(draft)


@pytest.mark.asyncio
async def test_draft_itinerary_uses_serpapi_discovery_then_amap_confirmation(monkeypatch):
    seen = {}

    async def fake_synthesis(**kwargs):
        output_model = kwargs.get("output_model")
        if output_model is SerpApiQueryPlanModel:
            seen["query_planning_variables"] = kwargs.get("variables")
            return SerpApiQueryPlanModel.model_validate(
                {
                    "queries": [
                        {
                            "tool": "search_google_maps",
                            "query": "Shanghai museums official cultural sites",
                            "location": "",
                            "reason": "Find named museum and culture POIs instead of keyword matching the broad interest.",
                        }
                    ],
                    "strategy_notes": ["Prefer named POIs with official/provider identifiers."],
                }
            )
        if output_model is SerpApiCandidateSelectionModel:
            seen["candidate_selection_variables"] = kwargs.get("variables")
            return SerpApiCandidateSelectionModel.model_validate(
                {
                    "selected_candidates": [
                        {
                            "candidate_id": "search_google_maps:0:0",
                            "title": "\u4e0a\u6d77\u535a\u7269\u9986",
                            "reason": "Directly matches the cultural museum intent.",
                            "confidence": 0.92,
                        }
                    ],
                    "selection_notes": ["Exclude broad or less relevant candidates before Amap confirmation."],
                }
            )
        seen["variables"] = kwargs.get("variables")
        return ItineraryDraftModel.model_validate(
            {
                "destination": "\u4e0a\u6d77",
                "overview": "Concrete Shanghai plan.",
                "trip_style": "structured",
                "daily_plan": [
                    {
                        "day_index": 1,
                        "date": "2026-10-24",
                        "city": "\u4e0a\u6d77",
                        "theme": "Culture",
                        "summary": "Named Shanghai places.",
                        "activities": [
                            {
                                "start_time": "09:00",
                                "end_time": "10:30",
                                "title": "\u4e0a\u6d77\u535a\u7269\u9986",
                                "location_name": "\u4e0a\u6d77\u535a\u7269\u9986",
                                "description": "Visit the museum.",
                            }
                        ],
                    }
                ],
            }
        )

    monkeypatch.setattr("provinces.zhongshu_itinerary.graph.run_structured_synthesis", fake_synthesis)

    class FakeSerpApiTool:
        name = "search_google_maps"

        async def ainvoke(self, args):
            seen.setdefault("serpapi_args", []).append(args)
            return {
                "search_metadata": {
                    "raw_html_file": "https://serpapi.com/raw.html",
                    "prettify_html_file": "https://serpapi.com/prettify.html",
                },
                "local_results": [
                    {
                        "title": "\u4e0a\u6d77\u535a\u7269\u9986",
                        "type": "Museum",
                        "address": "201 Renmin Avenue, Shanghai",
                    },
                    {
                        "title": "\u4e2d\u534e\u827a\u672f\u5bab",
                        "type": "Art museum",
                        "address": "Pudong, Shanghai",
                    },
                ]
            }

    class FakeAmapTool:
        name = "maps_text_search"

        async def ainvoke(self, args):
            seen.setdefault("amap_args", []).append(args)
            return {
                "pois": [
                    {
                        "name": args["keywords"],
                        "type": "\u79d1\u6559\u6587\u5316\u670d\u52a1;\u535a\u7269\u9986",
                        "address": "\u4e0a\u6d77\u5e02\u9ec4\u6d66\u533a\u4eba\u6c11\u5927\u9053201\u53f7",
                        "location": "121.475379,31.228017",
                    }
                ]
            }

    async def fake_load_agent_tools(agent, categories):
        seen["agent"] = agent
        seen["categories"] = categories
        return [FakeSerpApiTool(), FakeAmapTool()]

    monkeypatch.setattr(zhongshu_graph, "load_agent_tools", fake_load_agent_tools, raising=False)

    agent = ZhongshuItineraryAgent()
    await agent.draft_itinerary(
        {
            "request_id": "domestic_amap_research",
            "normalized_request": {
                "destination": "\u4e0a\u6d77",
                "origin_city": "\u5317\u4eac",
                "start_date": "2026-10-24",
                "end_date": "2026-10-24",
                "total_budget": 1800,
                "currency": "CNY",
                "adults": 1,
                "children": 0,
                "interests": ["culture", "food"],
                "constraints": ["include official links"],
                "user_message": "Use named places and transport durations.",
                "revision_round": 0,
                "rejection_reasons": [],
                "revision_requests": [],
            },
        }
    )

    assert seen["agent"] == "ZHONGSHU"
    assert seen["categories"] == {
        "web_search_discovery",
        "global_place_discovery",
        "semantic_local_discovery",
        "domestic_poi_confirmation",
    }
    assert seen["query_planning_variables"]["destination"] == "\u4e0a\u6d77"
    assert seen["serpapi_args"][0]["query"] == "Shanghai museums official cultural sites"
    assert [item["keywords"] for item in seen["amap_args"]] == ["\u4e0a\u6d77\u535a\u7269\u9986"]
    assert all(item["keywords"] != "\u4e0a\u6d77 culture" for item in seen["amap_args"])
    assert all(item["city"] == "\u4e0a\u6d77" for item in seen["amap_args"])
    research_payload = json.loads(seen["variables"]["research_context"])
    assert research_payload[0]["phase"] == "query_planning"
    assert research_payload[0]["status"] == "ok"
    assert any(item["phase"] == "candidate_selection" and item["status"] == "ok" for item in research_payload)
    discovery = next(item for item in research_payload if item["phase"] == "discovery")
    assert discovery["tool"] == "search_google_maps"
    assert list(discovery["result"].keys()) == ["candidates"]
    assert discovery["result"]["candidates"][0]["candidate_id"] == "search_google_maps:0:0"
    assert "raw_html_file" not in seen["variables"]["research_context"]
    confirmation = next(item for item in research_payload if item["phase"] == "confirmation")
    assert confirmation["tool"] == "maps_text_search"
    assert confirmation["result"]["pois"][0]["name"] == "\u4e0a\u6d77\u535a\u7269\u9986"
    assert "Direct MCP tool calls removed" not in seen["variables"]["research_context"]


@pytest.mark.asyncio
async def test_candidate_selection_fallback_filters_low_relevance_places(monkeypatch):
    async def fake_synthesis(**kwargs):
        raise RuntimeError("candidate selection unavailable")

    monkeypatch.setattr("provinces.zhongshu_itinerary.graph.run_structured_synthesis", fake_synthesis)

    agent = ZhongshuItineraryAgent()
    evidence = []
    selected = await agent._select_serpapi_candidates(
        {"destination": "\u4e0a\u6d77", "interests": ["\u6587\u5316"]},
        {
            "search_google_maps:0:0": {
                "_candidate_id": "search_google_maps:0:0",
                "title": "Shanghai Culture Square Garage",
                "type": "Parking garage",
                "address": "Shanghai",
            },
            "search_google_maps:0:1": {
                "_candidate_id": "search_google_maps:0:1",
                "title": "Shanghai Museum",
                "type": "Museum",
                "address": "201 Renmin Ave",
                "place_id": "ChIJPWUSbWlwsjURbNvIw3tOTE0",
            },
        },
        evidence,
    )

    assert [item["title"] for item in selected] == ["Shanghai Museum"]
    assert evidence[-1]["phase"] == "candidate_selection_fallback"


@pytest.mark.asyncio
async def test_draft_itinerary_does_not_hardcode_structured_synthesis_timeout(monkeypatch):
    seen = {}


    async def fake_synthesis(**kwargs):
        seen["timeout_seconds"] = kwargs.get("timeout_seconds")
        return ItineraryDraftModel.model_validate(
            {
                "destination": "Tokyo",
                "overview": "Concrete Tokyo plan.",
                "trip_style": "structured",
                "daily_plan": [
                    {
                        "day_index": 1,
                        "date": "2026-05-01",
                        "city": "Tokyo",
                        "theme": "Culture",
                        "summary": "Named Tokyo places.",
                        "activities": [
                            {
                                "start_time": "09:00",
                                "end_time": "10:30",
                                "title": "Senso-ji Temple",
                                "location_name": "Senso-ji Temple, Asakusa",
                                "description": "Visit the temple.",
                            }
                        ],
                    }
                ],
            }
        )

    monkeypatch.setattr("provinces.zhongshu_itinerary.graph.run_structured_synthesis", fake_synthesis)

    agent = ZhongshuItineraryAgent()
    await agent.draft_itinerary(
        {
            "request_id": "bounded_timeout",
            "normalized_request": {
                "destination": "Tokyo",
                "origin_city": "Beijing",
                "start_date": "2026-05-01",
                "end_date": "2026-05-01",
                "total_budget": 1800,
                "currency": "USD",
                "adults": 1,
                "children": 0,
                "interests": ["culture"],
                "constraints": [],
                "user_message": "Use named places.",
                "revision_round": 0,
                "rejection_reasons": [],
                "revision_requests": [],
            },
        }
    )

    assert seen["timeout_seconds"] is None


@pytest.mark.asyncio
async def test_draft_itinerary_passes_readable_chinese_prompt_to_synthesis(monkeypatch):
    seen = {}


    async def fake_synthesis(**kwargs):
        seen["user_prompt"] = kwargs.get("user_prompt")
        return ItineraryDraftModel.model_validate(
            {
                "destination": "Tokyo",
                "overview": "Concrete Tokyo plan.",
                "trip_style": "structured",
                "daily_plan": [
                    {
                        "day_index": 1,
                        "date": "2026-05-01",
                        "city": "Tokyo",
                        "theme": "Culture",
                        "summary": "Named Tokyo places.",
                        "activities": [
                            {
                                "start_time": "09:00",
                                "end_time": "10:30",
                                "title": "Senso-ji Temple",
                                "location_name": "Senso-ji Temple, Asakusa",
                                "description": "Visit the temple.",
                            }
                        ],
                    }
                ],
            }
        )

    monkeypatch.setattr("provinces.zhongshu_itinerary.graph.run_structured_synthesis", fake_synthesis)

    agent = ZhongshuItineraryAgent()
    await agent.draft_itinerary(
        {
            "request_id": "readable_prompt",
            "normalized_request": {
                "destination": "Tokyo",
                "origin_city": "Beijing",
                "start_date": "2026-05-01",
                "end_date": "2026-05-01",
                "total_budget": 1800,
                "currency": "USD",
                "adults": 1,
                "children": 0,
                "interests": ["culture"],
                "constraints": [],
                "user_message": "Use named places.",
                "revision_round": 0,
                "rejection_reasons": [],
                "revision_requests": [],
            },
        }
    )

    assert "请根据用户需求生成详细的旅行行程草案" in seen["user_prompt"]
    assert "鐠囬攱" not in seen["user_prompt"]


@pytest.mark.asyncio
async def test_draft_itinerary_with_synthesis(monkeypatch):
    async def fake_synthesis(**kwargs):
        variables = kwargs["variables"]
        return ItineraryDraftModel.model_validate(
            {
                "destination": variables["destination"],
                "overview": "Concrete Shanghai plan.",
                "trip_style": "structured",
                "daily_plan": [
                    {
                        "day_index": 1,
                        "date": variables["start_date"],
                        "city": variables["destination"],
                        "theme": "Culture",
                        "summary": "Named Shanghai places.",
                        "activities": [
                            {
                                "start_time": "09:00",
                                "end_time": "10:30",
                                "title": "\u4e0a\u6d77\u535a\u7269\u9986",
                                "location_name": "\u4e0a\u6d77\u535a\u7269\u9986",
                                "description": "Visit the museum.",
                            }
                        ],
                    }
                ],
            }
        )

    monkeypatch.setattr("provinces.zhongshu_itinerary.graph.run_structured_synthesis", fake_synthesis)

    agent = ZhongshuItineraryAgent()
    state = {
        "request_id": "test_llm_001",
        "normalized_request": {
            "destination": "\u4e0a\u6d77",
            "origin_city": "\u5317\u4eac",
            "start_date": "2026-10-24",
            "end_date": "2026-10-26",
            "total_budget": 3000,
            "currency": "CNY",
            "adults": 2,
            "interests": ["culture"],
            "constraints": [],
            "revision_round": 0,
            "rejection_reasons": [],
            "revision_requests": []
        }
    }
    result = await agent.draft_itinerary(state)
    assert "draft" in result
    draft = result["draft"]
    assert draft["destination"] == "\u4e0a\u6d77"
    assert "daily_plan" in draft
    assert len(draft["daily_plan"]) > 0






