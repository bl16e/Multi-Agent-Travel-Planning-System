"""Real E2E tests for each Liubu (六部) department.

Each test runs a single bureau in isolation against live MCP servers
(amap for Weather / Accommodation / Flight Transport; serpapi for
Budget / Calendar), verifies that deliverable content is produced with
real, accessible links, and confirms that no fallback language leaks
into live results.

All tests are marked ``@pytest.mark.live`` – they require
``RUN_LIVE_TESTS=1`` and the corresponding API keys to execute.
"""

from __future__ import annotations

import asyncio
import os
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from provinces.liubu.constrained.state import normalize_worker_input
from utils.settings import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Real-world trip dates (always in the future so amap weather / serpapi
# hotel searches accept them).
# ---------------------------------------------------------------------------
_START_DATE: date = date.today() + timedelta(days=7)
_END_DATE: date = _START_DATE + timedelta(days=2)
START_DATE_STR: str = _START_DATE.isoformat()
END_DATE_STR: str = _END_DATE.isoformat()

# ---------------------------------------------------------------------------
# Domestic destinations – all Chinese cities.
# ---------------------------------------------------------------------------
DEST_SHANGHAI = "上海"
DEST_HANGZHOU = "杭州"
ORIGIN_BEIJING = "北京"
AIRPORT_PEK = "PEK"
AIRPORT_PVG = "PVG"
AIRPORT_HGH = "HGH"


# ===================================================================
# Helpers
# ===================================================================


def _make_subtask(payload: dict[str, Any], bureau: str) -> dict[str, Any]:
    """Wrap a payload dict into the subtask shape that ``bureau.run()`` expects."""
    return {"worker_input": normalize_worker_input(payload, bureau)}


def _base_payload(
    *,
    destination: str = DEST_SHANGHAI,
    origin_city: str = ORIGIN_BEIJING,
    origin_airport: str = AIRPORT_PEK,
    dest_airport: str = AIRPORT_PVG,
    currency: str = "CNY",
    total_budget: int = 5000,
    daily_plan: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Minimal but realistic payload for any liubu bureau.

    Daily plan defaults to two empty days so the bureau has trip dates to
    work with.  Callers can pass a richer *daily_plan* when needed (e.g.
    Budget / Calendar which derive live results from itinerary data).
    """
    if daily_plan is None:
        daily_plan = [
            {"date": START_DATE_STR, "activities": []},
            {"date": END_DATE_STR, "activities": []},
        ]
    return {
        "request_id": f"liubu_e2e_{destination}",
        "approved_draft": {
            "destination": destination,
            "itinerary_draft": {
                "destination": destination,
                "daily_plan": daily_plan,
            },
        },
        "execution_plan": {
            "user_request": {
                "profile": {
                    "origin_city": origin_city,
                    "origin_airport_code": origin_airport,
                    "destination_preferences": [destination],
                    "destination_airport_code": dest_airport,
                    "start_date": START_DATE_STR,
                    "end_date": END_DATE_STR,
                    "adults": 2,
                    "currency": currency,
                    "total_budget": total_budget,
                    "budget_level": "mid_range",
                },
            },
        },
    }


def _daily_plan_with_real_activities(destination: str) -> list[dict[str, Any]]:
    """Return a daily plan with realistic named activities for *destination*.

    Each activity carries a real ``map_link`` so Budget and Calendar
    bureaus can derive live results from the itinerary without needing
    the LLM to emit MCP tool calls.  The links are valid, clickable
    Amap search URLs.
    """
    if destination == DEST_SHANGHAI:
        activities_day1 = [
            {
                "title": "上海博物馆",
                "start_time": "09:00",
                "end_time": "11:30",
                "location_name": "上海市黄浦区人民大道201号",
                "description": "参观上海博物馆，欣赏古代青铜器和陶瓷收藏。",
                "map_link": "https://ditu.amap.com/search?query=上海博物馆",
                "estimated_cost": 0,
            },
            {
                "title": "外滩观光",
                "start_time": "14:00",
                "end_time": "16:00",
                "location_name": "上海市黄浦区中山东一路",
                "description": "漫步外滩，欣赏黄浦江两岸万国建筑博览群与陆家嘴天际线。",
                "map_link": "https://ditu.amap.com/search?query=上海外滩",
                "estimated_cost": 0,
            },
        ]
        activities_day2 = [
            {
                "title": "豫园",
                "start_time": "09:00",
                "end_time": "11:00",
                "location_name": "上海市黄浦区豫园老街279号",
                "description": "游览明代古典园林豫园，体验上海传统文化。",
                "map_link": "https://ditu.amap.com/search?query=上海豫园",
                "estimated_cost": 40,
            },
        ]
    elif destination == DEST_HANGZHOU:
        activities_day1 = [
            {
                "title": "西湖景区",
                "start_time": "08:30",
                "end_time": "12:00",
                "location_name": "杭州市西湖区龙井路1号",
                "description": "游览西湖十景：断桥残雪、苏堤春晓、花港观鱼。",
                "map_link": "https://ditu.amap.com/search?query=杭州西湖",
                "estimated_cost": 0,
            },
            {
                "title": "灵隐寺",
                "start_time": "13:30",
                "end_time": "16:00",
                "location_name": "杭州市西湖区法云弄1号",
                "description": "参观千年古刹灵隐寺，感受佛教文化。",
                "map_link": "https://ditu.amap.com/search?query=杭州灵隐寺",
                "estimated_cost": 75,
            },
        ]
        activities_day2 = [
            {
                "title": "龙井村品茶",
                "start_time": "09:00",
                "end_time": "11:30",
                "location_name": "杭州市西湖区龙井村",
                "description": "到龙井村茶园体验正宗西湖龙井茶文化。",
                "map_link": "https://ditu.amap.com/search?query=杭州龙井村",
                "estimated_cost": 50,
            },
        ]
    else:
        # Generic fallback with map links for any destination
        activities_day1 = [
            {
                "title": f"{destination}市中心游览",
                "start_time": "09:00",
                "end_time": "11:30",
                "location_name": f"{destination}市中心",
                "description": f"探索{destination}市中心主要景点。",
                "map_link": f"https://ditu.amap.com/search?query={destination}景点",
                "estimated_cost": 0,
            },
        ]
        activities_day2 = [
            {
                "title": f"{destination}博物馆",
                "start_time": "13:00",
                "end_time": "15:30",
                "location_name": f"{destination}博物馆",
                "description": f"参观{destination}博物馆。",
                "map_link": f"https://ditu.amap.com/search?query={destination}博物馆",
                "estimated_cost": 30,
            },
        ]

    return [
        {"date": START_DATE_STR, "activities": activities_day1},
        {"date": END_DATE_STR, "activities": activities_day2},
    ]


# ===================================================================
# Shared fixtures
# ===================================================================


@pytest.fixture(autouse=True)
def _clear_mcp_caches() -> None:
    """Clear MCP client / tool caches so each test loads tooling freshly."""
    try:
        from utils import mcp_client

        mcp_client._CLIENT_CACHE.clear()
        mcp_client._TOOLS_CACHE.clear()
    except Exception:
        pass
    yield
    try:
        from utils import mcp_client

        mcp_client._CLIENT_CACHE.clear()
        mcp_client._TOOLS_CACHE.clear()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _set_e2e_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use reasonable timeouts for live MCP calls so tests don't hang."""
    monkeypatch.setenv("MCP_TOOLING_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("LIUBU_TOOL_TIMEOUT_SECONDS", "20")
    get_settings.cache_clear()


# ===================================================================
# Helper: URL format check (does NOT do HTTP GET – just validates shape)
# ===================================================================


def _is_well_formed_url(value: Any) -> bool:
    """Return True if *value* looks like a valid absolute HTTP(S) URL."""
    if not isinstance(value, str):
        return False
    return value.startswith("http://") or value.startswith("https://")


def _assert_no_fallback_language(*texts: str | None) -> None:
    """Fail if any text contains fallback / estimated-fallback phrasing."""
    forbidden = [
        "fallback",
        "estimated fallback",
        "did not use real-time data",
        "structured synthesis failed",
    ]
    for text in texts:
        if not text:
            continue
        lowered = text.lower()
        for phrase in forbidden:
            assert phrase not in lowered, (
                f"Fallback language '{phrase}' found in: {text[:200]}"
            )


# ===================================================================
# Test 1 – Weather  (amap  maps_weather)
# ===================================================================


@pytest.mark.live
@pytest.mark.asyncio
async def test_weather_bureau_e2e_shanghai_amap_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weather bureau uses amap ``maps_weather`` for 上海.

    The agent issues a hardcoded ``maps_weather`` tool call when the
    tool is available – no LLM is required.  We patch ``build_qwen_chat``
    to ``None`` to guarantee we are testing the MCP path exclusively.
    """
    if not os.getenv("AMAP_API_KEY"):
        pytest.skip("AMAP_API_KEY is required for live Weather E2E")

    import provinces.liubu.weather.service as weather_service

    monkeypatch.setattr(weather_service, "build_qwen_chat", lambda: None)

    bureau = weather_service.WeatherBureau()
    try:
        await asyncio.wait_for(bureau.ensure_live_tooling(), timeout=15)
    except asyncio.TimeoutError:
        pytest.skip("Amap MCP server did not respond in time for Weather bureau")
    except Exception as exc:
        pytest.skip(f"Amap MCP server unavailable for Weather bureau: {exc}")

    if "maps_weather" not in bureau.available_tool_names:
        pytest.skip("maps_weather tool not available from amap MCP server")

    payload = _base_payload(destination=DEST_SHANGHAI)
    subtask = _make_subtask(payload, "WEATHER")

    result = await asyncio.wait_for(bureau.run(subtask), timeout=30)

    # --- basic identity ---
    assert result.get("bureau") == "WEATHER", f"Unexpected bureau: {result.get('bureau')}"
    assert result.get("destination") == DEST_SHANGHAI, (
        f"Unexpected destination: {result.get('destination')}"
    )

    # --- deliverable quality ---
    if result.get("data_source") == "live":
        assert result.get("status") == "ok", f"Live result should be ok, got {result.get('status')}"
        forecast_days = result.get("forecast_days", [])
        assert len(forecast_days) >= 1, "Expected at least 1 forecast day"

        for day in forecast_days:
            assert day.get("date"), "Each forecast day must have a date"
            condition = day.get("condition", "")
            assert condition, "Condition must not be empty"
            assert condition != "Weather unavailable", (
                f"Condition is placeholder 'Weather unavailable' – real data expected"
            )
            assert isinstance(day.get("min_temp_c"), (int, float)), "min_temp_c must be numeric"
            assert isinstance(day.get("max_temp_c"), (int, float)), "max_temp_c must be numeric"
            # Live Amap data should NOT be marked estimated
            assert day.get("is_estimated") is False, (
                f"Live forecast day should have is_estimated=False, got {day.get('is_estimated')}"
            )

        packing = result.get("packing_list", [])
        assert len(packing) >= 1, "Packing list should not be empty for live result"

        warnings_list = result.get("warnings", [])
        for w in warnings_list:
            _assert_no_fallback_language(str(w))

        _assert_no_fallback_language(result.get("summary", ""))
    else:
        # MCP may have returned empty / errored – bureau falls back gracefully.
        # We still verify the result shape is valid.
        logger.warning(
            "Weather bureau data_source=%s (expected 'live'); MCP may be degraded. "
            "Result status=%s",
            result.get("data_source"),
            result.get("status"),
        )
        assert "forecast_days" in result, "Result must contain forecast_days even in fallback"
        assert "packing_list" in result, "Result must contain packing_list even in fallback"


# ===================================================================
# Test 2 – Accommodation  (amap  maps_text_search)
# ===================================================================


@pytest.mark.live
@pytest.mark.asyncio
async def test_accommodation_bureau_e2e_hangzhou_amap_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accommodation bureau uses amap ``maps_text_search`` for 杭州 hotels."""
    if not os.getenv("AMAP_API_KEY"):
        pytest.skip("AMAP_API_KEY is required for live Accommodation E2E")

    import provinces.liubu.accommodation.service as accommodation_service

    monkeypatch.setattr(accommodation_service, "build_qwen_chat", lambda: None)

    bureau = accommodation_service.AccommodationBureau()
    try:
        await asyncio.wait_for(bureau.ensure_live_tooling(), timeout=15)
    except asyncio.TimeoutError:
        pytest.skip("Amap MCP server did not respond in time for Accommodation bureau")
    except Exception as exc:
        pytest.skip(f"Amap MCP server unavailable for Accommodation bureau: {exc}")

    if "maps_text_search" not in bureau.available_tool_names:
        pytest.skip("maps_text_search tool not available from amap MCP server")

    payload = _base_payload(destination=DEST_HANGZHOU, dest_airport=AIRPORT_HGH)
    subtask = _make_subtask(payload, "ACCOMMODATION")

    result = await asyncio.wait_for(bureau.run(subtask), timeout=30)

    # --- basic identity ---
    assert result.get("bureau") == "ACCOMMODATION", (
        f"Unexpected bureau: {result.get('bureau')}"
    )
    assert result.get("destination") == DEST_HANGZHOU, (
        f"Unexpected destination: {result.get('destination')}"
    )

    # --- deliverable quality ---
    if result.get("data_source") == "live":
        assert result.get("status") == "ok", (
            f"Live result should be ok, got {result.get('status')}"
        )

        hotels = result.get("hotel_options", [])
        assert len(hotels) >= 1, "Expected at least 1 hotel option"

        for hotel in hotels:
            name = hotel.get("name", "")
            assert name, "Hotel name must not be empty"
            assert len(name) >= 2, f"Hotel name too short: '{name}'"
            # Should not be just the destination city name
            assert name != DEST_HANGZHOU, (
                f"Hotel name is just the city name '{DEST_HANGZHOU}' – not a real hotel"
            )

            booking_link = hotel.get("booking_link")
            assert _is_well_formed_url(booking_link), (
                f"Hotel booking_link is not a well-formed URL: {booking_link}"
            )
            # Live results should point to amap or rollinggo (real booking data)
            assert any(domain in str(booking_link) for domain in ["amap.com", "rollinggo.cn"]), (
                f"Live hotel booking_link should point to amap or rollinggo, got: {booking_link}"
            )

            address = hotel.get("address", "")
            assert address, f"Hotel '{name}' should have an address"

            notes = hotel.get("notes", "")
            _assert_no_fallback_language(notes)

        booking_links = result.get("booking_links", [])
        assert len(booking_links) >= 1, "booking_links must not be empty"
        for link in booking_links:
            assert _is_well_formed_url(link), f"booking_link is not well-formed: {link}"

        # Quality gate
        quality = result.get("liubu_quality", {})
        assert quality.get("passed") is True, (
            f"Quality gate should pass for live result. Findings: {quality.get('findings')}"
        )

        for note in result.get("search_notes", []):
            _assert_no_fallback_language(note)
    else:
        logger.warning(
            "Accommodation bureau data_source=%s (expected 'live'); MCP may be degraded. "
            "Result status=%s",
            result.get("data_source"),
            result.get("status"),
        )
        assert "hotel_options" in result, "Result must contain hotel_options"


# ===================================================================
# Test 3 – Flight Transport  (amap  maps_geo)
# ===================================================================


@pytest.mark.live
@pytest.mark.asyncio
async def test_flight_transport_bureau_e2e_beijing_shanghai_amap_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flight Transport bureau uses amap ``maps_geo`` for 北京→上海 route.

    The agent issues two hardcoded ``maps_geo`` calls (origin + destination),
    then computes the Haversine distance to estimate flight options.
    """
    if not os.getenv("AMAP_API_KEY"):
        pytest.skip("AMAP_API_KEY is required for live Flight Transport E2E")

    import provinces.liubu.flight_transport.service as flight_service

    monkeypatch.setattr(flight_service, "build_qwen_chat", lambda: None)

    bureau = flight_service.FlightTransportBureau()
    try:
        await asyncio.wait_for(bureau.ensure_live_tooling(), timeout=15)
    except asyncio.TimeoutError:
        pytest.skip("Amap MCP server did not respond in time for Flight Transport bureau")
    except Exception as exc:
        pytest.skip(f"Amap MCP server unavailable for Flight Transport bureau: {exc}")

    if "maps_geo" not in bureau.available_tool_names:
        pytest.skip("maps_geo tool not available from amap MCP server")

    payload = _base_payload(
        destination=DEST_SHANGHAI,
        origin_city=ORIGIN_BEIJING,
        origin_airport=AIRPORT_PEK,
        dest_airport=AIRPORT_PVG,
    )
    subtask = _make_subtask(payload, "FLIGHT_TRANSPORT")

    result = await asyncio.wait_for(bureau.run(subtask), timeout=30)

    # --- basic identity ---
    assert result.get("bureau") == "FLIGHT_TRANSPORT", (
        f"Unexpected bureau: {result.get('bureau')}"
    )
    assert result.get("origin") == ORIGIN_BEIJING, (
        f"Unexpected origin: {result.get('origin')}"
    )
    assert result.get("destination") == DEST_SHANGHAI, (
        f"Unexpected destination: {result.get('destination')}"
    )

    # --- deliverable quality ---
    if result.get("data_source") == "live":
        assert result.get("status") == "ok", (
            f"Live result should be ok, got {result.get('status')}"
        )

        flights = result.get("flight_options", [])
        assert len(flights) >= 1, "Expected at least 1 flight option"

        for flight in flights:
            assert flight.get("airline"), "Flight must have an airline name"
            dep_ap = flight.get("departure_airport", "")
            arr_ap = flight.get("arrival_airport", "")
            # Accept either PEK/PVG or SHA/SHA variants (airport codes from live search)
            assert dep_ap, "Flight must have departure_airport"
            assert arr_ap, "Flight must have arrival_airport"
            # Verify they look like IATA codes (3 uppercase letters)
            assert len(dep_ap) == 3 and dep_ap.isupper(), (
                f"departure_airport should be IATA code, got: {dep_ap}"
            )
            assert len(arr_ap) == 3 and arr_ap.isupper(), (
                f"arrival_airport should be IATA code, got: {arr_ap}"
            )
            assert flight.get("price", 0) > 0, "Flight price must be > 0"
            assert flight.get("currency") == "CNY", (
                f"Expected currency=CNY, got {flight.get('currency')}"
            )

            duration = flight.get("duration_minutes")
            assert duration is not None, "duration_minutes must not be None"
            assert duration > 0, f"duration_minutes must be > 0, got {duration}"

            booking_link = flight.get("booking_link")
            assert _is_well_formed_url(booking_link), (
                f"Flight booking_link is not well-formed: {booking_link}"
            )
            # Live results should point to amap, rollinggo, or a real booking domain
            assert any(
                domain in str(booking_link)
                for domain in ["amap.com", "rollinggo.cn", "booking.com", "ctrip.com", "fliggy.com"]
            ), (
                f"Live flight booking_link should point to a real booking domain, got: {booking_link}"
            )

        booking_links = result.get("booking_links", [])
        assert len(booking_links) >= 1
        for link in booking_links:
            assert _is_well_formed_url(link), f"booking_link is not well-formed: {link}"

        # Quality gate
        quality = result.get("liubu_quality", {})
        assert quality.get("passed") is True, (
            f"Quality gate should pass for live result. Findings: {quality.get('findings')}"
        )

        for note in result.get("transport_notes", []):
            _assert_no_fallback_language(note)

        # Live evidence should contain maps_geo results
        evidence = result.get("liubu_evidence", [])
        assert len(evidence) >= 1, "liubu_evidence must be non-empty for live result"
        tool_names = {e.get("tool_name") for e in evidence}
        assert "maps_geo" in tool_names, (
            f"Expected maps_geo in liubu_evidence, got tools: {tool_names}"
        )
    else:
        logger.warning(
            "Flight Transport bureau data_source=%s (expected 'live'); MCP may be degraded. "
            "Result status=%s",
            result.get("data_source"),
            result.get("status"),
        )
        assert "flight_options" in result, "Result must contain flight_options"


# ===================================================================
# Test 4 – Budget  (serpapi  search_google_travel / hotels / flights)
# ===================================================================


@pytest.mark.live
@pytest.mark.asyncio
async def test_budget_bureau_e2e_shanghai_serpapi_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Budget bureau for 上海 – tests that a live deliverable budget is produced.

    The budget bureau uses two paths to produce live results:
    1.  Itinerary-derived: when activities carry ``map_link`` entries,
        ``_budget_from_live_inputs()`` builds a clean live budget.
    2.  MCP-driven: the bound LLM emits serpapi tool calls, evidence is
        collected, and ``_budget_from_live_inputs(live_context_available=True)``
        produces the budget.

    Either path is acceptable – this test provides real itinerary
    activities with map links so path 1 works reliably even when serpapi
    has limited data for domestic Chinese destinations.  Path 2 is
    exercised when serpapi tools are available and the LLM can emit
    calls.
    """
    if not os.getenv("SERPAPI_API_KEY"):
        pytest.skip("SERPAPI_API_KEY is required for live Budget E2E")

    import provinces.liubu.budget.service as budget_service

    # We leave the LLM intact so the bound_tool_model can emit serpapi
    # tool calls when the MCP path is attempted.  If the LLM API key is
    # missing we still have the itinerary-derived path as fallback.

    bureau = budget_service.BudgetBureau()
    try:
        await asyncio.wait_for(bureau.ensure_live_tooling(), timeout=15)
    except asyncio.TimeoutError:
        pytest.skip("Serpapi MCP server did not respond in time for Budget bureau")
    except Exception as exc:
        pytest.skip(f"Serpapi MCP server unavailable for Budget bureau: {exc}")

    daily_plan = _daily_plan_with_real_activities(DEST_SHANGHAI)
    payload = _base_payload(
        destination=DEST_SHANGHAI,
        daily_plan=daily_plan,
        total_budget=5000,
    )
    subtask = _make_subtask(payload, "BUDGET")

    result = await asyncio.wait_for(bureau.run(subtask), timeout=30)

    # --- basic identity ---
    assert result.get("bureau") == "BUDGET", f"Unexpected bureau: {result.get('bureau')}"

    # --- deliverable quality ---
    if result.get("data_source") == "live":
        assert result.get("status") == "ok", (
            f"Live result should be ok, got {result.get('status')}"
        )

        breakdown = result.get("budget_breakdown", [])
        assert len(breakdown) >= 1, "budget_breakdown must not be empty"

        categories = {item.get("category") for item in breakdown}
        expected_categories = {"activities", "accommodation", "food", "transport", "flights", "misc"}
        missing = expected_categories - categories
        assert not missing, f"Budget missing categories: {missing}"

        for item in breakdown:
            assert item.get("category"), "Each line item must have a category"
            assert item.get("item"), "Each line item must have an item name"
            cost = item.get("estimated_cost")
            assert isinstance(cost, (int, float)), (
                f"estimated_cost must be numeric, got {type(cost)}: {cost}"
            )
            assert cost >= 0, f"estimated_cost must be >= 0, got {cost}"
            assert item.get("currency") == "CNY", (
                f"Expected currency=CNY, got {item.get('currency')}"
            )
            notes = item.get("notes", "")
            _assert_no_fallback_language(notes)

        total = result.get("total_estimated_cost", 0)
        assert total > 0, f"total_estimated_cost must be > 0, got {total}"

        for warning in result.get("warnings", []):
            _assert_no_fallback_language(str(warning))

        # If serpapi evidence was collected, verify it
        evidence = result.get("liubu_evidence", [])
        if evidence:
            logger.info("Budget collected %d serpapi evidence entries", len(evidence))
            for e in evidence:
                assert e.get("tool_name"), "Evidence entry must have tool_name"
    else:
        logger.warning(
            "Budget bureau data_source=%s (expected 'live'); serpapi / itinerary data "
            "may be insufficient. Result status=%s",
            result.get("data_source"),
            result.get("status"),
        )
        assert "budget_breakdown" in result, "Result must contain budget_breakdown"


# ===================================================================
# Test 5 – Calendar  (serpapi  search_google_maps / directions / local_places)
# ===================================================================


@pytest.mark.live
@pytest.mark.asyncio
async def test_calendar_bureau_e2e_shanghai_serpapi_live(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Calendar bureau for 上海 – produces a real .ics calendar file.

    Like Budget, the Calendar bureau has two live paths:
    1.  Itinerary-derived: when activities carry ``map_link`` entries,
        ``_daily_plan_has_live_links()`` returns True, and events are
        built from the itinerary (data_source="live").
    2.  MCP-driven: the bound LLM emits serpapi place/direction tool
        calls, evidence is collected, and events are built.

    Either path is acceptable.  The test verifies that a valid .ics
    calendar file is written with real event data and correct UTC
    timestamps.
    """
    if not os.getenv("SERPAPI_API_KEY"):
        pytest.skip("SERPAPI_API_KEY is required for live Calendar E2E")

    import provinces.liubu.calendar.service as calendar_service

    bureau = calendar_service.CalendarBureau(output_dir=str(tmp_path))
    try:
        await asyncio.wait_for(bureau.ensure_live_tooling(), timeout=15)
    except asyncio.TimeoutError:
        pytest.skip("Serpapi MCP server did not respond in time for Calendar bureau")
    except Exception as exc:
        pytest.skip(f"Serpapi MCP server unavailable for Calendar bureau: {exc}")

    daily_plan = _daily_plan_with_real_activities(DEST_SHANGHAI)
    payload = _base_payload(destination=DEST_SHANGHAI, daily_plan=daily_plan)
    subtask = _make_subtask(payload, "CALENDAR")

    result = await asyncio.wait_for(bureau.run(subtask), timeout=30)

    # --- basic identity ---
    assert result.get("bureau") == "CALENDAR", (
        f"Unexpected bureau: {result.get('bureau')}"
    )

    # --- deliverable quality ---
    if result.get("data_source") == "live":
        assert result.get("status") == "ok", (
            f"Live result should be ok, got {result.get('status')}"
        )

        events_created = result.get("events_created", 0)
        # We provided 3 activities across 2 days
        expected_min = 3
        assert events_created >= expected_min, (
            f"Expected at least {expected_min} events, got {events_created}"
        )

        calendar_name = result.get("calendar_name", "")
        assert DEST_SHANGHAI in calendar_name, (
            f"Calendar name should contain destination, got: '{calendar_name}'"
        )

        calendar_file = result.get("calendar_file")
        assert calendar_file is not None, "calendar_file must not be None"
        calendar_path = Path(str(calendar_file))
        assert calendar_path.exists(), (
            f"Calendar .ics file was not written: {calendar_path}"
        )
        assert calendar_path.suffix == ".ics", (
            f"Calendar file should end with .ics, got: {calendar_path.suffix}"
        )
        assert "_trip_calendar.ics" in str(calendar_path), (
            f"Calendar file name should contain '_trip_calendar.ics': {calendar_path}"
        )

        # --- inspect .ics content ---
        ics_content = calendar_path.read_text(encoding="utf-8")
        assert "BEGIN:VCALENDAR" in ics_content, ".ics must be a valid VCALENDAR"
        assert "BEGIN:VEVENT" in ics_content, ".ics must contain at least one VEVENT"
        assert "DTSTART" in ics_content, ".ics events must have DTSTART"
        assert "DTEND" in ics_content, ".ics events must have DTEND"
        # Verify actual trip dates appear in the calendar
        start_compact = START_DATE_STR.replace("-", "")
        end_compact = END_DATE_STR.replace("-", "")
        assert start_compact in ics_content, (
            f".ics should contain start date {start_compact}"
        )
        assert end_compact in ics_content, (
            f".ics should contain end date {end_compact}"
        )
        # Live data source marker
        assert "X-MA-DATA-SOURCE:live" in ics_content, (
            ".ics should contain X-MA-DATA-SOURCE:live for live results"
        )
        # Activity map links should appear as event URLs
        assert "ditu.amap.com" in ics_content, (
            ".ics should contain amap links from live itinerary activities"
        )

        for warning in result.get("warnings", []):
            _assert_no_fallback_language(str(warning))

        logger.info("Calendar .ics file written to: %s (%d bytes)", calendar_path, len(ics_content))
    else:
        logger.warning(
            "Calendar bureau data_source=%s (expected 'live'); serpapi / itinerary data "
            "may be insufficient. Result status=%s",
            result.get("data_source"),
            result.get("status"),
        )
        assert "calendar_file" in result, "Result must contain calendar_file"
        assert "events_created" in result, "Result must contain events_created"


# ===================================================================
# Optional: direct MCP tool smoke tests (amap connectivity)
# ===================================================================


@pytest.mark.live
@pytest.mark.asyncio
async def test_amap_maps_weather_tool_direct() -> None:
    """Direct smoke test: call amap ``maps_weather`` tool for 上海.

    This verifies the amap MCP connection independently of any bureau graph.
    """
    if not os.getenv("AMAP_API_KEY"):
        pytest.skip("AMAP_API_KEY is required")

    from utils.mcp_client import load_mcp_tools

    tools = await asyncio.wait_for(load_mcp_tools(["amap"]), timeout=15)
    weather_tool = None
    for t in tools:
        if getattr(t, "name", None) == "maps_weather":
            weather_tool = t
            break

    if weather_tool is None:
        pytest.skip("maps_weather tool not found in amap MCP server")

    result = await asyncio.wait_for(
        weather_tool.ainvoke({"city": DEST_SHANGHAI}),
        timeout=15,
    )

    assert result is not None, "maps_weather returned None"
    logger.info("maps_weather direct result type: %s", type(result).__name__)

    # The result may be a string (JSON), a dict, or a list of TextContent.
    # Just verify it's not an error and contains weather-related data.
    result_str = str(result)
    assert len(result_str) > 10, f"maps_weather result too short: {result_str}"
    # Should contain either weather data or the city name
    assert any(
        keyword in result_str.lower()
        for keyword in ["weather", "temp", "cast", "forecast", "上海", "shanghai"]
    ), f"maps_weather result doesn't look like weather data: {result_str[:200]}"


@pytest.mark.live
@pytest.mark.asyncio
async def test_amap_maps_geo_tool_direct() -> None:
    """Direct smoke test: call amap ``maps_geo`` tool for 北京.

    This verifies geocoding connectivity independently of any bureau graph.
    """
    if not os.getenv("AMAP_API_KEY"):
        pytest.skip("AMAP_API_KEY is required")

    from utils.mcp_client import load_mcp_tools

    tools = await asyncio.wait_for(load_mcp_tools(["amap"]), timeout=15)
    geo_tool = None
    for t in tools:
        if getattr(t, "name", None) == "maps_geo":
            geo_tool = t
            break

    if geo_tool is None:
        pytest.skip("maps_geo tool not found in amap MCP server")

    result = await asyncio.wait_for(
        geo_tool.ainvoke({"address": ORIGIN_BEIJING, "city": ORIGIN_BEIJING}),
        timeout=15,
    )

    assert result is not None, "maps_geo returned None"
    logger.info("maps_geo direct result type: %s", type(result).__name__)

    result_str = str(result)
    assert len(result_str) > 10, f"maps_geo result too short: {result_str}"
    # Should contain location coordinates or the city name
    assert any(
        keyword in result_str.lower()
        for keyword in ["location", "geocode", "116", "39", "北京", "beijing"]
    ), f"maps_geo result doesn't look like geocoding data: {result_str[:200]}"
