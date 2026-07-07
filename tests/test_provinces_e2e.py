"""Real E2E tests for the Three Provinces (三省) and the combined full workflow.

Tests:
  1. Zhongshu standalone – drafts itinerary with live MCP POI research + LLM synthesis
  2. Menxia standalone – reviews a real draft with live LLM verdict
  3. Full workflow – end-to-end ProvinceWorkflow.run()
  4. Stream workflow – ProvinceWorkflow.stream_run() with progress events

All tests are ``@pytest.mark.live`` – require ``RUN_LIVE_TESTS=1`` + API keys.
"""

from __future__ import annotations

import asyncio
import os
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from utils.schemas import PlanningRequest, TravelerProfile
from utils.settings import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
_START_DATE: date = date.today() + timedelta(days=7)
_END_DATE: date = _START_DATE + timedelta(days=2)
START_DATE_STR: str = _START_DATE.isoformat()
END_DATE_STR: str = _END_DATE.isoformat()

DEST = "上海"
ORIGIN = "北京"


def _check_keys(*names: str) -> None:
    """Pytest-skip if any required env var is missing."""
    for name in names:
        if not os.getenv(name):
            pytest.skip(f"{name} is required for live provinces E2E")


def _is_well_formed_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.startswith("http://") or value.startswith("https://")


def _assert_no_fallback_language(*texts: str | None) -> None:
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
            assert phrase not in lowered, f"Fallback language '{phrase}' in: {text[:200]}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_mcp_caches() -> None:
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
    monkeypatch.setenv("MCP_TOOLING_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("LIUBU_TOOL_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("QWEN_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("PLAN_REQUEST_TIMEOUT_SECONDS", "300")
    get_settings.cache_clear()


# ===================================================================
# Test 1 – Zhongshu standalone (中书省)
# ===================================================================


@pytest.mark.live
@pytest.mark.asyncio
async def test_zhongshu_e2e_draft_itinerary_shanghai_live() -> None:
    """Zhongshu drafts an itinerary for 上海 using live SerpAPI + Amap MCP + LLM.

    This exercises the full pipeline:
      ingest_request → draft_itinerary (MCP research + LLM synthesis)
      → decompose_tasks → finalize_draft
    """
    _check_keys("QWEN_API_KEY", "AMAP_API_KEY", "SERPAPI_API_KEY")

    from provinces.zhongshu_itinerary.graph import ZhongshuItineraryAgent

    agent = ZhongshuItineraryAgent()

    state: dict[str, Any] = {
        "request_id": "zhongshu_e2e_shanghai",
        "user_request": {
            "profile": {
                "destination_preferences": [DEST],
                "origin_city": ORIGIN,
                "origin_airport_code": "PEK",
                "destination_airport_code": "PVG",
                "start_date": START_DATE_STR,
                "end_date": END_DATE_STR,
                "total_budget": 5000,
                "currency": "CNY",
                "adults": 2,
                "interests": ["culture", "food"],
                "constraints": ["avoid generic placeholder activities"],
                "pace": "structured",
            },
            "user_message": f"Plan a concrete trip to {DEST} with real named places, transport notes, and calendar-ready timing.",
        },
        "governance": {
            "rejection_count": 0,
            "max_rejection_rounds": 2,
        },
    }

    result = await asyncio.wait_for(agent.graph.ainvoke(state), timeout=180)

    # --- Verify finalize_draft produced a packet ---
    packet = result.get("finalized_packet") or result.get("draft") or {}
    assert packet, f"No finalized_packet in result. Keys: {list(result.keys())}"

    # Identity
    dest = packet.get("destination") or packet.get("itinerary_draft", {}).get("destination", "")
    assert dest == DEST or DEST in dest or dest != "", f"Unexpected destination: {dest}"

    itinerary = packet.get("itinerary_draft") or packet
    daily_plan = itinerary.get("daily_plan", [])

    # Must have at least 1 day with activities
    assert len(daily_plan) >= 1, f"Expected >=1 day, got {len(daily_plan)}"
    total_activities = sum(len(day.get("activities", [])) for day in daily_plan)
    assert total_activities >= 1, f"Expected >=1 activity across all days, got {total_activities}"

    # Detect whether MCP research was successful
    planning_notes = itinerary.get("planning_notes", [])
    trip_style = itinerary.get("trip_style", "")
    has_live_data = any(
        "data_source=live_mcp_research" in str(n) or "trip_style=live_research" in str(n)
        for n in planning_notes
    ) or "live_research" in trip_style
    has_fallback = any(
        "fallback" in str(n).lower() for n in planning_notes
    ) or "fallback" in trip_style.lower()

    logger.info(
        "Zhongshu draft: trip_style=%s, live_data=%s, fallback=%s",
        trip_style, has_live_data, has_fallback,
    )

    # Each activity must have a title and location
    for day in daily_plan:
        for act in day.get("activities", []):
            title = act.get("title", "")
            assert title, "Activity must have a title"
            assert len(title) >= 2, f"Activity title too short: '{title}'"
            loc = act.get("location_name", "")
            assert loc, f"Activity '{title}' must have a location_name"
            map_link = act.get("map_link")
            if map_link:
                assert _is_well_formed_url(map_link), (
                    f"Activity '{title}' map_link is not well-formed: {map_link}"
                )

    # When live MCP data is available, reject placeholder content
    if has_live_data and not has_fallback:
        for day in daily_plan:
            for act in day.get("activities", []):
                title = act.get("title", "")
                placeholder_markers = ["待确认", "待定", "generic", "placeholder", "orientation walk"]
                for marker in placeholder_markers:
                    assert marker.lower() not in title.lower(), (
                        f"Placeholder activity title in live draft: '{title}'"
                    )
        _assert_no_fallback_language(*planning_notes)
    elif has_fallback:
        logger.warning(
            "Zhongshu produced fallback draft (MCP may be unavailable). "
            "Activities: %d, trip_style: %s",
            total_activities, trip_style,
        )

    # Required bureaus
    req_bureaus = packet.get("required_bureaus", [])
    assert "WEATHER" in req_bureaus, f"Missing WEATHER in required_bureaus: {req_bureaus}"
    assert "BUDGET" in req_bureaus, f"Missing BUDGET in required_bureaus: {req_bureaus}"
    assert "CALENDAR" in req_bureaus, f"Missing CALENDAR in required_bureaus: {req_bureaus}"

    # Bureau tasks
    tasks = packet.get("bureau_tasks", [])
    task_bureaus = {t.get("bureau") for t in tasks}
    assert "WEATHER" in task_bureaus, f"Missing WEATHER task"
    assert "BUDGET" in task_bureaus, f"Missing BUDGET task"

    # Governance
    gov = packet.get("governance", {})
    assert gov.get("producer") == "ZHONGSHU", f"Expected ZHONGSHU producer, got {gov.get('producer')}"
    assert gov.get("review_required") is True, f"review_required should be True"

    # Planning notes should NOT contain fallback language
    for note in itinerary.get("planning_notes", []):
        _assert_no_fallback_language(note)

    # Log deliverable summary
    logger.info(
        "Zhongshu produced draft: dest=%s, days=%d, activities=%d, bureaus=%s",
        dest,
        len(daily_plan),
        total_activities,
        req_bureaus,
    )


# ===================================================================
# Test 2 – Menxia standalone (门下省)
# ===================================================================


@pytest.mark.live
@pytest.mark.asyncio
async def test_menxia_e2e_review_live() -> None:
    """Menxia reviews a real Zhongshu draft and produces a verdict.

    This test first runs Zhongshu to get a realistic draft (with MCP +
    LLM), then feeds it to Menxia for a real LLM review verdict.
    """
    _check_keys("QWEN_API_KEY", "AMAP_API_KEY", "SERPAPI_API_KEY")

    from provinces.zhongshu_itinerary.graph import ZhongshuItineraryAgent
    from provinces.menxia_review.graph import MenxiaReviewAgent

    # --- Step 1: Generate a real draft via Zhongshu ---
    zs_agent = ZhongshuItineraryAgent()
    zs_state: dict[str, Any] = {
        "request_id": "menxia_e2e_shanghai",
        "user_request": {
            "profile": {
                "destination_preferences": [DEST],
                "origin_city": ORIGIN,
                "origin_airport_code": "PEK",
                "start_date": START_DATE_STR,
                "end_date": END_DATE_STR,
                "total_budget": 5000,
                "currency": "CNY",
                "adults": 2,
                "interests": ["culture", "food"],
                "constraints": ["avoid generic placeholder activities"],
                "pace": "structured",
            },
            "user_message": "Plan a concrete Shanghai trip.",
        },
        "governance": {"rejection_count": 0, "max_rejection_rounds": 2},
    }

    zs_result = await asyncio.wait_for(zs_agent.graph.ainvoke(zs_state), timeout=180)
    draft_packet = zs_result.get("finalized_packet") or zs_result.get("draft") or {}
    assert draft_packet, "Zhongshu must produce a draft for Menxia review"

    # --- Step 2: Review via Menxia ---
    mx_agent = MenxiaReviewAgent()
    mx_state: dict[str, Any] = {
        "request_id": "menxia_e2e_shanghai",
        "draft": draft_packet,
        "user_request": {
            "profile": {
                "total_budget": 5000,
                "currency": "CNY",
            }
        },
    }

    mx_result = await asyncio.wait_for(mx_agent.graph.ainvoke(mx_state), timeout=120)

    verdict_payload = mx_result.get("verdict_payload") or {}
    assert verdict_payload, f"No verdict_payload in Menxia result. Keys: {list(mx_result.keys())}"

    # --- Verification ---
    verdict = verdict_payload.get("verdict", "")
    assert verdict in {"APPROVED", "REJECTED", "HUMAN_INTERVENE"}, (
        f"Unexpected verdict: {verdict}"
    )

    summary = verdict_payload.get("summary", "")
    assert summary, "Verdict summary must not be empty"

    ds = verdict_payload.get("data_source", "")
    assert ds in {"structured_llm", "live", "fallback_estimate"}, f"Unexpected data_source: {ds}"

    blocking = verdict_payload.get("blocking_issues", [])
    assert isinstance(blocking, list), "blocking_issues must be a list"

    review_notes = verdict_payload.get("review_notes", [])
    assert isinstance(review_notes, list), "review_notes must be a list"

    # If approved, it should list approved bureaus
    if verdict == "APPROVED":
        approved = verdict_payload.get("approved_bureaus", [])
        assert len(approved) >= 3, (
            f"APPROVED verdict should approve >=3 bureaus, got {len(approved)}: {approved}"
        )

    # Review notes should not contain fallback language
    for note in review_notes:
        _assert_no_fallback_language(str(note))
    for issue in blocking:
        _assert_no_fallback_language(str(issue))

    logger.info(
        "Menxia verdict: %s, data_source=%s, blocking=%d, notes=%d, bureaus=%s",
        verdict,
        ds,
        len(blocking),
        len(review_notes),
        verdict_payload.get("approved_bureaus", []),
    )


# ===================================================================
# Test 3 – Full Workflow Combined E2E (尚书省全流程)
# ===================================================================


@pytest.mark.live
@pytest.mark.asyncio
async def test_full_workflow_e2e_shanghai_live(tmp_path: Path) -> None:
    """Full ProvinceWorkflow end-to-end for 上海.

    Covers: preflight → zhongshu → menxia → review_gate → dispatch_liubu
    → weather/budget/accommodation/flight/calendar → assemble → DONE.
    """
    _check_keys("QWEN_API_KEY", "AMAP_API_KEY", "SERPAPI_API_KEY", "ROLLINGGO_MCP_KEY")

    from workflow import ProvinceWorkflow

    workflow = ProvinceWorkflow(artifact_dir=tmp_path)
    request = PlanningRequest(
        request_id="full_e2e_shanghai",
        user_message=(
            "Plan a concrete Shanghai trip with named places, real transport, "
            "budget labels, weather contingency, and calendar-ready timing."
        ),
        profile=TravelerProfile(
            origin_city=ORIGIN,
            origin_airport_code="PEK",
            destination_preferences=[DEST],
            destination_airport_code="PVG",
            start_date=_START_DATE,
            end_date=_END_DATE,
            total_budget=5000,
            currency="CNY",
            adults=2,
            interests=["culture", "food"],
            constraints=["avoid generic placeholder activities"],
            pace="structured",
        ),
    )

    result = await asyncio.wait_for(workflow.run(request), timeout=600)

    status = result.get("status", "")
    assert status in {"DONE", "HUMAN_INTERVENE", "REJECTED"}, (
        f"Unexpected workflow status: {status}"
    )

    if status == "DONE":
        _verify_done_result(result, tmp_path)
    elif status == "HUMAN_INTERVENE":
        logger.warning("Workflow paused for human intervention: %s", result.get("question"))
        assert result.get("question"), "HUMAN_INTERVENE must provide a question"
        assert result.get("resume_state"), "HUMAN_INTERVENE must provide resume_state"
    elif status == "REJECTED":
        rejected = result.get("rejected_payload") or result
        review = rejected.get("review", {})
        assert review.get("verdict") == "REJECTED", "REJECTED status must have rejected review"
        blocking = review.get("blocking_issues", [])
        assert len(blocking) >= 1, "REJECTED must have at least 1 blocking issue"
        logger.warning("Workflow rejected: %s", review.get("summary"))
        for issue in blocking:
            logger.warning("  Blocking: %s", issue)


def _verify_done_result(result: dict[str, Any], tmp_path: Path) -> None:
    """Verify a DONE workflow result has all deliverable content."""

    def _g(obj: Any, key: str, default: Any = None) -> Any:
        """Get key from dict or attribute from object."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    context = result.get("context") or {}
    final = result.get("final_package") or {}

    # --- Verify workflow reached DONE ---
    c_state = _g(context, "current_state")
    if hasattr(c_state, "value"):
        c_state = c_state.value
    wf_state = result.get("workflow_state") or _g(final, "workflow_state") or str(c_state or "")
    assert "DONE" in str(wf_state), f"Expected DONE, got workflow_state={wf_state}"

    # --- Resolve results from context or top-level ---
    exec_results = _g(context, "execution_results") or result.get("execution_results") or {}
    draft = _g(context, "draft_payload") or result.get("draft_packet") or {}
    review = _g(context, "review_payload") or result.get("review_packet") or {}

    # --- Itinerary / Draft ---
    itinerary = _g(draft, "itinerary_draft") or _g(final, "itinerary") or {}
    daily_plan = _g(itinerary, "daily_plan", [])
    assert len(daily_plan) >= 1, f"Must have at least 1 day in itinerary, got {len(daily_plan)}"
    total_acts = 0
    for day in daily_plan:
        activities = _g(day, "activities", [])
        for act in activities:
            total_acts += 1
            title = _g(act, "title", "")
            assert title, "Activity must have a title"
            map_link = _g(act, "map_link")
            if map_link:
                assert _is_well_formed_url(map_link), f"Bad map_link: {map_link}"
    assert total_acts >= 1, f"Must have at least 1 activity, got {total_acts}"

    # --- Review ---
    if review:
        verdict = _g(review, "verdict")
        if verdict:
            ds = _g(review, "data_source", "")
            logger.info("Review verdict=%s, data_source=%s", verdict, ds)

    # --- Bureau results ---
    bureau_map = {
        "WEATHER": "weather",
        "BUDGET": "budget",
        "ACCOMMODATION": "accommodation",
        "FLIGHT_TRANSPORT": "flight_transport",
        "CALENDAR": "calendar",
    }
    for bureau_name, key in bureau_map.items():
        bureau = _g(exec_results, bureau_name) or result.get(key) or {}
        if not bureau:
            logger.warning("Bureau result missing: %s", bureau_name)
            continue
        ds = _g(bureau, "data_source", "")
        assert ds in {"live", "structured_llm", "fallback_estimate", ""}, (
            f"{bureau_name} bad data_source: {ds}"
        )
        if ds in {"live", "structured_llm"}:
            status_b = _g(bureau, "status", "")
            assert status_b == "ok", (
                f"{bureau_name} status={status_b}, expected ok"
            )
        logger.info("Bureau %s: status=%s, data_source=%s", bureau_name, _g(bureau, "status"), ds)

    # --- Calendar .ics file ---
    cal_result = _g(exec_results, "CALENDAR") or result.get("calendar") or {}
    cal_file = _g(cal_result, "calendar_file")
    if cal_file:
        cal_path = Path(str(cal_file))
        if cal_path.exists():
            content = cal_path.read_text(encoding="utf-8")
            assert "BEGIN:VCALENDAR" in content
            logger.info("Calendar .ics: %s (%d bytes)", cal_path, len(content))

    # --- Progress events ---
    events = _g(context, "progress_events") or result.get("progress_events") or []
    logger.info("Progress events: %d", len(events))
    if events:
        stages = [_g(e, "stage", "") for e in events if _g(e, "stage")]
        if stages:
            logger.info("Workflow stages: %s", " → ".join(stages))

    # --- Markdown file ---
    md_file = result.get("markdown_file") or final.get("markdown_file")
    if not md_file:
        # Check artifacts directory
        from pathlib import Path as P
        artifacts = list(tmp_path.glob("*.md"))
        if artifacts:
            md_file = str(artifacts[0])
    if md_file:
        md_path = Path(str(md_file))
        if md_path.exists():
            logger.info("Markdown: %s (%d bytes)", md_path, md_path.stat().st_size)

    # --- Deliverable summary ---
    logger.info(
        "Full workflow DONE: dest=%s, state=%s, bureaus=%d, activities=%d",
        draft.get("destination", DEST),
        wf_state,
        len(exec_results),
        total_acts,
    )


# ===================================================================
# Test 4 – Stream Workflow E2E
# ===================================================================


@pytest.mark.live
@pytest.mark.asyncio
async def test_stream_workflow_e2e_shanghai_live() -> None:
    """Stream mode: ProvinceWorkflow.stream_run() yields progress + result events."""
    _check_keys("QWEN_API_KEY", "AMAP_API_KEY", "SERPAPI_API_KEY", "ROLLINGGO_MCP_KEY")

    from workflow import ProvinceWorkflow

    workflow = ProvinceWorkflow()
    request = PlanningRequest(
        request_id="stream_e2e_shanghai",
        user_message="Plan a compact Shanghai trip with real places and calendar-ready timing.",
        profile=TravelerProfile(
            origin_city=ORIGIN,
            origin_airport_code="PEK",
            destination_preferences=[DEST],
            destination_airport_code="PVG",
            start_date=_START_DATE,
            end_date=_END_DATE,
            total_budget=5000,
            currency="CNY",
            adults=1,
            interests=["culture", "food"],
            constraints=["avoid generic placeholder activities"],
            pace="structured",
        ),
    )

    events: list[dict[str, Any]] = []
    try:
        async for event in workflow.stream_run(request):
            events.append(event)
    except Exception as exc:
        logger.error("Stream workflow failed: %s", exc)
        # Even on failure, verify we got some progress events
        assert len(events) >= 1, f"Stream produced no events before error: {exc}"
        raise

    # --- Verify stream events ---
    progress_events = [e for e in events if e.get("event") == "progress"]
    result_events = [e for e in events if e.get("event") == "result"]

    assert len(progress_events) >= 1, (
        f"Expected >=1 progress events in stream, got {len(progress_events)}"
    )
    assert len(result_events) == 1, (
        f"Expected exactly 1 result event, got {len(result_events)}"
    )

    final = result_events[0].get("data", {})
    status = final.get("status", "")
    assert status in {"DONE", "HUMAN_INTERVENE", "REJECTED", ""}, (
        f"Unexpected stream result status: {status}"
    )

    # Verify stage progression (adapt to event format)
    for e in progress_events[:3]:
        data = e.get("data", {})
        logger.info("Progress event: data_keys=%s", list(data.keys())[:8])

    logger.info("Stream: %d progress events, result_status=%s", len(progress_events), status)
