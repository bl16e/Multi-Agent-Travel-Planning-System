import asyncio

import pytest

from provinces.liubu.constrained.state import normalize_worker_input
from provinces.liubu.constrained.tools import execute_constrained_tool_call


class FakeTool:
    name = "google_flights"

    async def ainvoke(self, args):
        return {
            "search_parameters": args,
            "best_flights": [{"airline": "ANA", "price": 500, "departure_airport": "PEK", "arrival_airport": "HND"}],
            "secret": "must not be retained",
        }


class SlowTool:
    name = "google_flights"

    async def ainvoke(self, args):
        await asyncio.sleep(1)
        return {"ok": True}


def _worker_input():
    payload = {
        "request_id": "tool_constraints",
        "target_bureau": "FLIGHT_TRANSPORT",
        "approved_draft": {
            "destination": "Tokyo",
            "itinerary_draft": {"destination": "Tokyo", "daily_plan": [{"date": "2026-10-01", "activities": []}]},
        },
        "execution_plan": {
            "user_request": {
                "profile": {
                    "origin_city": "Beijing",
                    "origin_airport_code": "PEK",
                    "destination_airport_code": "HND",
                    "start_date": "2026-10-01",
                    "end_date": "2026-10-01",
                    "adults": 2,
                    "currency": "USD",
                }
            }
        },
    }
    return normalize_worker_input(payload, "FLIGHT_TRANSPORT")


@pytest.mark.asyncio
async def test_execute_constrained_tool_call_blocks_unsupported_tool():
    evidence = await execute_constrained_tool_call(
        worker_input=_worker_input(),
        tool_map={"google_flights": FakeTool()},
        allowed_tool_names={"google_flights"},
        tool_name="google_hotels",
        args={},
    )

    assert evidence.status == "blocked"
    assert evidence.error == "Tool google_hotels is not allowed for FLIGHT_TRANSPORT."


@pytest.mark.asyncio
async def test_execute_constrained_tool_call_blocks_wrong_airport_date_adults_and_currency():
    evidence = await execute_constrained_tool_call(
        worker_input=_worker_input(),
        tool_map={"google_flights": FakeTool()},
        allowed_tool_names={"google_flights"},
        tool_name="google_flights",
        args={
            "departure_id": "SHA",
            "arrival_id": "NRT",
            "outbound_date": "2023-10-01",
            "adults": 1,
            "currency": "JPY",
        },
    )

    assert evidence.status == "blocked"
    assert "departure_id must match PEK" in evidence.error
    assert "arrival_id must match HND" in evidence.error
    assert "outbound_date must match 2026-10-01" in evidence.error
    assert "adults must match 2" in evidence.error
    assert "currency must match USD" in evidence.error


@pytest.mark.asyncio
async def test_execute_constrained_tool_call_records_sanitized_success():
    evidence = await execute_constrained_tool_call(
        worker_input=_worker_input(),
        tool_map={"google_flights": FakeTool()},
        allowed_tool_names={"google_flights"},
        tool_name="google_flights",
        args={"departure_id": "PEK", "arrival_id": "HND", "outbound_date": "2026-10-01", "adults": 2, "currency": "USD", "api_key": "abc"},
    )

    assert evidence.status == "ok"
    assert evidence.args["api_key"] == "[REDACTED]"
    assert evidence.result["search_parameters"]["departure_id"] == "PEK"
    assert "secret" not in evidence.result


@pytest.mark.asyncio
async def test_execute_constrained_tool_call_records_timeout():
    evidence = await execute_constrained_tool_call(
        worker_input=_worker_input(),
        tool_map={"google_flights": SlowTool()},
        allowed_tool_names={"google_flights"},
        tool_name="google_flights",
        args={"departure_id": "PEK", "arrival_id": "HND", "outbound_date": "2026-10-01", "adults": 2, "currency": "USD"},
        timeout_seconds=0.01,
    )

    assert evidence.status == "timeout"
    assert evidence.error == "Tool google_flights timed out after 0.01s."
