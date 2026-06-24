import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from provinces.liubu.constrained.state import normalize_worker_input
from provinces.liubu.constrained.tool_node import run_constrained_tool_node, wrap_constrained_tools
from provinces.liubu.flight_transport.service import FlightTransportBureau


def _flight_worker_input():
    return normalize_worker_input(
        {
            "request_id": "tool_node_flight",
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
        },
        "FLIGHT_TRANSPORT",
    )


@pytest.mark.asyncio
async def test_constrained_tool_node_executes_ai_message_tool_calls_and_records_evidence():
    @tool
    async def google_flights(departure_id: str, arrival_id: str, outbound_date: str, adults: int, currency: str):
        """Search flights."""
        return {"route": f"{departure_id}-{arrival_id}", "outbound_date": outbound_date, "adults": adults, "currency": currency}

    worker_input = _flight_worker_input()
    wrapped_tools = wrap_constrained_tools(worker_input, [google_flights], {"google_flights"})
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "google_flights",
                    "args": {"departure_id": "PEK", "arrival_id": "HND", "outbound_date": "2026-10-01", "adults": 2, "currency": "USD"},
                    "id": "flight_call_1",
                    "type": "tool_call",
                }
            ],
        )
    ]

    tool_messages, evidence = await run_constrained_tool_node(messages=messages, tools=wrapped_tools)

    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "flight_call_1"
    payload = json.loads(tool_messages[0].content)
    assert payload["status"] == "ok"
    assert payload["args"]["departure_id"] == "PEK"
    assert evidence[0].status == "ok"
    assert evidence[0].tool_name == "google_flights"


@pytest.mark.asyncio
async def test_flight_agent_reasoning_uses_ai_message_tool_calls_without_llm(monkeypatch):
    import provinces.liubu.flight_transport.service as flight_service

    monkeypatch.setattr(flight_service, "build_qwen_chat", lambda: None)
    bureau = FlightTransportBureau()
    state = await bureau.prepare_context(
        {
            "payload": {
                "request_id": "message_agent",
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
        }
    )

    update = await bureau.agent_reasoning(state)

    assert "tool_requests" not in update
    assert isinstance(update["messages"][0], AIMessage)
    assert update["messages"][0].tool_calls[0]["name"] == "google_flights"
    assert update["messages"][0].tool_calls[0]["args"]["departure_id"] == "PEK"
