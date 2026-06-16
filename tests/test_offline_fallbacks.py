import pytest

from provinces.liubu.flight_transport.service import FlightTransportBureau


@pytest.mark.asyncio
async def test_flight_transport_fallback_is_marked_as_estimated():
    bureau = FlightTransportBureau()
    payload = {
        "approved_draft": {
            "destination": "Tokyo",
            "itinerary_draft": {"destination": "Tokyo", "daily_plan": []},
        },
        "execution_plan": {
            "user_request": {
                "profile": {
                    "origin_city": "Beijing",
                    "currency": "USD",
                    "start_date": "2026-05-01",
                }
            }
        },
    }

    result = await bureau.run(payload)

    assert result["bureau"] == "FLIGHT_TRANSPORT"
    assert all(option["airline"] != "Placeholder Air" for option in result["flight_options"])
    assert any("estimated" in note.lower() for note in result["transport_notes"])
    assert any("not use real-time data" in option["notes"].lower() for option in result["flight_options"])
