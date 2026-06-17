import pytest

from provinces.liubu.flight_transport.service import FlightTransportBureau
from provinces.liubu.accommodation.service import AccommodationBureau


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


@pytest.mark.asyncio
async def test_flight_transport_fallback_uses_trip_date_when_profile_start_date_missing():
    bureau = FlightTransportBureau()
    payload = {
        "approved_draft": {
            "destination": "Tokyo",
            "itinerary_draft": {
                "destination": "Tokyo",
                "daily_plan": [{"date": "2026-05-01", "activities": []}],
            },
        },
        "execution_plan": {
            "user_request": {
                "profile": {
                    "origin_city": "Beijing",
                    "currency": "USD",
                }
            }
        },
    }

    result = await bureau.run(payload)

    assert all("TBD" not in option["departure_time"] for option in result["flight_options"])
    assert all(option["departure_time"].startswith("2026-05-01") for option in result["flight_options"])


@pytest.mark.asyncio
async def test_accommodation_fallback_uses_trip_night_count():
    bureau = AccommodationBureau()
    payload = {
        "approved_draft": {
            "destination": "Tokyo",
            "itinerary_draft": {
                "destination": "Tokyo",
                "daily_plan": [
                    {"date": "2026-05-01", "activities": []},
                    {"date": "2026-05-02", "activities": []},
                    {"date": "2026-05-03", "activities": []},
                    {"date": "2026-05-04", "activities": []},
                ],
            },
        },
        "execution_plan": {
            "user_request": {
                "profile": {
                    "currency": "USD",
                }
            }
        },
    }

    result = await bureau.run(payload)

    first_hotel = result["hotel_options"][0]
    assert first_hotel["nightly_rate"] == 140
    assert first_hotel["total_rate"] == 420
