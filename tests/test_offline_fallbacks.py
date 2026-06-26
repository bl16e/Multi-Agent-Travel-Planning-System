import pytest

import utils.agent_runtime as agent_runtime
from provinces.liubu.constrained.state import normalize_worker_input
from provinces.liubu.flight_transport.service import FlightTransportBureau
from provinces.liubu.accommodation.service import AccommodationBureau
from provinces.liubu.budget.service import BudgetBureau
from utils.schemas import ItineraryDraftModel
from workflow import build_markdown


def _liubu_subtask(payload: dict, bureau: str) -> dict:
    return {"worker_input": normalize_worker_input(payload, bureau)}


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

    result = await bureau.run(_liubu_subtask(payload, "FLIGHT_TRANSPORT"))

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

    result = await bureau.run(_liubu_subtask(payload, "FLIGHT_TRANSPORT"))

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

    result = await bureau.run(_liubu_subtask(payload, "ACCOMMODATION"))

    first_hotel = result["hotel_options"][0]
    assert first_hotel["nightly_rate"] == 140
    assert first_hotel["total_rate"] == 420


def test_offline_zhongshu_draft_has_provenance_and_no_generic_placeholders():
    draft = agent_runtime._offline_structured_output(
        ItineraryDraftModel,
        {
            "destination": "Kyoto",
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "interests": "temples, food",
            "research_context": agent_runtime.FALLBACK_MESSAGE,
        },
    )

    serialized = str(draft.model_dump(mode="json")).lower()
    assert any("data_source=fallback_estimate" in note for note in draft.planning_notes)
    assert "orientation walk" not in serialized
    assert "fallback activity" not in serialized
    assert "estimated attraction slot" not in serialized
    assert "kyoto" in serialized


@pytest.mark.asyncio
async def test_liubu_fallback_outputs_expose_status_and_data_source():
    payload = {
            "approved_draft": {
                "destination": "Kyoto",
                "itinerary_draft": {
                    "destination": "Kyoto",
                    "daily_plan": [{"date": "2026-05-01", "activities": []}],
                },
            },
            "execution_plan": {
                "user_request": {
                    "profile": {
                        "currency": "USD",
                        "total_budget": 1000,
                    }
                }
            },
        }
    result = await BudgetBureau().run(_liubu_subtask(payload, "BUDGET"))

    assert result["bureau"] == "BUDGET"
    assert result["status"] == "fallback"
    assert result["data_source"] == "fallback_estimate"
    assert any("fallback" in warning.lower() for warning in result["warnings"])


def test_shangshu_markdown_renders_bureau_data_source_labels(tmp_path):
    request = type("Request", (), {"request_id": "offline_labels", "profile": type("Profile", (), {"currency": "USD"})()})()
    path = build_markdown(
        request,
        {
            "destination": "Kyoto",
            "itinerary_draft": {
                "daily_plan": [
                    {
                        "day_index": 1,
                        "date": "2026-05-01",
                        "theme": "Temples",
                        "summary": "Visit named temples.",
                        "activities": [],
                    }
                ]
            },
        },
        {"verdict": "APPROVED", "data_source": "fallback_estimate", "warnings": ["offline review"]},
        {
            "WEATHER": {
                "bureau": "WEATHER",
                "status": "fallback",
                "data_source": "fallback_estimate",
                "summary": "Estimated weather",
                "forecast_days": [],
                "packing_list": [],
                "warnings": ["offline weather"],
            },
            "BUDGET": {
                "bureau": "BUDGET",
                "status": "fallback",
                "data_source": "fallback_estimate",
                "currency": "USD",
                "budget_breakdown": [],
                "total_estimated_cost": 0,
                "warnings": ["offline budget"],
            },
        },
        "http://testserver/dashboard/offline_labels",
        tmp_path,
    )

    content = path.read_text(encoding="utf-8")
    assert "## Data Sources" in content
    assert "WEATHER: fallback / fallback_estimate" in content
    assert "BUDGET: fallback / fallback_estimate" in content
    assert "Menxia Review: fallback_estimate" in content
