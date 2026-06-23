from provinces.liubu.constrained.gates import gate_accommodation_result, gate_flight_transport_result
from provinces.liubu.constrained.state import LiubuToolEvidence, normalize_worker_input


def _flight_input():
    return normalize_worker_input(
        {
            "request_id": "flight_gate",
            "approved_draft": {"destination": "Tokyo", "itinerary_draft": {"destination": "Tokyo", "daily_plan": [{"date": "2026-10-01", "activities": []}]}},
            "execution_plan": {"user_request": {"profile": {"origin_city": "Beijing", "origin_airport_code": "PEK", "destination_airport_code": "HND", "start_date": "2026-10-01", "end_date": "2026-10-01", "adults": 1, "currency": "USD"}}},
        },
        "FLIGHT_TRANSPORT",
    )


def _hotel_input():
    return normalize_worker_input(
        {
            "request_id": "hotel_gate",
            "approved_draft": {"destination": "Tokyo", "itinerary_draft": {"destination": "Tokyo", "daily_plan": [{"date": "2026-10-01", "activities": []}, {"date": "2026-10-02", "activities": []}]}},
            "execution_plan": {"user_request": {"profile": {"start_date": "2026-10-01", "end_date": "2026-10-02", "adults": 2, "currency": "USD"}}},
        },
        "ACCOMMODATION",
    )


def test_flight_gate_accepts_live_evidence_with_matching_airports_currency_and_date():
    findings = gate_flight_transport_result(
        _flight_input(),
        {
            "bureau": "FLIGHT_TRANSPORT",
            "status": "ok",
            "data_source": "live",
            "origin": "Beijing",
            "destination": "Tokyo",
            "flight_options": [{"airline": "ANA", "price": 500, "currency": "USD", "departure_airport": "PEK", "arrival_airport": "HND", "departure_time": "2026-10-01 08:00", "arrival_time": "2026-10-01 12:00"}],
            "transport_notes": [],
            "booking_links": [],
        },
        [LiubuToolEvidence(tool_name="google_flights", status="ok", args={"departure_id": "PEK", "arrival_id": "HND", "outbound_date": "2026-10-01"})],
    )

    assert findings == []


def test_flight_gate_rejects_clean_success_without_live_evidence_and_matching_facts():
    findings = gate_flight_transport_result(
        _flight_input(),
        {
            "bureau": "FLIGHT_TRANSPORT",
            "status": "ok",
            "data_source": "structured_llm",
            "origin": "Beijing",
            "destination": "Tokyo",
            "flight_options": [{"airline": "ANA", "price": 500, "currency": "JPY", "departure_airport": "SHA", "arrival_airport": "NRT", "departure_time": "2023-10-01 08:00", "arrival_time": "2023-10-01 12:00"}],
            "transport_notes": [],
            "booking_links": [],
        },
        [],
    )

    codes = {item.code for item in findings}
    assert {"missing_live_evidence", "wrong_departure_airport", "wrong_arrival_airport", "wrong_currency", "wrong_departure_date"} <= codes


def test_accommodation_gate_rejects_agent_hotel_date_conflicts():
    findings = gate_accommodation_result(
        _hotel_input(),
        {
            "bureau": "ACCOMMODATION",
            "status": "ok",
            "data_source": "live",
            "destination": "Tokyo",
            "hotel_options": [{"name": "Hotel", "currency": "USD", "notes": "check_in_date=2023-10-01 check_out_date=2023-10-02"}],
            "booking_links": [],
            "search_notes": [],
            "warnings": [],
        },
        [LiubuToolEvidence(tool_name="google_hotels", status="blocked", args={"check_in_date": "2023-10-01"}, error="check_in_date must match 2026-10-01")],
    )

    codes = {item.code for item in findings}
    assert "blocked_tool_call" in codes
    assert "hotel_date_conflict" in codes


def test_migrated_result_models_preserve_liubu_metadata():
    from utils.schemas import AccommodationExecutionResult, FlightTransportExecutionResult

    flight = FlightTransportExecutionResult.model_validate(
        {
            "bureau": "FLIGHT_TRANSPORT",
            "status": "fallback",
            "data_source": "fallback_estimate",
            "origin": "Beijing",
            "destination": "Tokyo",
            "flight_options": [],
            "transport_notes": [],
            "booking_links": [],
            "liubu_evidence": [{"tool_name": "google_flights", "status": "blocked"}],
            "liubu_quality": {"passed": False, "findings": [{"code": "wrong_departure_date"}]},
        }
    ).model_dump(mode="json")
    accommodation = AccommodationExecutionResult.model_validate(
        {
            "bureau": "ACCOMMODATION",
            "status": "fallback",
            "data_source": "fallback_estimate",
            "destination": "Tokyo",
            "hotel_options": [],
            "booking_links": [],
            "search_notes": [],
            "warnings": [],
            "liubu_evidence": [{"tool_name": "google_hotels", "status": "blocked"}],
            "liubu_quality": {"passed": False, "findings": [{"code": "hotel_date_conflict"}]},
        }
    ).model_dump(mode="json")

    assert flight["liubu_quality"]["passed"] is False
    assert flight["liubu_evidence"][0]["tool_name"] == "google_flights"
    assert accommodation["liubu_quality"]["findings"][0]["code"] == "hotel_date_conflict"
    assert accommodation["liubu_evidence"][0]["tool_name"] == "google_hotels"
