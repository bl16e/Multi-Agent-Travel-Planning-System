from provinces.liubu.constrained.state import normalize_worker_input


def _payload():
    return {
        "request_id": "liubu_state",
        "target_bureau": "FLIGHT_TRANSPORT",
        "approved_draft": {
            "destination": "Tokyo",
            "itinerary_draft": {
                "destination": "Tokyo",
                "daily_plan": [
                    {"day_index": 1, "date": "2026-10-01", "activities": []},
                    {"day_index": 2, "date": "2026-10-02", "activities": []},
                ],
            },
            "bureau_tasks": [
                {
                    "bureau": "FLIGHT_TRANSPORT",
                    "objective": "Find flights",
                    "inputs_required": ["origin_city", "destination"],
                    "deliverables": ["flight_options"],
                    "priority": "high",
                }
            ],
        },
        "review_payload": {"review_notes": ["Approved after review."]},
        "execution_plan": {
            "user_request": {
                "profile": {
                    "origin_city": "Beijing",
                    "origin_airport_code": "PEK",
                    "destination_airport_code": "HND",
                    "destination_preferences": ["Tokyo"],
                    "start_date": "2026-10-01",
                    "end_date": "2026-10-02",
                    "adults": 2,
                    "children": 1,
                    "currency": "USD",
                    "constraints": ["avoid red-eye flights"],
                    "interests": ["culture"],
                }
            }
        },
    }


def test_normalize_worker_input_preserves_trip_constraints():
    worker_input = normalize_worker_input(_payload(), "FLIGHT_TRANSPORT")

    assert worker_input.request_id == "liubu_state"
    assert worker_input.bureau == "FLIGHT_TRANSPORT"
    assert worker_input.destination == "Tokyo"
    assert worker_input.profile["origin_city"] == "Beijing"
    assert worker_input.constraints["origin_airport_code"] == "PEK"
    assert worker_input.constraints["destination_airport_code"] == "HND"
    assert worker_input.constraints["start_date"] == "2026-10-01"
    assert worker_input.constraints["end_date"] == "2026-10-02"
    assert worker_input.constraints["adults"] == 2
    assert worker_input.constraints["children"] == 1
    assert worker_input.constraints["currency"] == "USD"
    assert worker_input.trip_dates == ["2026-10-01", "2026-10-02"]
    assert worker_input.bureau_task["objective"] == "Find flights"
    assert worker_input.review_notes == ["Approved after review."]


def test_normalize_worker_input_uses_itinerary_dates_when_profile_dates_missing():
    payload = _payload()
    payload["execution_plan"]["user_request"]["profile"].pop("start_date")
    payload["execution_plan"]["user_request"]["profile"].pop("end_date")

    worker_input = normalize_worker_input(payload, "FLIGHT_TRANSPORT")

    assert worker_input.constraints["start_date"] == "2026-10-01"
    assert worker_input.constraints["end_date"] == "2026-10-02"
