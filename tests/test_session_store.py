import pytest

from utils.session_store import CorruptSessionError, JsonSessionStore, StoredSession


def test_json_session_store_round_trips_session(tmp_path):
    store = JsonSessionStore(tmp_path)
    session = StoredSession(
        request_id="trip/../tokyo",
        request={"request_id": "trip/../tokyo", "user_message": "Plan Tokyo", "profile": {}},
        status="HUMAN_INTERVENE",
        context_snapshot={"pending_user_inputs": ["Need budget"]},
        result={"status": "HUMAN_INTERVENE"},
    )

    store.save(session)

    loaded = store.load("trip/../tokyo")
    assert loaded == session
    assert (tmp_path / "trip_.._tokyo.json").exists()


def test_json_session_store_returns_none_for_missing_session(tmp_path):
    assert JsonSessionStore(tmp_path).load("missing") is None


def test_json_session_store_raises_clear_error_for_corrupt_session(tmp_path):
    store = JsonSessionStore(tmp_path)
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(CorruptSessionError):
        store.load("broken")
