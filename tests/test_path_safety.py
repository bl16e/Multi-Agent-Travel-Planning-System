import pytest

from utils.path_safety import sanitize_request_id


def test_sanitize_request_id_replaces_unsafe_characters():
    assert sanitize_request_id("../trip id:tokyo") == ".._trip_id_tokyo"


def test_sanitize_request_id_rejects_empty_values():
    with pytest.raises(ValueError):
        sanitize_request_id("")


def test_sanitize_request_id_truncates_long_values():
    assert sanitize_request_id("a" * 101) == "a" * 100
