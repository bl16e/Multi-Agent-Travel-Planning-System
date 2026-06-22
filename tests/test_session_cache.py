from __future__ import annotations

from utils.session_cache import InMemorySessionCache


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_cache_returns_inserted_entry_before_ttl_expires():
    clock = FakeClock()
    cache = InMemorySessionCache(max_entries=2, ttl_seconds=10, now=clock)

    cache.set("trip_a", {"status": "RUNNING"})

    assert cache.get("trip_a") == {"status": "RUNNING"}
    assert len(cache) == 1


def test_cache_expires_entries_after_ttl():
    clock = FakeClock()
    cache = InMemorySessionCache(max_entries=2, ttl_seconds=10, now=clock)
    cache.set("trip_a", {"status": "RUNNING"})

    clock.advance(11)

    assert cache.get("trip_a") is None
    assert len(cache) == 0


def test_cache_evicts_oldest_entry_when_capacity_is_exceeded():
    clock = FakeClock()
    cache = InMemorySessionCache(max_entries=2, ttl_seconds=100, now=clock)
    cache.set("trip_a", {"status": "A"})
    clock.advance(1)
    cache.set("trip_b", {"status": "B"})
    clock.advance(1)

    cache.set("trip_c", {"status": "C"})

    assert cache.get("trip_a") is None
    assert cache.get("trip_b") == {"status": "B"}
    assert cache.get("trip_c") == {"status": "C"}


def test_cache_delete_removes_entry():
    cache = InMemorySessionCache(max_entries=2, ttl_seconds=100)
    cache.set("trip_a", {"status": "RUNNING"})

    assert cache.delete("trip_a") is True
    assert cache.get("trip_a") is None
    assert cache.delete("trip_a") is False
