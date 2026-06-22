from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class _CacheEntry:
    value: dict[str, Any]
    expires_at: float


class InMemorySessionCache:
    def __init__(
        self,
        max_entries: int = 500,
        ttl_seconds: int = 86400,
        now: Callable[[], float] | None = None,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be at least 1")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._now = now or time.time
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()

    def set(self, request_id: str, value: dict[str, Any]) -> None:
        self.cleanup_expired()
        self._entries.pop(request_id, None)
        self._entries[request_id] = _CacheEntry(
            value=value,
            expires_at=self._now() + self.ttl_seconds,
        )
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def get(self, request_id: str) -> dict[str, Any] | None:
        entry = self._entries.get(request_id)
        if entry is None:
            return None
        if entry.expires_at <= self._now():
            self._entries.pop(request_id, None)
            return None
        self._entries.move_to_end(request_id)
        return entry.value

    def delete(self, request_id: str) -> bool:
        return self._entries.pop(request_id, None) is not None

    def cleanup_expired(self) -> None:
        now = self._now()
        expired = [key for key, entry in self._entries.items() if entry.expires_at <= now]
        for key in expired:
            self._entries.pop(key, None)

    def __len__(self) -> int:
        self.cleanup_expired()
        return len(self._entries)
