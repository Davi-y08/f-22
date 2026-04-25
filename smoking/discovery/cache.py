from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass
from typing import Generic, Hashable, TypeVar


T = TypeVar("T")


@dataclass(slots=True)
class _CacheEntry(Generic[T]):
    value: T
    created_at: float


class DiscoveryCache(Generic[T]):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[Hashable, _CacheEntry[T]] = {}

    def get(self, key: Hashable, ttl_seconds: float) -> T | None:
        if ttl_seconds <= 0:
            return None

        now = time.monotonic()
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                return None
            if (now - entry.created_at) > ttl_seconds:
                self._items.pop(key, None)
                return None
            return copy.deepcopy(entry.value)

    def set(self, key: Hashable, value: T) -> None:
        with self._lock:
            self._items[key] = _CacheEntry(value=copy.deepcopy(value), created_at=time.monotonic())

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
