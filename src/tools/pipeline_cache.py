"""
Pipeline Cache module.

This module will cache evaluated pipeline configurations and their scores.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class PipelineCache:
    """Simple in-memory cache for evaluated pipeline results."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
        self._hits = 0
        self._misses = 0

    def make_key(self, dataset_name: str, actions: list[str]) -> tuple[str, tuple[str, ...]]:
        """Build a stable cache key from dataset name and effective actions."""
        return dataset_name, tuple(actions)

    def has(self, dataset_name: str, actions: list[str]) -> bool:
        """Return whether a cached result exists and update hit/miss counters."""
        key = self.make_key(dataset_name, actions)
        if key in self._cache:
            self._hits += 1
            return True
        self._misses += 1
        return False

    def get(self, dataset_name: str, actions: list[str]) -> dict[str, Any]:
        """Return a copy of a cached result."""
        key = self.make_key(dataset_name, actions)
        return deepcopy(self._cache[key])

    def set(self, dataset_name: str, actions: list[str], result: dict[str, Any]) -> None:
        """Store a copy of an evaluation result."""
        key = self.make_key(dataset_name, actions)
        self._cache[key] = deepcopy(result)

    def clear(self) -> None:
        """Remove all cached results and reset counters."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict[str, float | int]:
        """Return cache usage statistics."""
        total = self._hits + self._misses
        hit_rate = float(self._hits) / float(total) if total else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "hit_rate": hit_rate,
        }
