"""Unit tests for cached local knowledge retrieval."""

from __future__ import annotations

from backend import knowledge


def test_search_knowledge_returns_relevant_cached_entries() -> None:
    """Verify repeated retrieval reuses the cached ranking for the same query."""
    knowledge._knowledge_search_index.cache_clear()
    knowledge._search_knowledge_cached.cache_clear()

    first = knowledge.search_knowledge("HER2 靶向治疗", limit=5)
    cache_after_first = knowledge._search_knowledge_cached.cache_info()
    second = knowledge.search_knowledge("HER2 靶向治疗", limit=5)
    cache_after_second = knowledge._search_knowledge_cached.cache_info()

    assert first
    assert [entry.id for entry in second] == [entry.id for entry in first]
    assert cache_after_first.misses == 1
    assert cache_after_second.hits == 1
    assert any("HER2" in entry.title or "HER2" in entry.summary for entry in first)
