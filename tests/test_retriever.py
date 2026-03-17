"""Tests for retriever implementations."""

from __future__ import annotations

import pytest

from ragbench.models import Query
from ragbench.retriever import DatasetRetriever, MockRetriever


def _make_query(
    qid: str = "q1",
    text: str = "test query",
    relevant_ids: list[str] | None = None,
    retrieved_ids: list[str] | None = None,
) -> Query:
    return Query(
        id=qid,
        text=text,
        relevant_ids=relevant_ids or ["d1"],
        retrieved_ids=retrieved_ids or ["d1", "d2", "d3", "d4", "d5"],
    )


# ---------------------------------------------------------------------------
# DatasetRetriever
# ---------------------------------------------------------------------------
class TestDatasetRetriever:
    def test_returns_precomputed_ids(self):
        q = _make_query(retrieved_ids=["a", "b", "c", "d", "e"])
        retriever = DatasetRetriever()
        result = retriever.retrieve(q, top_k=5)
        assert result.query_id == "q1"
        assert result.retrieved_ids == ["a", "b", "c", "d", "e"]

    def test_truncates_to_top_k(self):
        q = _make_query(retrieved_ids=["a", "b", "c", "d", "e"])
        retriever = DatasetRetriever()
        result = retriever.retrieve(q, top_k=3)
        assert result.retrieved_ids == ["a", "b", "c"]

    def test_top_k_larger_than_available(self):
        q = _make_query(retrieved_ids=["a", "b"])
        retriever = DatasetRetriever()
        result = retriever.retrieve(q, top_k=10)
        assert result.retrieved_ids == ["a", "b"]

    def test_batch_retrieve(self):
        q1 = _make_query(qid="q1", retrieved_ids=["a", "b"])
        q2 = _make_query(qid="q2", retrieved_ids=["c", "d"])
        retriever = DatasetRetriever()
        results = retriever.retrieve_batch([q1, q2], top_k=5)
        assert len(results) == 2
        assert results[0].query_id == "q1"
        assert results[1].query_id == "q2"


# ---------------------------------------------------------------------------
# MockRetriever
# ---------------------------------------------------------------------------
class TestMockRetriever:
    def test_returns_configured_results(self):
        retriever = MockRetriever(results={"q1": ["x", "y", "z"]})
        q = _make_query(qid="q1")
        result = retriever.retrieve(q, top_k=3)
        assert result.retrieved_ids == ["x", "y", "z"]

    def test_unknown_query_returns_empty(self):
        retriever = MockRetriever(results={})
        q = _make_query(qid="unknown")
        result = retriever.retrieve(q, top_k=5)
        assert result.retrieved_ids == []

    def test_truncates_to_top_k(self):
        retriever = MockRetriever(results={"q1": ["a", "b", "c", "d", "e"]})
        q = _make_query(qid="q1")
        result = retriever.retrieve(q, top_k=2)
        assert result.retrieved_ids == ["a", "b"]

    def test_batch_retrieve(self):
        retriever = MockRetriever(
            results={"q1": ["a"], "q2": ["b", "c"]}
        )
        q1 = _make_query(qid="q1")
        q2 = _make_query(qid="q2")
        results = retriever.retrieve_batch([q1, q2], top_k=5)
        assert len(results) == 2
        assert results[0].retrieved_ids == ["a"]
        assert results[1].retrieved_ids == ["b", "c"]

    def test_scores_are_none(self):
        retriever = MockRetriever(results={"q1": ["a"]})
        q = _make_query(qid="q1")
        result = retriever.retrieve(q, top_k=5)
        assert result.scores is None
