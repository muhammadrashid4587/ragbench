"""Thorough unit tests for all IR metrics."""

from __future__ import annotations

import math

import pytest

from ragbench.metrics import (
    compute_all_metrics,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


# ---------------------------------------------------------------------------
# Reciprocal Rank
# ---------------------------------------------------------------------------
class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank({"a"}, ["a", "b", "c"]) == 1.0

    def test_second_position(self):
        assert reciprocal_rank({"a"}, ["b", "a", "c"]) == 0.5

    def test_third_position(self):
        assert reciprocal_rank({"a"}, ["b", "c", "a"]) == pytest.approx(1 / 3)

    def test_not_found(self):
        assert reciprocal_rank({"a"}, ["b", "c", "d"]) == 0.0

    def test_empty_retrieved(self):
        assert reciprocal_rank({"a"}, []) == 0.0

    def test_multiple_relevant_returns_first(self):
        # Two relevant docs; RR should use the first one found
        assert reciprocal_rank({"a", "c"}, ["b", "a", "c"]) == 0.5

    def test_relevant_at_first_when_multiple(self):
        assert reciprocal_rank({"a", "b"}, ["a", "b", "c"]) == 1.0


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------
class TestMRR:
    def test_basic(self):
        relevant = [{"a"}, {"b"}]
        retrieved = [["a", "b", "c"], ["c", "b", "a"]]
        # RR(q1) = 1.0, RR(q2) = 0.5 => MRR = 0.75
        assert mrr(relevant, retrieved) == pytest.approx(0.75)

    def test_no_hits(self):
        relevant = [{"a"}, {"b"}]
        retrieved = [["c", "d"], ["c", "d"]]
        assert mrr(relevant, retrieved) == 0.0

    def test_empty(self):
        assert mrr([], []) == 0.0

    def test_single_query(self):
        assert mrr([{"a"}], [["b", "a"]]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# NDCG@k
# ---------------------------------------------------------------------------
class TestNDCG:
    def test_perfect_ranking(self):
        # Two relevant docs in positions 1 and 2 with k=2
        relevant = {"a", "b"}
        retrieved = ["a", "b", "c"]
        assert ndcg_at_k(relevant, retrieved, k=2) == pytest.approx(1.0)

    def test_reversed_ranking(self):
        # Two relevant docs, both in top-2 but in reversed ideal order
        # With binary relevance, order within relevant docs doesn't matter
        # DCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309...
        # IDCG = same since both are relevant
        relevant = {"a", "b"}
        retrieved = ["b", "a", "c"]
        assert ndcg_at_k(relevant, retrieved, k=2) == pytest.approx(1.0)

    def test_one_relevant_at_position_2(self):
        relevant = {"a"}
        retrieved = ["b", "a", "c"]
        # DCG = 0/log2(2) + 1/log2(3) = 1/log2(3)
        # IDCG = 1/log2(2) = 1.0
        expected = (1.0 / math.log2(3)) / 1.0
        assert ndcg_at_k(relevant, retrieved, k=2) == pytest.approx(expected)

    def test_no_relevant_in_topk(self):
        relevant = {"a"}
        retrieved = ["b", "c", "a"]
        assert ndcg_at_k(relevant, retrieved, k=2) == pytest.approx(0.0)

    def test_k_larger_than_retrieved(self):
        relevant = {"a", "b"}
        retrieved = ["a"]
        # DCG = 1/log2(2) = 1.0
        # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        expected = 1.0 / idcg
        assert ndcg_at_k(relevant, retrieved, k=5) == pytest.approx(expected)

    def test_k_zero(self):
        assert ndcg_at_k({"a"}, ["a", "b"], k=0) == 0.0

    def test_empty_relevant(self):
        assert ndcg_at_k(set(), ["a", "b"], k=5) == 0.0

    def test_k_equals_one_hit(self):
        assert ndcg_at_k({"a"}, ["a"], k=1) == pytest.approx(1.0)

    def test_k_equals_one_miss(self):
        assert ndcg_at_k({"a"}, ["b"], k=1) == pytest.approx(0.0)

    def test_three_relevant_partial_retrieval(self):
        relevant = {"a", "b", "c"}
        retrieved = ["x", "a", "y", "b", "z"]
        k = 5
        # DCG = 0 + 1/log2(3) + 0 + 1/log2(5) + 0
        dcg = 1.0 / math.log2(3) + 1.0 / math.log2(5)
        # IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4) (3 relevant, k=5)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3) + 1.0 / math.log2(4)
        assert ndcg_at_k(relevant, retrieved, k=k) == pytest.approx(dcg / idcg)


# ---------------------------------------------------------------------------
# Recall@k
# ---------------------------------------------------------------------------
class TestRecall:
    def test_full_recall(self):
        assert recall_at_k({"a", "b"}, ["a", "b", "c"], k=3) == pytest.approx(1.0)

    def test_partial_recall(self):
        assert recall_at_k({"a", "b"}, ["a", "c", "d"], k=3) == pytest.approx(0.5)

    def test_zero_recall(self):
        assert recall_at_k({"a", "b"}, ["c", "d", "e"], k=3) == pytest.approx(0.0)

    def test_k_limits_retrieval(self):
        # "b" is at position 3 but k=2
        assert recall_at_k({"a", "b"}, ["a", "c", "b"], k=2) == pytest.approx(0.5)

    def test_empty_relevant(self):
        assert recall_at_k(set(), ["a", "b"], k=5) == 0.0

    def test_k_zero(self):
        assert recall_at_k({"a"}, ["a"], k=0) == 0.0

    def test_k_larger_than_retrieved(self):
        assert recall_at_k({"a", "b"}, ["a"], k=10) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Precision@k
# ---------------------------------------------------------------------------
class TestPrecision:
    def test_all_relevant(self):
        assert precision_at_k({"a", "b", "c"}, ["a", "b", "c"], k=3) == pytest.approx(1.0)

    def test_some_relevant(self):
        assert precision_at_k({"a"}, ["a", "b", "c"], k=3) == pytest.approx(1 / 3)

    def test_none_relevant(self):
        assert precision_at_k({"a"}, ["b", "c", "d"], k=3) == pytest.approx(0.0)

    def test_k_one_hit(self):
        assert precision_at_k({"a"}, ["a", "b"], k=1) == pytest.approx(1.0)

    def test_k_one_miss(self):
        assert precision_at_k({"a"}, ["b", "a"], k=1) == pytest.approx(0.0)

    def test_k_zero(self):
        assert precision_at_k({"a"}, ["a"], k=0) == 0.0

    def test_two_of_five(self):
        assert precision_at_k({"a", "c"}, ["a", "b", "c", "d", "e"], k=5) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Hit Rate@k
# ---------------------------------------------------------------------------
class TestHitRate:
    def test_hit(self):
        assert hit_rate_at_k({"a"}, ["b", "a", "c"], k=3) == 1.0

    def test_miss(self):
        assert hit_rate_at_k({"a"}, ["b", "c", "d"], k=3) == 0.0

    def test_hit_at_boundary(self):
        assert hit_rate_at_k({"a"}, ["b", "c", "a"], k=3) == 1.0

    def test_miss_beyond_k(self):
        assert hit_rate_at_k({"a"}, ["b", "c", "a"], k=2) == 0.0

    def test_k_zero(self):
        assert hit_rate_at_k({"a"}, ["a"], k=0) == 0.0

    def test_empty_retrieved(self):
        assert hit_rate_at_k({"a"}, [], k=5) == 0.0


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------
class TestComputeAllMetrics:
    def test_returns_all_keys(self):
        result = compute_all_metrics({"a"}, ["a", "b"], [5, 10])
        assert "mrr" in result
        assert "ndcg" in result
        assert "recall" in result
        assert "precision" in result
        assert "hit_rate" in result

    def test_correct_k_values(self):
        result = compute_all_metrics({"a"}, ["a", "b"], [3, 7])
        assert set(result["ndcg"].keys()) == {3, 7}  # type: ignore[union-attr]
        assert set(result["recall"].keys()) == {3, 7}  # type: ignore[union-attr]

    def test_perfect_retrieval(self):
        result = compute_all_metrics({"a"}, ["a", "b", "c"], [1, 3])
        assert result["mrr"] == 1.0
        assert result["ndcg"][1] == pytest.approx(1.0)  # type: ignore[index]
        assert result["recall"][1] == pytest.approx(1.0)  # type: ignore[index]
        assert result["precision"][1] == pytest.approx(1.0)  # type: ignore[index]
        assert result["hit_rate"][1] == 1.0  # type: ignore[index]

    def test_no_retrieval(self):
        result = compute_all_metrics({"a"}, ["b", "c", "d"], [1, 3])
        assert result["mrr"] == 0.0
        assert result["recall"][3] == pytest.approx(0.0)  # type: ignore[index]
        assert result["hit_rate"][3] == 0.0  # type: ignore[index]

    def test_hand_computed_example(self):
        """Full hand-computed example from the project spec."""
        relevant = {"doc-42", "doc-87"}
        retrieved = ["doc-87", "doc-15", "doc-42", "doc-99", "doc-03"]

        result = compute_all_metrics(relevant, retrieved, [3, 5])

        # MRR: first relevant at position 1 => RR = 1.0
        assert result["mrr"] == pytest.approx(1.0)

        # Recall@3: {doc-87, doc-42} ∩ {doc-87, doc-15, doc-42} = 2 => 2/2 = 1.0
        assert result["recall"][3] == pytest.approx(1.0)  # type: ignore[index]

        # Precision@3: 2 relevant in top 3 => 2/3
        assert result["precision"][3] == pytest.approx(2 / 3)  # type: ignore[index]

        # Hit Rate@3: yes, doc-87 is in top 3
        assert result["hit_rate"][3] == 1.0  # type: ignore[index]

        # NDCG@5:
        # DCG = 1/log2(2) + 0 + 1/log2(4) + 0 + 0 = 1.0 + 0.5 = 1.5
        # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309 = 1.6309
        dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        assert result["ndcg"][5] == pytest.approx(dcg / idcg)  # type: ignore[index]
