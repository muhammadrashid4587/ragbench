"""Pure, stateless metric functions for evaluating retrieval quality.

All functions operate on simple Python types (lists, sets) and have no
side effects. They implement standard Information Retrieval metrics with
binary relevance.
"""

from __future__ import annotations

import math
from typing import Dict, List, Set, Union


def reciprocal_rank(relevant: Set[str], retrieved: List[str]) -> float:
    """Compute the Reciprocal Rank (RR).

    RR = 1 / rank_of_first_relevant_result

    If no relevant document appears in *retrieved*, returns 0.0.

    Args:
        relevant: Set of relevant document IDs.
        retrieved: Ranked list of retrieved document IDs (best first).

    Returns:
        Reciprocal rank as a float in [0, 1].
    """
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def mrr(relevant_per_query: List[Set[str]], retrieved_per_query: List[List[str]]) -> float:
    """Compute Mean Reciprocal Rank (MRR) across multiple queries.

    MRR = (1/|Q|) * sum_{q in Q} RR(q)

    Args:
        relevant_per_query: List of sets of relevant doc IDs, one per query.
        retrieved_per_query: List of ranked retrieved doc ID lists, one per query.

    Returns:
        MRR as a float in [0, 1].
    """
    if not relevant_per_query:
        return 0.0
    if len(relevant_per_query) != len(retrieved_per_query):
        raise ValueError(
            f"Length mismatch: {len(relevant_per_query)} relevant sets "
            f"vs {len(retrieved_per_query)} retrieved lists"
        )
    total = sum(
        reciprocal_rank(rel, ret)
        for rel, ret in zip(relevant_per_query, retrieved_per_query)
    )
    return total / len(relevant_per_query)


def ndcg_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank k (NDCG@k).

    Uses binary relevance: rel(doc) = 1 if doc in relevant else 0.

    DCG@k  = sum_{i=1}^{k} rel(i) / log2(i + 1)
    IDCG@k = sum_{i=1}^{min(k, |relevant|)} 1 / log2(i + 1)
    NDCG@k = DCG@k / IDCG@k

    If IDCG@k is zero (no relevant documents), returns 0.0.

    Args:
        relevant: Set of relevant document IDs.
        retrieved: Ranked list of retrieved document IDs.
        k: Cutoff rank.

    Returns:
        NDCG@k as a float in [0, 1].
    """
    if k <= 0:
        return 0.0

    # DCG@k
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 1)

    # IDCG@k: best possible DCG with min(k, |relevant|) relevant docs at the top
    ideal_hits = min(k, len(relevant))
    if ideal_hits == 0:
        return 0.0

    idcg = 0.0
    for i in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(i + 1)

    return dcg / idcg


def recall_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    """Compute Recall at rank k.

    Recall@k = |relevant ∩ retrieved@k| / |relevant|

    Args:
        relevant: Set of relevant document IDs.
        retrieved: Ranked list of retrieved document IDs.
        k: Cutoff rank.

    Returns:
        Recall@k as a float in [0, 1]. Returns 0.0 if relevant is empty.
    """
    if not relevant or k <= 0:
        return 0.0
    retrieved_at_k = set(retrieved[:k])
    return len(relevant & retrieved_at_k) / len(relevant)


def precision_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    """Compute Precision at rank k.

    Precision@k = |relevant ∩ retrieved@k| / k

    Args:
        relevant: Set of relevant document IDs.
        retrieved: Ranked list of retrieved document IDs.
        k: Cutoff rank.

    Returns:
        Precision@k as a float in [0, 1].
    """
    if k <= 0:
        return 0.0
    retrieved_at_k = set(retrieved[:k])
    return len(relevant & retrieved_at_k) / k


def hit_rate_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    """Compute Hit Rate at rank k.

    Hit Rate@k = 1 if any relevant doc appears in top-k, else 0.

    Args:
        relevant: Set of relevant document IDs.
        retrieved: Ranked list of retrieved document IDs.
        k: Cutoff rank.

    Returns:
        1.0 or 0.0.
    """
    if k <= 0:
        return 0.0
    retrieved_at_k = set(retrieved[:k])
    return 1.0 if relevant & retrieved_at_k else 0.0


def compute_all_metrics(
    relevant: Set[str],
    retrieved: List[str],
    k_values: List[int],
) -> Dict[str, Union[float, Dict[int, float]]]:
    """Compute all metrics for a single query at multiple k values.

    Args:
        relevant: Set of relevant document IDs.
        retrieved: Ranked list of retrieved document IDs.
        k_values: List of cutoff ranks to evaluate.

    Returns:
        Dictionary with keys: mrr, ndcg, recall, precision, hit_rate.
        mrr is a float; the rest are dicts mapping k -> metric value.
    """
    result: Dict[str, Union[float, Dict[int, float]]] = {
        "mrr": reciprocal_rank(relevant, retrieved),
        "ndcg": {},
        "recall": {},
        "precision": {},
        "hit_rate": {},
    }
    for k in k_values:
        result["ndcg"][k] = ndcg_at_k(relevant, retrieved, k)  # type: ignore[index]
        result["recall"][k] = recall_at_k(relevant, retrieved, k)  # type: ignore[index]
        result["precision"][k] = precision_at_k(relevant, retrieved, k)  # type: ignore[index]
        result["hit_rate"][k] = hit_rate_at_k(relevant, retrieved, k)  # type: ignore[index]
    return result
