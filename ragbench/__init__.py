"""ragbench - A CLI tool for evaluating RAG retrieval quality."""

__version__ = "0.1.0"

from ragbench.metrics import (
    compute_all_metrics,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ragbench.models import (
    AggregateMetrics,
    Dataset,
    EvalReport,
    Query,
    QueryMetrics,
    RetrievalResult,
)

__all__ = [
    "__version__",
    "compute_all_metrics",
    "hit_rate_at_k",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "AggregateMetrics",
    "Dataset",
    "EvalReport",
    "Query",
    "QueryMetrics",
    "RetrievalResult",
]
