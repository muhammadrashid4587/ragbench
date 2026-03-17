"""Pydantic models for ragbench data structures."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class Query(BaseModel):
    """A single evaluation query with ground-truth relevant documents."""

    id: str = Field(..., description="Unique identifier for the query")
    text: str = Field(..., description="The query text")
    relevant_ids: List[str] = Field(
        ..., description="Document IDs that are relevant to this query"
    )
    retrieved_ids: List[str] = Field(
        default_factory=list,
        description="Document IDs returned by the retriever (ranked)",
    )

    @field_validator("relevant_ids")
    @classmethod
    def relevant_ids_not_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("relevant_ids must contain at least one document ID")
        return v


class Dataset(BaseModel):
    """A ground-truth evaluation dataset."""

    name: str = Field(..., description="Name of the dataset")
    queries: List[Query] = Field(..., description="List of evaluation queries")

    @field_validator("queries")
    @classmethod
    def queries_not_empty(cls, v: List[Query]) -> List[Query]:
        if not v:
            raise ValueError("Dataset must contain at least one query")
        return v


class QueryMetrics(BaseModel):
    """Computed metrics for a single query."""

    query_id: str
    query_text: str
    mrr: float = Field(..., ge=0.0, le=1.0)
    ndcg: Dict[int, float] = Field(default_factory=dict)
    recall: Dict[int, float] = Field(default_factory=dict)
    precision: Dict[int, float] = Field(default_factory=dict)
    hit_rate: Dict[int, float] = Field(default_factory=dict)


class AggregateMetrics(BaseModel):
    """Aggregate metrics across all queries."""

    num_queries: int
    mean_mrr: float
    mean_ndcg: Dict[int, float] = Field(default_factory=dict)
    mean_recall: Dict[int, float] = Field(default_factory=dict)
    mean_precision: Dict[int, float] = Field(default_factory=dict)
    mean_hit_rate: Dict[int, float] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Result from a retriever for a single query."""

    query_id: str
    retrieved_ids: List[str] = Field(
        ..., description="Ranked list of retrieved document IDs"
    )
    scores: Optional[List[float]] = Field(
        None, description="Optional relevance scores for each retrieved document"
    )


class EvalReport(BaseModel):
    """Full evaluation report."""

    dataset_name: str
    k_values: List[int]
    per_query: List[QueryMetrics]
    aggregate: AggregateMetrics
