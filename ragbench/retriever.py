"""Retriever interface and implementations.

Provides an abstract base class for retrievers, a mock retriever that uses
pre-computed results from the dataset, and an HTTP retriever that calls
an external endpoint.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional

import httpx

from ragbench.models import Query, RetrievalResult


class RetrieverError(Exception):
    """Raised when a retriever encounters an error."""


class BaseRetriever(abc.ABC):
    """Abstract base class for retrievers."""

    @abc.abstractmethod
    def retrieve(self, query: Query, top_k: int) -> RetrievalResult:
        """Retrieve documents for a given query.

        Args:
            query: The query to retrieve documents for.
            top_k: Maximum number of documents to return.

        Returns:
            A RetrievalResult containing ranked document IDs.
        """

    def retrieve_batch(
        self, queries: List[Query], top_k: int
    ) -> List[RetrievalResult]:
        """Retrieve documents for multiple queries.

        Default implementation calls retrieve() sequentially.

        Args:
            queries: List of queries to retrieve documents for.
            top_k: Maximum number of documents to return per query.

        Returns:
            List of RetrievalResult instances.
        """
        return [self.retrieve(q, top_k) for q in queries]


class DatasetRetriever(BaseRetriever):
    """Retriever that uses pre-computed retrieved_ids from the dataset.

    This is the default retriever used when the dataset already contains
    retrieved results (the ``retrieved_ids`` field on each query).
    """

    def retrieve(self, query: Query, top_k: int) -> RetrievalResult:
        """Return the pre-computed retrieved_ids truncated to top_k."""
        return RetrievalResult(
            query_id=query.id,
            retrieved_ids=query.retrieved_ids[:top_k],
        )


class MockRetriever(BaseRetriever):
    """Retriever with configurable results, useful for testing.

    Args:
        results: Mapping of query_id -> list of retrieved document IDs.
    """

    def __init__(self, results: Dict[str, List[str]]) -> None:
        self._results = results

    def retrieve(self, query: Query, top_k: int) -> RetrievalResult:
        """Return mock results for the given query."""
        doc_ids = self._results.get(query.id, [])
        return RetrievalResult(
            query_id=query.id,
            retrieved_ids=doc_ids[:top_k],
        )


class HttpRetriever(BaseRetriever):
    """Retriever that calls an external HTTP endpoint.

    The endpoint should accept POST requests with a JSON body:

    .. code-block:: json

        {
            "query": "query text",
            "top_k": 10
        }

    And return a JSON response:

    .. code-block:: json

        {
            "ids": ["doc-1", "doc-2", ...],
            "scores": [0.95, 0.87, ...]
        }

    Args:
        url: The endpoint URL.
        headers: Optional HTTP headers (e.g., for authentication).
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
    ) -> None:
        self._url = url
        self._headers = headers or {}
        self._timeout = timeout

    def retrieve(self, query: Query, top_k: int) -> RetrievalResult:
        """Call the HTTP endpoint and return the results."""
        payload: Dict[str, Any] = {
            "query": query.text,
            "top_k": top_k,
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    self._url,
                    json=payload,
                    headers=self._headers,
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RetrieverError(
                f"HTTP request failed for query '{query.id}': {exc}"
            ) from exc

        try:
            data = response.json()
        except Exception as exc:
            raise RetrieverError(
                f"Failed to parse response JSON for query '{query.id}': {exc}"
            ) from exc

        ids = data.get("ids", [])
        if not isinstance(ids, list):
            raise RetrieverError(
                f"Expected 'ids' to be a list for query '{query.id}', got {type(ids).__name__}"
            )
        scores = data.get("scores")

        return RetrievalResult(
            query_id=query.id,
            retrieved_ids=ids[:top_k],
            scores=scores[:top_k] if scores else None,
        )
