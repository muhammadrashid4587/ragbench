"""Load and validate ground-truth JSON datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from ragbench.models import Dataset


class DatasetError(Exception):
    """Raised when a dataset cannot be loaded or is invalid."""


def load_dataset(path: Union[str, Path]) -> Dataset:
    """Load a ground-truth dataset from a JSON file.

    The expected JSON format:

    .. code-block:: json

        {
            "name": "my-dataset",
            "queries": [
                {
                    "id": "q1",
                    "text": "query text",
                    "relevant_ids": ["doc-1", "doc-2"],
                    "retrieved_ids": ["doc-2", "doc-3", "doc-1"]
                }
            ]
        }

    Args:
        path: Path to the JSON file.

    Returns:
        Validated Dataset instance.

    Raises:
        DatasetError: If the file cannot be read or the content is invalid.
    """
    path = Path(path)

    if not path.exists():
        raise DatasetError(f"Dataset file not found: {path}")

    if not path.suffix.lower() == ".json":
        raise DatasetError(f"Dataset file must be a .json file, got: {path.suffix}")

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DatasetError(f"Failed to read dataset file: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"Invalid JSON in dataset file: {exc}") from exc

    if not isinstance(data, dict):
        raise DatasetError("Dataset JSON must be an object at the top level")

    try:
        dataset = Dataset(**data)
    except Exception as exc:
        raise DatasetError(f"Dataset validation failed: {exc}") from exc

    return dataset


def validate_dataset(dataset: Dataset) -> list[str]:
    """Run additional validation checks on a dataset.

    Returns a list of warning messages (empty if everything looks good).
    """
    warnings: list[str] = []

    query_ids = [q.id for q in dataset.queries]
    if len(query_ids) != len(set(query_ids)):
        warnings.append("Duplicate query IDs detected")

    for query in dataset.queries:
        if not query.retrieved_ids:
            warnings.append(
                f"Query '{query.id}' has no retrieved_ids; "
                "metrics will be zero unless a retriever is used"
            )
        if len(query.relevant_ids) != len(set(query.relevant_ids)):
            warnings.append(f"Query '{query.id}' has duplicate relevant_ids")
        if len(query.retrieved_ids) != len(set(query.retrieved_ids)):
            warnings.append(f"Query '{query.id}' has duplicate retrieved_ids")

    return warnings
