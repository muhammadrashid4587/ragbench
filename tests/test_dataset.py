"""Tests for dataset loading and validation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from ragbench.dataset import DatasetError, load_dataset, validate_dataset
from ragbench.models import Dataset, Query


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------
class TestLoadDataset:
    def test_load_valid_dataset(self, tmp_path: Path):
        data = {
            "name": "test-ds",
            "queries": [
                {
                    "id": "q1",
                    "text": "query one",
                    "relevant_ids": ["d1"],
                    "retrieved_ids": ["d1", "d2"],
                }
            ],
        }
        fp = tmp_path / "ds.json"
        fp.write_text(json.dumps(data))

        ds = load_dataset(fp)
        assert ds.name == "test-ds"
        assert len(ds.queries) == 1
        assert ds.queries[0].id == "q1"

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(DatasetError, match="not found"):
            load_dataset(tmp_path / "nonexistent.json")

    def test_wrong_extension(self, tmp_path: Path):
        fp = tmp_path / "ds.txt"
        fp.write_text("{}")
        with pytest.raises(DatasetError, match=".json"):
            load_dataset(fp)

    def test_invalid_json(self, tmp_path: Path):
        fp = tmp_path / "ds.json"
        fp.write_text("{not valid json}")
        with pytest.raises(DatasetError, match="Invalid JSON"):
            load_dataset(fp)

    def test_not_an_object(self, tmp_path: Path):
        fp = tmp_path / "ds.json"
        fp.write_text("[1, 2, 3]")
        with pytest.raises(DatasetError, match="object"):
            load_dataset(fp)

    def test_missing_required_fields(self, tmp_path: Path):
        fp = tmp_path / "ds.json"
        fp.write_text(json.dumps({"name": "test"}))
        with pytest.raises(DatasetError, match="validation failed"):
            load_dataset(fp)

    def test_empty_queries(self, tmp_path: Path):
        fp = tmp_path / "ds.json"
        fp.write_text(json.dumps({"name": "test", "queries": []}))
        with pytest.raises(DatasetError, match="validation failed"):
            load_dataset(fp)

    def test_empty_relevant_ids(self, tmp_path: Path):
        data = {
            "name": "test",
            "queries": [
                {
                    "id": "q1",
                    "text": "query",
                    "relevant_ids": [],
                    "retrieved_ids": [],
                }
            ],
        }
        fp = tmp_path / "ds.json"
        fp.write_text(json.dumps(data))
        with pytest.raises(DatasetError, match="validation failed"):
            load_dataset(fp)

    def test_no_retrieved_ids_is_ok(self, tmp_path: Path):
        data = {
            "name": "test",
            "queries": [
                {
                    "id": "q1",
                    "text": "query",
                    "relevant_ids": ["d1"],
                }
            ],
        }
        fp = tmp_path / "ds.json"
        fp.write_text(json.dumps(data))
        ds = load_dataset(fp)
        assert ds.queries[0].retrieved_ids == []


# ---------------------------------------------------------------------------
# validate_dataset
# ---------------------------------------------------------------------------
class TestValidateDataset:
    def test_valid_dataset_no_warnings(self):
        ds = Dataset(
            name="ok",
            queries=[
                Query(
                    id="q1",
                    text="t",
                    relevant_ids=["d1"],
                    retrieved_ids=["d1", "d2"],
                )
            ],
        )
        assert validate_dataset(ds) == []

    def test_duplicate_query_ids(self):
        ds = Dataset(
            name="dup",
            queries=[
                Query(id="q1", text="t1", relevant_ids=["d1"], retrieved_ids=["d1"]),
                Query(id="q1", text="t2", relevant_ids=["d2"], retrieved_ids=["d2"]),
            ],
        )
        warnings = validate_dataset(ds)
        assert any("Duplicate query IDs" in w for w in warnings)

    def test_no_retrieved_ids_warning(self):
        ds = Dataset(
            name="no-ret",
            queries=[
                Query(id="q1", text="t", relevant_ids=["d1"]),
            ],
        )
        warnings = validate_dataset(ds)
        assert any("no retrieved_ids" in w for w in warnings)

    def test_duplicate_relevant_ids(self):
        ds = Dataset(
            name="dup-rel",
            queries=[
                Query(
                    id="q1",
                    text="t",
                    relevant_ids=["d1", "d1"],
                    retrieved_ids=["d1"],
                ),
            ],
        )
        warnings = validate_dataset(ds)
        assert any("duplicate relevant_ids" in w for w in warnings)

    def test_duplicate_retrieved_ids(self):
        ds = Dataset(
            name="dup-ret",
            queries=[
                Query(
                    id="q1",
                    text="t",
                    relevant_ids=["d1"],
                    retrieved_ids=["d1", "d1"],
                ),
            ],
        )
        warnings = validate_dataset(ds)
        assert any("duplicate retrieved_ids" in w for w in warnings)
