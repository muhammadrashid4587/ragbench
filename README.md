# ragbench

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Pydantic v2](https://img.shields.io/badge/pydantic-v2-e92063.svg)](https://docs.pydantic.dev/)

**A CLI tool for evaluating RAG (Retrieval-Augmented Generation) pipeline retrieval quality.**

Point it at a ground-truth dataset, and ragbench computes standard Information Retrieval metrics so you can measure how well your retriever is working -- before you ever send tokens to an LLM.

---

## Why ragbench?

Building a RAG pipeline? The retrieval step is the foundation. If your retriever misses relevant documents, no amount of prompt engineering will save you. **ragbench** gives you a fast, reproducible way to measure retrieval quality with the same metrics used in academic IR research.

- **Fast**: Pure Python metric computations, no heavy dependencies
- **Correct**: Mathematically verified implementations with thorough test coverage
- **Flexible**: Use pre-computed results or plug in a live HTTP retriever
- **Beautiful**: Rich console tables for quick analysis, plus JSON/CSV export

## Installation

```bash
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
# Evaluate with pre-computed retrieval results
ragbench eval datasets/example.json

# Custom k values
ragbench eval datasets/example.json --k 1 --k 3 --k 5 --k 10

# JSON output
ragbench eval datasets/example.json --output json

# CSV output saved to file
ragbench eval datasets/example.json --output csv --save results.csv

# Use a live HTTP retriever
ragbench eval datasets/example.json --retriever-url http://localhost:8000/search
```

## Dataset Format

Create a JSON file with your ground-truth queries and relevant document IDs:

```json
{
  "name": "my-evaluation-set",
  "queries": [
    {
      "id": "q1",
      "text": "How to configure SSL certificates",
      "relevant_ids": ["doc-42", "doc-87"],
      "retrieved_ids": ["doc-87", "doc-15", "doc-42", "doc-99", "doc-03"]
    }
  ]
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Name for the dataset |
| `queries[].id` | Yes | Unique query identifier |
| `queries[].text` | Yes | The query text |
| `queries[].relevant_ids` | Yes | Ground-truth relevant document IDs |
| `queries[].retrieved_ids` | No | Pre-computed retrieval results (ranked) |

If `retrieved_ids` is omitted, use `--retriever-url` to point at a live endpoint.

## Metrics

ragbench computes five standard IR metrics at configurable cutoff values (k):

### Mean Reciprocal Rank (MRR)

The average of the reciprocal of the rank of the first relevant document across all queries.

$$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q}$$

where `rank_q` is the position of the first relevant document for query `q`.

### Normalized Discounted Cumulative Gain (NDCG@k)

Measures ranking quality by penalizing relevant documents that appear lower in the results, normalized against the ideal ranking. Uses binary relevance.

$$\text{DCG@k} = \sum_{i=1}^{k} \frac{\text{rel}(i)}{\log_2(i + 1)}$$

$$\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}$$

where `IDCG@k` is the DCG of the ideal ranking (all relevant documents at the top).

### Recall@k

The fraction of relevant documents that appear in the top-k results.

$$\text{Recall@k} = \frac{|\text{relevant} \cap \text{retrieved@k}|}{|\text{relevant}|}$$

### Precision@k

The fraction of top-k results that are relevant.

$$\text{Precision@k} = \frac{|\text{relevant} \cap \text{retrieved@k}|}{k}$$

### Hit Rate@k

Binary indicator: 1 if any relevant document appears in the top-k results, 0 otherwise.

$$\text{HitRate@k} = \begin{cases} 1 & \text{if } |\text{relevant} \cap \text{retrieved@k}| > 0 \\ 0 & \text{otherwise} \end{cases}$$

## HTTP Retriever

You can evaluate a live retriever endpoint. The endpoint should accept POST requests:

**Request:**
```json
{
  "query": "How to configure SSL certificates",
  "top_k": 10
}
```

**Response:**
```json
{
  "ids": ["doc-42", "doc-87", "doc-15"],
  "scores": [0.95, 0.87, 0.72]
}
```

Usage:
```bash
ragbench eval dataset.json \
  --retriever-url http://localhost:8000/search \
  --retriever-header Authorization "Bearer token123"
```

## Output Formats

### Table (default)

Rich, colorized console tables showing aggregate and per-query metrics.

```
        Aggregate Metrics - tech-docs-v1
┏━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Metric    ┃    @5  ┃   @10  ┃   @20  ┃
┡━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ MRR       │ 0.8542 │ 0.8542 │ 0.8542 │
│ NDCG      │ 0.9065 │ 0.9065 │ 0.9065 │
│ Recall    │ 0.8750 │ 0.8750 │ 0.8750 │
│ Precision │ 0.3250 │ 0.3250 │ 0.3250 │
│ Hit Rate  │ 1.0000 │ 1.0000 │ 1.0000 │
└───────────┴────────┴────────┴────────┘
```

### JSON

```bash
ragbench eval dataset.json --output json --save report.json
```

### CSV

```bash
ragbench eval dataset.json --output csv --save report.csv
```

## Python API

You can also use ragbench as a library:

```python
from ragbench.metrics import ndcg_at_k, recall_at_k, precision_at_k, reciprocal_rank, hit_rate_at_k

relevant = {"doc-42", "doc-87"}
retrieved = ["doc-87", "doc-15", "doc-42", "doc-99", "doc-03"]

print(reciprocal_rank(relevant, retrieved))       # 1.0
print(ndcg_at_k(relevant, retrieved, k=5))        # 0.9197...
print(recall_at_k(relevant, retrieved, k=3))      # 1.0
print(precision_at_k(relevant, retrieved, k=3))   # 0.6667
print(hit_rate_at_k(relevant, retrieved, k=1))    # 1.0
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

With coverage:

```bash
pytest --cov=ragbench --cov-report=term-missing
```

## Project Structure

```
ragbench/
├── ragbench/
│   ├── __init__.py        # Version
│   ├── cli.py             # Click CLI entry point
│   ├── metrics.py         # Pure metric functions (MRR, NDCG, Recall, Precision, Hit Rate)
│   ├── dataset.py         # Dataset loading and validation
│   ├── retriever.py       # Abstract retriever interface + implementations
│   ├── reporter.py        # Rich table, JSON, and CSV output
│   └── models.py          # Pydantic data models
├── datasets/
│   └── example.json       # Example ground-truth dataset
├── tests/
│   ├── test_metrics.py    # Comprehensive metric tests
│   ├── test_dataset.py    # Dataset loading/validation tests
│   └── test_retriever.py  # Retriever tests
├── pyproject.toml
├── LICENSE
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes and add tests
4. Run the test suite: `pytest`
5. Submit a pull request

## License

MIT License. See [LICENSE](LICENSE) for details.
