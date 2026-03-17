"""Report generation for evaluation results.

Supports Rich console tables, JSON, and CSV output formats.
"""

from __future__ import annotations

import csv
import io
import json
from typing import TextIO

from rich.console import Console
from rich.table import Table

from ragbench.models import EvalReport


def render_table(report: EvalReport, console: Console | None = None) -> None:
    """Render the evaluation report as a Rich console table.

    Args:
        report: The evaluation report to render.
        console: Optional Rich Console instance. Creates a new one if None.
    """
    if console is None:
        console = Console()

    k_values = sorted(report.k_values)

    # --- Aggregate summary table ---
    agg_table = Table(
        title=f"Aggregate Metrics - {report.dataset_name}",
        show_header=True,
        header_style="bold cyan",
    )
    agg_table.add_column("Metric", style="bold")
    for k in k_values:
        agg_table.add_column(f"@{k}", justify="right")

    agg = report.aggregate

    # MRR row (same value for all k)
    mrr_row = ["MRR"] + [f"{agg.mean_mrr:.4f}"] * len(k_values)
    agg_table.add_row(*mrr_row)

    for metric_name, metric_dict in [
        ("NDCG", agg.mean_ndcg),
        ("Recall", agg.mean_recall),
        ("Precision", agg.mean_precision),
        ("Hit Rate", agg.mean_hit_rate),
    ]:
        row = [metric_name]
        for k in k_values:
            val = metric_dict.get(k, 0.0)
            row.append(f"{val:.4f}")
        agg_table.add_row(*row)

    console.print()
    console.print(agg_table)

    # --- Per-query table ---
    if len(report.per_query) <= 50:
        pq_table = Table(
            title="Per-Query Metrics",
            show_header=True,
            header_style="bold green",
        )
        pq_table.add_column("Query ID", style="bold")
        pq_table.add_column("MRR", justify="right")
        for k in k_values:
            pq_table.add_column(f"NDCG@{k}", justify="right")
            pq_table.add_column(f"Recall@{k}", justify="right")

        for qm in report.per_query:
            row = [qm.query_id, f"{qm.mrr:.4f}"]
            for k in k_values:
                row.append(f"{qm.ndcg.get(k, 0.0):.4f}")
                row.append(f"{qm.recall.get(k, 0.0):.4f}")
            pq_table.add_row(*row)

        console.print()
        console.print(pq_table)
    else:
        console.print(
            f"\n[dim]({len(report.per_query)} queries — per-query table "
            "omitted; use --output json or --output csv for full details)[/dim]"
        )

    console.print(
        f"\n[dim]Evaluated {agg.num_queries} queries across "
        f"k = {k_values}[/dim]\n"
    )


def render_json(report: EvalReport, fp: TextIO | None = None) -> str:
    """Render the evaluation report as a JSON string.

    Args:
        report: The evaluation report to render.
        fp: Optional file-like object to write to. If None, returns the string.

    Returns:
        JSON string representation of the report.
    """
    # Convert int keys in dicts to strings for JSON compatibility, then back
    data = json.loads(report.model_dump_json())
    output = json.dumps(data, indent=2)
    if fp is not None:
        fp.write(output)
    return output


def render_csv(report: EvalReport, fp: TextIO | None = None) -> str:
    """Render the evaluation report as CSV.

    Each row represents one query. Columns include query_id, mrr, and
    ndcg/recall/precision/hit_rate for each k value.

    Args:
        report: The evaluation report to render.
        fp: Optional file-like object to write to. If None, returns the string.

    Returns:
        CSV string.
    """
    k_values = sorted(report.k_values)
    buf = io.StringIO()
    writer = csv.writer(buf)

    # Header
    header = ["query_id", "query_text", "mrr"]
    for k in k_values:
        header.extend([f"ndcg@{k}", f"recall@{k}", f"precision@{k}", f"hit_rate@{k}"])
    writer.writerow(header)

    # Per-query rows
    for qm in report.per_query:
        row = [qm.query_id, qm.query_text, f"{qm.mrr:.6f}"]
        for k in k_values:
            row.append(f"{qm.ndcg.get(k, 0.0):.6f}")
            row.append(f"{qm.recall.get(k, 0.0):.6f}")
            row.append(f"{qm.precision.get(k, 0.0):.6f}")
            row.append(f"{qm.hit_rate.get(k, 0.0):.6f}")
        writer.writerow(row)

    # Aggregate row
    agg = report.aggregate
    agg_row = ["__AGGREGATE__", f"({agg.num_queries} queries)", f"{agg.mean_mrr:.6f}"]
    for k in k_values:
        agg_row.append(f"{agg.mean_ndcg.get(k, 0.0):.6f}")
        agg_row.append(f"{agg.mean_recall.get(k, 0.0):.6f}")
        agg_row.append(f"{agg.mean_precision.get(k, 0.0):.6f}")
        agg_row.append(f"{agg.mean_hit_rate.get(k, 0.0):.6f}")
    writer.writerow(agg_row)

    output = buf.getvalue()
    if fp is not None:
        fp.write(output)
    return output
