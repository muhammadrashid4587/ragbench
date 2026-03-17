"""Click CLI for ragbench."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import click
from rich.console import Console

from ragbench import __version__
from ragbench.dataset import DatasetError, load_dataset, validate_dataset
from ragbench.metrics import compute_all_metrics
from ragbench.models import (
    AggregateMetrics,
    EvalReport,
    QueryMetrics,
)
from ragbench.reporter import render_csv, render_json, render_table
from ragbench.retriever import (
    BaseRetriever,
    DatasetRetriever,
    HttpRetriever,
    RetrieverError,
)

console = Console(stderr=True)


@click.group()
@click.version_option(version=__version__, prog_name="ragbench")
def cli() -> None:
    """ragbench - Evaluate RAG retrieval quality with standard IR metrics."""


@cli.command()
@click.argument("dataset_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--k",
    "-k",
    "k_values",
    multiple=True,
    type=int,
    default=(5, 10, 20),
    show_default=True,
    help="Cutoff values for @k metrics. Can be specified multiple times.",
)
@click.option(
    "--output",
    "-o",
    "output_format",
    type=click.Choice(["table", "json", "csv"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--retriever-url",
    type=str,
    default=None,
    help="URL of an HTTP retriever endpoint. If not provided, uses pre-computed results from the dataset.",
)
@click.option(
    "--retriever-header",
    type=(str, str),
    multiple=True,
    help="HTTP header for the retriever (key value). Can be specified multiple times.",
)
@click.option(
    "--save",
    "-s",
    "save_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Save the report to a file (format inferred from --output).",
)
def eval(
    dataset_path: Path,
    k_values: Tuple[int, ...],
    output_format: str,
    retriever_url: Optional[str],
    retriever_header: Tuple[Tuple[str, str], ...],
    save_path: Optional[Path],
) -> None:
    """Evaluate retrieval quality on a ground-truth dataset.

    DATASET_PATH is the path to a JSON file containing queries with
    relevant document IDs and (optionally) retrieved document IDs.

    Examples:

        ragbench eval dataset.json

        ragbench eval dataset.json --k 5 --k 10 --k 20 --output json

        ragbench eval dataset.json --retriever-url http://localhost:8000/search
    """
    k_list = sorted(set(k_values))
    max_k = max(k_list)

    # Load dataset
    try:
        dataset = load_dataset(dataset_path)
    except DatasetError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    # Validate and warn
    warnings = validate_dataset(dataset)
    for warning in warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")

    # Select retriever
    retriever: BaseRetriever
    if retriever_url:
        headers = dict(retriever_header)
        retriever = HttpRetriever(url=retriever_url, headers=headers)
        console.print(f"[dim]Using HTTP retriever: {retriever_url}[/dim]")
    else:
        retriever = DatasetRetriever()
        console.print("[dim]Using pre-computed results from dataset[/dim]")

    # Evaluate
    per_query_metrics: list[QueryMetrics] = []

    with console.status("[bold green]Evaluating...") as _status:
        for query in dataset.queries:
            # Retrieve
            try:
                result = retriever.retrieve(query, top_k=max_k)
            except RetrieverError as exc:
                console.print(f"[red]Retriever error:[/red] {exc}")
                sys.exit(1)

            relevant_set = set(query.relevant_ids)
            retrieved_list = result.retrieved_ids

            # Compute metrics
            metrics = compute_all_metrics(relevant_set, retrieved_list, k_list)

            qm = QueryMetrics(
                query_id=query.id,
                query_text=query.text,
                mrr=metrics["mrr"],  # type: ignore[arg-type]
                ndcg=metrics["ndcg"],  # type: ignore[arg-type]
                recall=metrics["recall"],  # type: ignore[arg-type]
                precision=metrics["precision"],  # type: ignore[arg-type]
                hit_rate=metrics["hit_rate"],  # type: ignore[arg-type]
            )
            per_query_metrics.append(qm)

    # Aggregate
    num_queries = len(per_query_metrics)
    mean_mrr = sum(qm.mrr for qm in per_query_metrics) / num_queries

    mean_ndcg = {}
    mean_recall = {}
    mean_precision = {}
    mean_hit_rate = {}

    for k in k_list:
        mean_ndcg[k] = sum(qm.ndcg.get(k, 0.0) for qm in per_query_metrics) / num_queries
        mean_recall[k] = sum(qm.recall.get(k, 0.0) for qm in per_query_metrics) / num_queries
        mean_precision[k] = sum(qm.precision.get(k, 0.0) for qm in per_query_metrics) / num_queries
        mean_hit_rate[k] = sum(qm.hit_rate.get(k, 0.0) for qm in per_query_metrics) / num_queries

    aggregate = AggregateMetrics(
        num_queries=num_queries,
        mean_mrr=mean_mrr,
        mean_ndcg=mean_ndcg,
        mean_recall=mean_recall,
        mean_precision=mean_precision,
        mean_hit_rate=mean_hit_rate,
    )

    report = EvalReport(
        dataset_name=dataset.name,
        k_values=k_list,
        per_query=per_query_metrics,
        aggregate=aggregate,
    )

    # Output
    output_console = Console()

    if output_format == "table":
        render_table(report, console=output_console)
    elif output_format == "json":
        output = render_json(report)
        output_console.print_json(output)
    elif output_format == "csv":
        output = render_csv(report)
        click.echo(output)

    # Save to file
    if save_path:
        with open(save_path, "w", encoding="utf-8") as fp:
            if output_format == "json":
                render_json(report, fp=fp)
            elif output_format == "csv":
                render_csv(report, fp=fp)
            else:
                # For table, save as JSON by default
                render_json(report, fp=fp)
        console.print(f"[green]Report saved to {save_path}[/green]")


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
