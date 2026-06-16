"""Eval reporting — render results as rich terminal tables."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from localagent.evals.scenario import EvalResult


def print_result(result: EvalResult, console: Console | None = None) -> None:
    """Print a single model's eval result."""
    console = console or Console()

    table = Table(title=f"Eval: {result.model_name}")
    table.add_column("Scenario", style="cyan")
    table.add_column("Pass", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Time", justify="right", style="dim")
    table.add_column("Details", style="dim")

    for s in result.scores:
        status = "[green]PASS[/green]" if s.passed else "[red]FAIL[/red]"
        score_str = f"{s.score:.1f}/{s.max_score:.1f} ({s.percentage:.0f}%)"
        time_str = f"{s.details.get('time_seconds', 0):.1f}s"
        detail = s.error or _format_details(s.details)
        table.add_row(s.scenario_name, status, score_str, time_str, detail)

    # Summary row
    table.add_section()
    overall_status = "[green]PASS[/green]" if result.failed == 0 else "[yellow]PARTIAL[/yellow]"
    table.add_row(
        f"[bold]Total ({result.passed}/{len(result.scores)})[/bold]",
        overall_status,
        f"[bold]{result.total_score:.1f}/{result.max_total_score:.1f} ({result.percentage:.0f}%)[/bold]",
        f"{result.total_time_seconds:.1f}s",
        "",
    )

    console.print(table)
    console.print()


def print_benchmark(results: list[EvalResult], console: Console | None = None) -> None:
    """Print a comparison table across multiple models."""
    console = console or Console()

    if not results:
        console.print("[yellow]No results to display[/yellow]")
        return

    # Leaderboard
    table = Table(title="Benchmark Leaderboard")
    table.add_column("Rank", justify="center", style="bold", width=4)
    table.add_column("Model", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Time", justify="right", style="dim")

    # Sort by percentage descending
    ranked = sorted(results, key=lambda r: r.percentage, reverse=True)

    for i, r in enumerate(ranked, 1):
        medal = {1: "[yellow]1[/yellow]", 2: "2", 3: "3"}.get(i, str(i))
        score_str = f"{r.total_score:.1f}/{r.max_total_score:.1f} ({r.percentage:.0f}%)"
        pass_str = f"{r.passed}/{len(r.scores)}"
        table.add_row(medal, r.model_name, score_str, pass_str, f"{r.total_time_seconds:.1f}s")

    console.print(table)
    console.print()

    # Detailed per-scenario breakdown
    if len(results) > 1:
        _print_scenario_comparison(results, console)


def _print_scenario_comparison(results: list[EvalResult], console: Console) -> None:
    """Print per-scenario scores across all models."""
    # Collect all scenario names
    scenario_names: list[str] = []
    for r in results:
        for s in r.scores:
            if s.scenario_name not in scenario_names:
                scenario_names.append(s.scenario_name)

    table = Table(title="Per-Scenario Comparison")
    table.add_column("Scenario", style="cyan")
    for r in results:
        # Shorten model name for column header
        short_name = r.model_name.split("/")[-1]
        table.add_column(short_name, justify="center")

    for scenario_name in scenario_names:
        row = [scenario_name]
        for r in results:
            score = next((s for s in r.scores if s.scenario_name == scenario_name), None)
            if score is None:
                row.append("[dim]--[/dim]")
            elif score.passed:
                row.append(f"[green]{score.percentage:.0f}%[/green]")
            else:
                row.append(f"[red]{score.percentage:.0f}%[/red]")
        table.add_row(*row)

    console.print(table)
    console.print()


def _format_details(details: dict) -> str:
    """Format details dict into a compact string, excluding time."""
    parts = []
    for k, v in details.items():
        if k == "time_seconds":
            continue
        if isinstance(v, float):
            parts.append(f"{k}={v:.1f}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts) if parts else ""
