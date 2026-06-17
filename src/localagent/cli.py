"""CLI entry point for localagent."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from localagent.config import (
    CONFIG_DIR,
    LOG_DIR,
    USER_CONFIG_PATH,
    get_model_config,
    init_user_config,
    load_config,
)
from localagent.core.engine import Engine
from localagent.core.registry import get_registry, register_builtin_skills
from localagent.core.triggers import CronTrigger

console = Console()


# ── Logging setup ───────────────────────────────────────────────────────────


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


# ── Commands ────────────────────────────────────────────────────────────────


def cmd_run(args: argparse.Namespace) -> int:
    """Run a skill — interactive or auto mode."""
    config = load_config()
    registry = get_registry()

    skill = registry.instantiate(args.skill, config)
    model_cfg = get_model_config(config)
    model_path = args.model if args.model else model_cfg.get("model_path", "mlx-community/Llama-3.2-3B-Instruct-4bit")
    engine = Engine(
        model_path=model_path,
        max_tokens=model_cfg.get("max_tokens", 2048),
        temperature=model_cfg.get("temperature", 0.3),
    )

    try:
        actions = skill.plan(engine)
    except Exception as exc:
        if args.auto:
            logging.getLogger(__name__).error("Plan failed: %s", exc)
            return 1
        raise

    if not actions:
        console.print("[green]Nothing to do.[/green]")
        return 0

    # Auto mode: execute immediately
    if args.auto:
        report = skill.execute(actions)
        if not report.success:
            logging.getLogger(__name__).error(
                "Execution failed with %d errors: %s",
                len(report.errors),
                report.errors,
            )
            return 1
        console.print(
            f"[green]Done: {report.actions_performed} actions performed[/green]"
        )
        return 0

    # Interactive mode: confirm first
    console.print()
    response = console.input(
        "[bold]Proceed with these moves? [y/N]: [/bold]"
    )
    if response.strip().lower() not in ("y", "yes"):
        console.print("[yellow]Aborted.[/yellow]")
        return 0

    report = skill.execute(actions)
    if report.errors:
        for err in report.errors:
            console.print(f"[red]Error: {err}[/red]")

    console.print(
        f"\n[green]Done: {report.actions_performed} moved, "
        f"{report.actions_skipped} skipped[/green]"
    )
    return 0


def cmd_undo(args: argparse.Namespace) -> int:
    """Undo the last run of a skill."""
    config = load_config()
    registry = get_registry()

    skill = registry.instantiate(args.skill, config)
    report = skill.undo(Path("."), interactive=args.interactive)

    if report.errors:
        for err in report.errors:
            console.print(f"[red]Error: {err}[/red]")

    console.print(
        f"\n[green]Undo: {report.actions_performed} reversed, "
        f"{report.actions_skipped} skipped[/green]"
    )
    return 0 if report.success else 1


def cmd_schedule(args: argparse.Namespace) -> int:
    """Install a cron trigger for a skill."""
    trigger = CronTrigger()
    cron_line = trigger.install(args.skill, schedule=args.cron)
    console.print(f"[green]Installed cron job:[/green]\n{cron_line}")
    return 0


def cmd_unschedule(args: argparse.Namespace) -> int:
    """Remove a cron trigger for a skill."""
    trigger = CronTrigger()
    removed = trigger.uninstall(args.skill)
    if removed:
        console.print(f"[green]Removed cron job for '{args.skill}'[/green]")
    else:
        console.print(f"[yellow]No cron job found for '{args.skill}'[/yellow]")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Show configuration."""
    config_path = init_user_config()
    config = load_config()

    console.print(f"[bold]Config file:[/bold] {config_path}")
    console.print()

    import yaml

    console.print(yaml.dump(config, default_flow_style=False))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Run evals against a skill, optionally benchmarking multiple models."""
    from localagent.evals.runner import run_benchmark, save_results
    from localagent.evals.report import print_result, print_benchmark

    config = load_config()
    model_cfg = get_model_config(config)

    # Resolve models
    if args.models:
        model_paths = args.models
    else:
        model_paths = [model_cfg.get("model_path", "mlx-community/Llama-3.2-3B-Instruct-4bit")]

    # Resolve scenarios
    if args.skill == "file-organizer":
        from localagent.skills.file_organizer.evals.scenarios import get_scenarios
        scenarios = get_scenarios(args.scenarios)
    else:
        console.print(f"[red]No evals defined for skill '{args.skill}'[/red]")
        return 1

    if not scenarios:
        console.print("[yellow]No matching scenarios found[/yellow]")
        return 1

    console.print(
        f"[bold]Running {len(scenarios)} scenario(s) against "
        f"{len(model_paths)} model(s)[/bold]\n"
    )

    results = run_benchmark(
        model_paths,
        scenarios,
        skill_name=args.skill,
        max_tokens=model_cfg.get("max_tokens", 4096),
        temperature=model_cfg.get("temperature", 0.3),
    )

    # Display results
    if len(results) == 1:
        print_result(results[0], console)
    else:
        print_benchmark(results, console)

    # Save if requested
    if args.output:
        save_results(results, Path(args.output))
        console.print(f"[dim]Results saved to {args.output}[/dim]")

    # Exit non-zero if any model had failures
    any_failures = any(r.failed > 0 for r in results)
    return 1 if any_failures else 0


def cmd_list(args: argparse.Namespace) -> int:
    """List available skills."""
    registry = get_registry()
    manifests = registry.list_manifests()

    if not manifests:
        console.print("[yellow]No skills registered.[/yellow]")
        return 0

    table = Table(title="Available Skills")
    table.add_column("Name", style="bold cyan")
    table.add_column("Description")
    table.add_column("Permissions", style="dim")

    for m in manifests:
        table.add_row(
            m["name"],
            m["description"],
            ", ".join(m["permissions"]),
        )

    console.print(table)
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    """Reset a skill by clearing its learned state and undo journals."""
    import shutil

    skill_name = args.skill
    state_dir = CONFIG_DIR / skill_name
    journal_pattern = f"{skill_name}-*.jsonl"

    # Collect what exists
    has_state = state_dir.exists() and any(state_dir.iterdir())
    journals = list(LOG_DIR.glob(journal_pattern)) if LOG_DIR.exists() else []

    if not has_state and not journals:
        console.print(f"[yellow]Nothing to reset for '{skill_name}'.[/yellow]")
        return 0

    # Show what will be deleted
    console.print(f"[bold]Will reset skill '{skill_name}':[/bold]")
    if has_state:
        for f in sorted(state_dir.iterdir()):
            console.print(f"  [dim]state:[/dim]  {f}")
    for j in sorted(journals):
        console.print(f"  [dim]journal:[/dim] {j}")

    # Confirm unless --force
    if not args.force:
        response = console.input("\n[bold]Proceed? [y/N]: [/bold]")
        if response.strip().lower() not in ("y", "yes"):
            console.print("[yellow]Aborted.[/yellow]")
            return 0

    # Delete state directory contents
    removed = 0
    if has_state:
        for f in state_dir.iterdir():
            if f.is_file():
                f.unlink()
                removed += 1
            elif f.is_dir():
                shutil.rmtree(f)
                removed += 1

    # Delete journal files
    for j in journals:
        j.unlink()
        removed += 1

    console.print(f"[green]Reset '{skill_name}': removed {removed} file(s).[/green]")
    return 0


# ── Argument parser ─────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="localagent",
        description="Local AI agent framework powered by MLX",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # run
    run_parser = sub.add_parser("run", help="Run a skill")
    run_parser.add_argument("skill", help="Skill name (e.g. file-organizer)")
    run_parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-execute without confirmation (for cron)",
    )
    run_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the model to use (e.g. mlx-community/gemma-2-9b-it-4bit)",
    )

    # undo
    undo_parser = sub.add_parser("undo", help="Undo the last run of a skill")
    undo_parser.add_argument("skill", help="Skill name")
    undo_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactively select which moves to undo",
    )

    # schedule
    sched_parser = sub.add_parser("schedule", help="Install a cron trigger")
    sched_parser.add_argument("skill", help="Skill name")
    sched_parser.add_argument(
        "--cron",
        default="0 12 * * *",
        help="Cron schedule expression (default: '0 12 * * *')",
    )

    # unschedule
    unsched_parser = sub.add_parser("unschedule", help="Remove a cron trigger")
    unsched_parser.add_argument("skill", help="Skill name")

    # config
    sub.add_parser("config", help="Show current configuration")

    # list
    sub.add_parser("list", help="List available skills")

    # reset
    reset_parser = sub.add_parser(
        "reset", help="Reset a skill (clear learned state and journals)"
    )
    reset_parser.add_argument("skill", help="Skill name (e.g. file-organizer)")
    reset_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )

    # eval
    eval_parser = sub.add_parser("eval", help="Run evals against a skill")
    eval_parser.add_argument("skill", help="Skill name (e.g. file-organizer)")
    eval_parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model to evaluate (can be repeated for benchmarking). "
        "Defaults to the configured model.",
    )
    eval_parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        help="Run specific scenario(s) by name (default: all)",
    )
    eval_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to a YAML file",
    )

    return parser


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    _setup_logging(verbose=args.verbose if hasattr(args, "verbose") else False)

    # Register built-in skills
    register_builtin_skills()

    command_map = {
        "run": cmd_run,
        "undo": cmd_undo,
        "schedule": cmd_schedule,
        "unschedule": cmd_unschedule,
        "config": cmd_config,
        "list": cmd_list,
        "reset": cmd_reset,
        "eval": cmd_eval,
    }

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    handler = command_map.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        exit_code = handler(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        exit_code = 130
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        logging.getLogger(__name__).debug("Full traceback:", exc_info=True)
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
