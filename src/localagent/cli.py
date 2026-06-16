"""CLI entry point for localagent."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from localagent.config import (
    USER_CONFIG_PATH,
    get_model_config,
    init_user_config,
    load_config,
    skill_state_dir,
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
    engine = Engine(
        model_path=model_cfg.get("model_path", "mlx-community/Llama-3.2-3B-Instruct-4bit"),
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


def cmd_taxonomy(args: argparse.Namespace) -> int:
    """Manage the learned taxonomy."""
    state_dir = skill_state_dir(args.skill)
    taxonomy_path = state_dir / "taxonomy.yaml"

    if args.taxonomy_action == "show":
        if not taxonomy_path.exists():
            console.print("[yellow]No taxonomy learned yet.[/yellow]")
            return 0
        content = taxonomy_path.read_text()
        console.print(f"[bold]Taxonomy[/bold] ({taxonomy_path}):\n")
        console.print(content)
        return 0

    elif args.taxonomy_action == "edit":
        if not taxonomy_path.exists():
            console.print("[yellow]No taxonomy learned yet. Run the skill first.[/yellow]")
            return 1
        editor = os.environ.get("EDITOR", "nano")
        subprocess.run([editor, str(taxonomy_path)])
        return 0

    elif args.taxonomy_action == "reset":
        if taxonomy_path.exists():
            taxonomy_path.unlink()
            console.print("[green]Taxonomy reset. Next run will start fresh.[/green]")
        else:
            console.print("[yellow]No taxonomy to reset.[/yellow]")
        return 0

    return 1


def cmd_config(args: argparse.Namespace) -> int:
    """Show configuration."""
    config_path = init_user_config()
    config = load_config()

    console.print(f"[bold]Config file:[/bold] {config_path}")
    console.print()

    import yaml

    console.print(yaml.dump(config, default_flow_style=False))
    return 0


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

    # taxonomy
    tax_parser = sub.add_parser("taxonomy", help="Manage learned taxonomy")
    tax_parser.add_argument("taxonomy_action", choices=["show", "edit", "reset"])
    tax_parser.add_argument(
        "--skill",
        default="file-organizer",
        help="Skill name (default: file-organizer)",
    )

    # config
    sub.add_parser("config", help="Show current configuration")

    # list
    sub.add_parser("list", help="List available skills")

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
        "taxonomy": cmd_taxonomy,
        "config": cmd_config,
        "list": cmd_list,
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
