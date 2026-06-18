"""FileOrganizerSkill — the main skill implementation.

Ties together scanner, embedder, categorizer, and mover behind the Skill interface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from localagent.config import resolve_paths
from localagent.core.embedder import DEFAULT_EMBEDDING_MODEL, Embedder
from localagent.core.engine import Engine
from localagent.core.skill import Action, Report, Skill, SkillManifest
from localagent.skills.file_organizer.categorizer import categorize
from localagent.skills.file_organizer.mover import (
    build_actions,
    execute_moves,
    undo_moves,
)
from localagent.skills.file_organizer.scanner import scan_all

logger = logging.getLogger(__name__)


class FileOrganizerSkill(Skill):
    """Organizes files in watched directories into embedding-clustered categories."""

    manifest = SkillManifest(
        name="file-organizer",
        description="Organizes files into smart, embedding-clustered categories",
        permissions={"read", "move"},
        config_key="file-organizer",
    )

    def plan(self, engine: Engine) -> list[Action]:
        """Scan watched directories and propose file categorization moves."""
        console = Console()

        # Resolve watched directories from config
        watch_dirs = resolve_paths(
            self._config.get("watch_directories", [])
        )

        if not watch_dirs:
            console.print("[yellow]No watch directories configured[/yellow]")
            return []

        console.print(
            f"[bold]Scanning {len(watch_dirs)} director{'y' if len(watch_dirs) == 1 else 'ies'}...[/bold]"
        )

        # Scan files
        profiles = scan_all(
            self.safefs,
            watch_dirs,
            exclude_patterns=self._config.get("exclude_patterns", []),
            skip_hidden=self._config.get("skip_hidden", True),
            content_preview_bytes=self._config.get("content_preview_bytes", 512),
            extra_text_extensions=self._config.get("extra_text_extensions"),
        )

        if not profiles:
            console.print("[green]No files to organize![/green]")
            return []

        console.print(f"Found [bold]{len(profiles)}[/bold] files to categorize")

        # Build embedder from config
        embedding_model = self._config.get(
            "embedding_model_path", DEFAULT_EMBEDDING_MODEL,
        )
        embedder = Embedder(model_path=embedding_model)

        console.print("[dim]Embedding files and clustering...[/dim]")

        # Categorize with embeddings + LLM naming
        result = categorize(
            engine,
            embedder,
            profiles,
            self.state_dir,
            distance_threshold=self._config.get("cluster_distance_threshold", 0.4),
            max_cluster_samples=self._config.get("max_cluster_samples", 8),
        )

        taxonomy = result.get("taxonomy", {})
        assignments = result.get("assignments", {})

        if not assignments:
            console.print("[green]No files need organizing![/green]")
            return []

        # Build concrete move actions
        profiles_by_name = {p.name: p for p in profiles}
        actions = build_actions(assignments, profiles_by_name, watch_dirs)

        # Display proposed taxonomy
        if taxonomy:
            tax_table = Table(title="Proposed Categories")
            tax_table.add_column("Category", style="bold green")
            tax_table.add_column("Description", style="dim")
            for cat_name, cat_desc in taxonomy.items():
                if isinstance(cat_desc, dict):
                    desc = cat_desc.get("description", str(cat_desc))
                else:
                    desc = str(cat_desc)
                tax_table.add_row(cat_name, desc)
            console.print(tax_table)
            console.print()

        # Display proposed moves
        move_table = Table(title=f"Proposed Moves ({len(actions)})")
        move_table.add_column("File", style="cyan")
        move_table.add_column("Category", style="green")
        move_table.add_column("Current Location", style="dim")

        for action in actions:
            src = Path(action.source)
            dst = Path(action.destination)
            move_table.add_row(
                src.name,
                dst.parent.name,
                str(src.parent),
            )

        console.print(move_table)

        return actions

    def execute(self, actions: list[Action]) -> Report:
        """Execute proposed move actions."""
        if not actions:
            return Report(
                skill_name=self.manifest.name,
                actions_performed=0,
            )

        threshold = self._config.get("move_warning_threshold", 50)
        performed, skipped, errors = execute_moves(
            actions,
            self.safefs,
            move_warning_threshold=threshold,
        )

        return Report(
            skill_name=self.manifest.name,
            actions_performed=performed,
            actions_skipped=skipped,
            errors=errors,
        )

    def undo(self, log_path: Path, *, interactive: bool = False) -> Report:
        """Reverse the last set of file moves."""
        performed, skipped, errors = undo_moves(
            self.safefs,
            interactive=interactive,
            skill_name=self.manifest.name,
        )

        return Report(
            skill_name=self.manifest.name,
            actions_performed=performed,
            actions_skipped=skipped,
            errors=errors,
        )
