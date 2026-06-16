"""Safe file mover with undo journal.

All file operations go through SafeFS — no raw os/shutil usage.
Every move is journaled before execution so it can be reversed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from localagent.config import LOG_DIR, ensure_data_dirs
from localagent.core.safefs import SafeFS
from localagent.core.skill import Action

logger = logging.getLogger(__name__)


@dataclass
class MoveRecord:
    """A single move operation, persisted to the undo journal."""

    filename: str
    source: str
    destination: str
    category: str
    timestamp: str

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, line: str) -> MoveRecord:
        return cls(**json.loads(line))


def _journal_path(skill_name: str = "file-organizer") -> Path:
    """Return today's journal file path."""
    ensure_data_dirs()
    date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    return LOG_DIR / f"{skill_name}-{date_str}.jsonl"


def _latest_journal(skill_name: str = "file-organizer") -> Path | None:
    """Find the most recent journal file."""
    ensure_data_dirs()
    journals = sorted(LOG_DIR.glob(f"{skill_name}-*.jsonl"), reverse=True)
    return journals[0] if journals else None


def build_actions(
    assignments: dict[str, str],
    profiles_by_name: dict[str, Any],
    watch_directories: list[Path],
) -> list[Action]:
    """Convert LLM assignments into concrete move Actions.

    Files are moved into a category subfolder within their current
    parent directory (e.g. ``~/Downloads/Screenshots/image.png``).
    """
    actions: list[Action] = []

    for filename, category in assignments.items():
        profile = profiles_by_name.get(filename)
        if profile is None:
            logger.warning("No profile for '%s' — skipping", filename)
            continue

        src = profile.path
        # Place category folder inside the file's current parent directory
        dst = src.parent / category / filename

        actions.append(
            Action(
                action_type="move",
                source=str(src),
                destination=str(dst),
                detail=f"{filename} → {category}/",
            )
        )

    return actions


def execute_moves(
    actions: list[Action],
    safefs: SafeFS,
    *,
    move_warning_threshold: int = 50,
) -> tuple[int, int, list[str]]:
    """Execute move actions via SafeFS, journaling each one.

    Returns (performed, skipped, errors).
    """
    if len(actions) > move_warning_threshold:
        logger.warning(
            "High move count: %d moves proposed (threshold: %d)",
            len(actions),
            move_warning_threshold,
        )

    journal = _journal_path()
    performed = 0
    skipped = 0
    errors: list[str] = []

    for action in actions:
        if action.action_type != "move":
            continue

        src = Path(action.source)
        dst = Path(action.destination)

        try:
            # Ensure category directory exists
            safefs.make_dir(dst.parent)

            # Journal BEFORE moving (for crash recovery)
            record = MoveRecord(
                filename=src.name,
                source=str(src),
                destination=str(dst),
                category=dst.parent.name,
                timestamp=datetime.now(tz=timezone.utc).isoformat(),
            )
            with open(journal, "a") as f:
                f.write(record.to_json() + "\n")

            # Execute the move
            actual_dst = safefs.move_file(src, dst)
            performed += 1

            # If destination changed due to collision, update journal
            if str(actual_dst) != str(dst):
                record.destination = str(actual_dst)
                # Rewrite last line (simplistic but works for append-only)
                lines = journal.read_text().splitlines()
                lines[-1] = record.to_json()
                journal.write_text("\n".join(lines) + "\n")

        except Exception as exc:
            msg = f"Failed to move '{src.name}': {exc}"
            logger.error(msg)
            errors.append(msg)
            skipped += 1

    logger.info(
        "Moves complete: %d performed, %d skipped, %d errors",
        performed,
        skipped,
        len(errors),
    )
    return performed, skipped, errors


def undo_moves(
    safefs: SafeFS,
    *,
    interactive: bool = False,
    skill_name: str = "file-organizer",
) -> tuple[int, int, list[str]]:
    """Reverse moves from the latest journal.

    If *interactive*, show a selection UI. Otherwise, reverse all.
    """
    journal = _latest_journal(skill_name)
    if journal is None:
        logger.info("No journal found — nothing to undo")
        return 0, 0, []

    # Read all records
    records: list[MoveRecord] = []
    for line in journal.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(MoveRecord.from_json(line))

    if not records:
        logger.info("Journal is empty — nothing to undo")
        return 0, 0, []

    # Interactive selection
    if interactive:
        records = _interactive_select(records)
        if not records:
            logger.info("No moves selected for undo")
            return 0, 0, []

    performed = 0
    skipped = 0
    errors: list[str] = []

    for record in reversed(records):
        src = Path(record.destination)  # reverse: dst → src
        dst = Path(record.source)  # reverse: src → dst

        try:
            if not src.exists():
                logger.warning("File not found for undo: %s", src)
                skipped += 1
                continue

            actual = safefs.move_file(src, dst)
            performed += 1
            logger.info("Undone: %s → %s", src, actual)
        except Exception as exc:
            msg = f"Undo failed for '{record.filename}': {exc}"
            logger.error(msg)
            errors.append(msg)
            skipped += 1

    # Clean up empty category directories
    _cleanup_empty_dirs(records, safefs)

    logger.info(
        "Undo complete: %d reversed, %d skipped, %d errors",
        performed,
        skipped,
        len(errors),
    )
    return performed, skipped, errors


def _interactive_select(records: list[MoveRecord]) -> list[MoveRecord]:
    """Show moves and let the user select which to undo."""
    console = Console()

    table = Table(title="Moves to Undo")
    table.add_column("#", style="dim", width=4)
    table.add_column("File", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("From", style="dim")
    table.add_column("To", style="dim")

    for i, rec in enumerate(records):
        table.add_row(
            str(i + 1),
            rec.filename,
            rec.category,
            rec.source,
            rec.destination,
        )

    console.print(table)
    console.print()

    response = console.input(
        "[bold]Enter numbers to undo (comma-separated, or 'all'): [/bold]"
    )
    response = response.strip().lower()

    if response == "all":
        return records

    try:
        indices = [int(x.strip()) - 1 for x in response.split(",")]
        return [records[i] for i in indices if 0 <= i < len(records)]
    except (ValueError, IndexError):
        console.print("[red]Invalid selection — aborting undo[/red]")
        return []


def _cleanup_empty_dirs(
    records: list[MoveRecord], safefs: SafeFS
) -> None:
    """Remove category directories that are now empty after undo."""
    dirs_to_check = set()
    for rec in records:
        dst = Path(rec.destination)
        dirs_to_check.add(dst.parent)

    for d in dirs_to_check:
        try:
            if d.exists() and d.is_dir() and not any(d.iterdir()):
                d.rmdir()
                logger.info("Removed empty directory: %s", d)
        except OSError:
            pass  # Not empty or permission issue — leave it
