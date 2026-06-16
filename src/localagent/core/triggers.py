"""Trigger system — schedule skills to run automatically.

Currently supports cron (via the system crontab). Designed to be extensible
for future trigger types (filesystem watch, webhook, etc.).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Marker used to identify localagent entries in the crontab
_CRONTAB_MARKER = "# localagent:{skill_name}"


@dataclass
class CronEntry:
    """Represents an installed cron job for a skill."""

    skill_name: str
    schedule: str
    command: str


class CronTrigger:
    """Manage cron-based scheduling for skills."""

    @staticmethod
    def _localagent_bin() -> str:
        """Find the localagent CLI binary path."""
        which = shutil.which("localagent")
        if which:
            return which
        # Fallback: use the Python that's running us
        return f"{sys.executable} -m localagent"

    @staticmethod
    def _read_crontab() -> str:
        """Read the current user crontab."""
        try:
            result = subprocess.run(
                ["crontab", "-l"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                return ""
            return result.stdout
        except FileNotFoundError:
            return ""

    @staticmethod
    def _write_crontab(content: str) -> None:
        """Write a new crontab for the current user."""
        subprocess.run(
            ["crontab", "-"],
            input=content,
            text=True,
            check=True,
        )

    def install(self, skill_name: str, schedule: str = "0 12 * * *") -> str:
        """Install a cron job for *skill_name*.

        Returns the crontab line that was added.
        """
        marker = _CRONTAB_MARKER.format(skill_name=skill_name)
        bin_path = self._localagent_bin()
        cron_line = f"{schedule} {bin_path} run {skill_name} --auto  {marker}"

        current = self._read_crontab()

        # Remove any existing entry for this skill
        lines = [
            line
            for line in current.splitlines()
            if marker not in line
        ]

        lines.append(cron_line)

        new_crontab = "\n".join(lines) + "\n"
        self._write_crontab(new_crontab)
        logger.info("Installed cron job for '%s': %s", skill_name, schedule)
        return cron_line

    def uninstall(self, skill_name: str) -> bool:
        """Remove the cron job for *skill_name*.

        Returns True if an entry was found and removed.
        """
        marker = _CRONTAB_MARKER.format(skill_name=skill_name)
        current = self._read_crontab()

        original_lines = current.splitlines()
        filtered = [line for line in original_lines if marker not in line]

        if len(filtered) == len(original_lines):
            logger.info("No cron entry found for '%s'", skill_name)
            return False

        new_crontab = "\n".join(filtered) + "\n" if filtered else ""
        self._write_crontab(new_crontab)
        logger.info("Removed cron job for '%s'", skill_name)
        return True

    def list_installed(self) -> list[CronEntry]:
        """List all localagent cron entries."""
        current = self._read_crontab()
        entries: list[CronEntry] = []

        for line in current.splitlines():
            if "# localagent:" not in line:
                continue
            # Extract skill name from marker
            marker_pos = line.index("# localagent:")
            skill_name = line[marker_pos:].split(":")[1].strip()

            # Extract schedule (first 5 fields)
            parts = line[:marker_pos].strip().split()
            if len(parts) >= 5:
                schedule = " ".join(parts[:5])
                command = " ".join(parts[5:])
                entries.append(CronEntry(
                    skill_name=skill_name,
                    schedule=schedule,
                    command=command,
                ))

        return entries
