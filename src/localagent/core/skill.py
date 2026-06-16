"""Base Skill class and SkillManifest — the contract every skill must follow."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from localagent.config import skill_state_dir
from localagent.core.engine import Engine
from localagent.core.safefs import Permission, SafeFS


@dataclass(frozen=True)
class SkillManifest:
    """Declares what a skill needs to operate.

    The framework validates this at registration time and constructs a
    sandboxed ``SafeFS`` accordingly.

    Attributes
    ----------
    name:
        Unique identifier for the skill (e.g. ``"file-organizer"``).
    description:
        Human-readable one-liner shown in ``localagent list``.
    permissions:
        Set of filesystem capabilities the skill requires.
    config_key:
        Key under ``skills:`` in the config YAML. Defaults to *name*.
    """

    name: str
    description: str
    permissions: set[Permission] = field(default_factory=lambda: {"read"})
    config_key: str | None = None

    @property
    def effective_config_key(self) -> str:
        return self.config_key or self.name


@dataclass
class Action:
    """A single proposed action a skill wants to perform.

    Used for dry-run display and undo journaling.
    """

    action_type: str  # e.g. "move", "create_dir"
    source: str | None = None
    destination: str | None = None
    detail: str = ""  # human-readable description


@dataclass
class Report:
    """Summary returned after a skill execution."""

    skill_name: str
    actions_performed: int
    actions_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    log_path: str | None = None

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


class Skill(abc.ABC):
    """Abstract base class for all skills.

    Subclasses must define ``manifest`` and implement ``plan`` / ``execute``.
    """

    # -- to be set by subclass -----------------------------------------------

    manifest: SkillManifest

    # -- set by the framework at configure time ------------------------------

    _safefs: SafeFS | None = None
    _config: dict[str, Any] = {}

    # -- lifecycle -----------------------------------------------------------

    def configure(
        self,
        config: dict[str, Any],
        safefs: SafeFS,
    ) -> None:
        """Called by the framework after construction.

        ``config`` is the skill-specific section from the YAML.
        ``safefs`` is a sandboxed FS handle scoped to the skill's allowed paths.
        """
        self._config = config
        self._safefs = safefs

    @property
    def safefs(self) -> SafeFS:
        assert self._safefs is not None, "Skill not configured — call configure() first"
        return self._safefs

    @property
    def state_dir(self) -> Path:
        """Per-skill state directory: ``~/.config/localagent/<skill-name>/``."""
        return skill_state_dir(self.manifest.name)

    # -- abstract interface --------------------------------------------------

    @abc.abstractmethod
    def plan(self, engine: Engine) -> list[Action]:
        """Dry-run: scan and return proposed actions without executing.

        The engine is provided for LLM queries.
        """
        ...

    @abc.abstractmethod
    def execute(self, actions: list[Action]) -> Report:
        """Execute the given actions and return a report."""
        ...

    @abc.abstractmethod
    def undo(self, log_path: Path, *, interactive: bool = False) -> Report:
        """Reverse actions from *log_path*.

        If *interactive* is True, prompt the user to select which actions
        to reverse.
        """
        ...
