"""Skill registry — discovers, validates, and instantiates skills."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from localagent.config import get_skill_config, resolve_paths
from localagent.core.safefs import SafeFS
from localagent.core.skill import Skill
from localagent.core.engine import Engine

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Central registry of available skills.

    Skills are registered explicitly (not via entry-points magic) for
    clarity and auditability.
    """

    def __init__(self) -> None:
        self._skills: dict[str, type[Skill]] = {}

    def register(self, skill_cls: type[Skill]) -> None:
        """Register a skill class.

        Validates that the class has a proper manifest.
        """
        if not hasattr(skill_cls, "manifest"):
            raise TypeError(
                f"Skill class {skill_cls.__name__} is missing a 'manifest' attribute"
            )
        name = skill_cls.manifest.name
        if name in self._skills:
            raise ValueError(f"Skill '{name}' is already registered")
        self._skills[name] = skill_cls
        logger.info("Registered skill: %s", name)

    def get(self, name: str) -> type[Skill] | None:
        """Look up a skill class by name."""
        return self._skills.get(name)

    def list_skills(self) -> list[str]:
        """Return names of all registered skills."""
        return sorted(self._skills.keys())

    def list_manifests(self) -> list[dict[str, Any]]:
        """Return manifest info for all registered skills."""
        result = []
        for name in sorted(self._skills):
            m = self._skills[name].manifest
            result.append({
                "name": m.name,
                "description": m.description,
                "permissions": sorted(m.permissions),
            })
        return result

    def instantiate(
        self,
        name: str,
        config: dict[str, Any],
    ) -> Skill:
        """Create a configured skill instance.

        Reads the skill's config section and allowed_paths from the global
        config, constructs a ``SafeFS``, and calls ``skill.configure()``.
        """
        skill_cls = self._skills.get(name)
        if skill_cls is None:
            raise KeyError(f"Unknown skill: '{name}'")

        manifest = skill_cls.manifest
        skill_config = get_skill_config(config, manifest.effective_config_key)

        # Resolve allowed paths from the skill config
        raw_paths = skill_config.get("watch_directories", [])
        allowed_paths = resolve_paths(raw_paths)

        # Validate paths exist
        for p in allowed_paths:
            if not p.is_dir():
                logger.warning("Configured path does not exist: %s", p)

        # Build sandboxed FS
        safefs = SafeFS(
            allowed_paths=allowed_paths,
            permissions=manifest.permissions,
        )

        # Instantiate and configure
        skill = skill_cls()
        skill.configure(skill_config, safefs)
        return skill


# -- global registry ---------------------------------------------------------

_registry = SkillRegistry()


def get_registry() -> SkillRegistry:
    """Return the global skill registry."""
    return _registry


def register_builtin_skills() -> None:
    """Register all built-in skills.

    Called once at startup.
    """
    from localagent.skills.file_organizer.skill import FileOrganizerSkill

    _registry.register(FileOrganizerSkill)
