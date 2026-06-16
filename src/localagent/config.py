"""Configuration loading, merging, and persistence."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import yaml


CONFIG_DIR = Path.home() / ".config" / "localagent"
DATA_DIR = Path.home() / ".local" / "share" / "localagent"
LOG_DIR = DATA_DIR / "logs"

# Shipped default config bundled with the package
_PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent  # repo root
DEFAULT_CONFIG_PATH = _PACKAGE_DIR / "config" / "default.yaml"

USER_CONFIG_PATH = CONFIG_DIR / "config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config() -> dict[str, Any]:
    """Load configuration with user overrides merged on top of defaults.

    Precedence: user config (~/.config/localagent/config.yaml) > default.yaml
    """
    # Load shipped defaults
    if DEFAULT_CONFIG_PATH.exists():
        with open(DEFAULT_CONFIG_PATH) as f:
            base = yaml.safe_load(f) or {}
    else:
        base = {}

    # Load user overrides if present
    if USER_CONFIG_PATH.exists():
        with open(USER_CONFIG_PATH) as f:
            user = yaml.safe_load(f) or {}
        config = _deep_merge(base, user)
    else:
        config = base

    return config


def ensure_config_dir() -> None:
    """Create the config directory if it doesn't exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def ensure_data_dirs() -> None:
    """Create data and log directories if they don't exist."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def init_user_config() -> Path:
    """Copy default config to user config location if it doesn't exist.

    Returns the path to the user config file.
    """
    ensure_config_dir()
    if not USER_CONFIG_PATH.exists() and DEFAULT_CONFIG_PATH.exists():
        shutil.copy2(DEFAULT_CONFIG_PATH, USER_CONFIG_PATH)
    return USER_CONFIG_PATH


def skill_state_dir(skill_name: str) -> Path:
    """Return the state directory for a skill, creating it if needed.

    e.g. ~/.config/localagent/file-organizer/
    """
    d = CONFIG_DIR / skill_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_skill_config(config: dict[str, Any], skill_name: str) -> dict[str, Any]:
    """Extract a specific skill's configuration section."""
    return config.get("skills", {}).get(skill_name, {})


def get_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract the model configuration section."""
    return config.get("model", {})


def resolve_paths(paths: list[str]) -> list[Path]:
    """Expand ~ and env vars in a list of path strings, return resolved Paths."""
    resolved = []
    for p in paths:
        expanded = Path(os.path.expandvars(os.path.expanduser(p))).resolve()
        resolved.append(expanded)
    return resolved
