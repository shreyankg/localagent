"""Tests for configuration loading and merging."""

import tempfile
from pathlib import Path

import pytest
import yaml

from localagent.config import (
    _deep_merge,
    get_model_config,
    get_skill_config,
    resolve_paths,
)


class TestDeepMerge:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"model": {"path": "default", "temp": 0.5}}
        override = {"model": {"temp": 0.3}}
        result = _deep_merge(base, override)
        assert result == {"model": {"path": "default", "temp": 0.3}}

    def test_base_unchanged(self):
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        _deep_merge(base, override)
        assert base["a"]["b"] == 1  # original not mutated

    def test_empty_override(self):
        base = {"a": 1}
        result = _deep_merge(base, {})
        assert result == {"a": 1}


class TestConfigHelpers:
    def test_get_model_config(self):
        config = {"model": {"model_path": "some-model", "max_tokens": 1024}}
        mc = get_model_config(config)
        assert mc["model_path"] == "some-model"
        assert mc["max_tokens"] == 1024

    def test_get_model_config_missing(self):
        assert get_model_config({}) == {}

    def test_get_skill_config(self):
        config = {
            "skills": {
                "file-organizer": {"watch_directories": ["~/Desktop"]},
            }
        }
        sc = get_skill_config(config, "file-organizer")
        assert sc["watch_directories"] == ["~/Desktop"]

    def test_get_skill_config_missing(self):
        assert get_skill_config({}, "nonexistent") == {}

    def test_resolve_paths_expands_tilde(self):
        paths = resolve_paths(["~/Desktop"])
        assert len(paths) == 1
        assert "~" not in str(paths[0])
        assert paths[0].is_absolute()

    def test_resolve_paths_empty(self):
        assert resolve_paths([]) == []
