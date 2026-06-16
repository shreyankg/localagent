"""Tests for SafeFS sandboxed filesystem."""

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from localagent.core.safefs import SafeFS, SafeFSError


@pytest.fixture
def sandbox(tmp_path):
    """Create a sandbox with two allowed directories and some test files."""
    dir_a = tmp_path / "allowed_a"
    dir_b = tmp_path / "allowed_b"
    forbidden = tmp_path / "forbidden"

    for d in (dir_a, dir_b, forbidden):
        d.mkdir()

    # Create test files
    (dir_a / "hello.txt").write_text("hello world")
    (dir_a / "data.csv").write_text("a,b,c\n1,2,3\n")
    (dir_b / "notes.md").write_text("# Notes\nSome notes here")
    (forbidden / "secret.txt").write_text("top secret")

    return {
        "tmp": tmp_path,
        "dir_a": dir_a,
        "dir_b": dir_b,
        "forbidden": forbidden,
        "safefs": SafeFS(
            allowed_paths=[dir_a, dir_b],
            permissions={"read", "move"},
        ),
    }


class TestPathBoundaries:
    def test_read_allowed_file(self, sandbox):
        content = sandbox["safefs"].read_file(sandbox["dir_a"] / "hello.txt")
        assert content == "hello world"

    def test_read_forbidden_file_raises(self, sandbox):
        with pytest.raises(SafeFSError):
            sandbox["safefs"].read_file(sandbox["forbidden"] / "secret.txt")

    def test_list_dir_allowed(self, sandbox):
        entries = sandbox["safefs"].list_dir(sandbox["dir_a"])
        names = [e.name for e in entries]
        assert "hello.txt" in names
        assert "data.csv" in names

    def test_list_dir_forbidden_raises(self, sandbox):
        with pytest.raises(SafeFSError):
            sandbox["safefs"].list_dir(sandbox["forbidden"])

    def test_path_traversal_blocked(self, sandbox):
        """Attempting to escape via .. should be caught."""
        sneaky = sandbox["dir_a"] / ".." / "forbidden" / "secret.txt"
        with pytest.raises(SafeFSError):
            sandbox["safefs"].read_file(sneaky)

    def test_symlink_escape_blocked(self, sandbox):
        """Symlinks pointing outside the sandbox should be caught."""
        link = sandbox["dir_a"] / "escape_link"
        link.symlink_to(sandbox["forbidden"] / "secret.txt")
        with pytest.raises(SafeFSError):
            sandbox["safefs"].read_file(link)


class TestPermissions:
    def test_read_only_cannot_move(self, sandbox):
        read_only = SafeFS(
            allowed_paths=[sandbox["dir_a"]],
            permissions={"read"},
        )
        with pytest.raises(SafeFSError, match="move"):
            read_only.move_file(
                sandbox["dir_a"] / "hello.txt",
                sandbox["dir_a"] / "subdir" / "hello.txt",
            )

    def test_read_only_cannot_make_dir(self, sandbox):
        read_only = SafeFS(
            allowed_paths=[sandbox["dir_a"]],
            permissions={"read"},
        )
        with pytest.raises(SafeFSError, match="move"):
            read_only.make_dir(sandbox["dir_a"] / "new_subdir")


class TestMoveOperations:
    def test_move_within_sandbox(self, sandbox):
        src = sandbox["dir_a"] / "hello.txt"
        subdir = sandbox["dir_a"] / "organized"
        dst = subdir / "hello.txt"

        actual = sandbox["safefs"].move_file(src, dst)
        assert actual.exists()
        assert actual.read_text() == "hello world"
        assert not src.exists()

    def test_move_collision_appends_timestamp(self, sandbox):
        src = sandbox["dir_a"] / "hello.txt"
        dst = sandbox["dir_b"] / "notes.md"  # already exists

        # Move hello.txt to dir_b with name "notes.md"
        actual = sandbox["safefs"].move_file(src, dst)
        assert actual.exists()
        assert "notes" in actual.name
        # Original notes.md should still exist untouched
        assert (sandbox["dir_b"] / "notes.md").exists()

    def test_move_outside_sandbox_raises(self, sandbox):
        with pytest.raises(SafeFSError):
            sandbox["safefs"].move_file(
                sandbox["dir_a"] / "hello.txt",
                sandbox["forbidden"] / "hello.txt",
            )

    def test_move_creates_parent_dirs(self, sandbox):
        src = sandbox["dir_a"] / "data.csv"
        dst = sandbox["dir_a"] / "deep" / "nested" / "dir" / "data.csv"

        actual = sandbox["safefs"].move_file(src, dst)
        assert actual.exists()
        assert actual.read_text() == "a,b,c\n1,2,3\n"


class TestStatAndMetadata:
    def test_stat_returns_size(self, sandbox):
        info = sandbox["safefs"].stat(sandbox["dir_a"] / "hello.txt")
        assert info["size"] == len("hello world")
        assert isinstance(info["modified"], datetime)

    def test_exists_and_is_file(self, sandbox):
        assert sandbox["safefs"].exists(sandbox["dir_a"] / "hello.txt")
        assert sandbox["safefs"].is_file(sandbox["dir_a"] / "hello.txt")
        assert not sandbox["safefs"].is_file(sandbox["dir_a"])

    def test_read_with_max_bytes(self, sandbox):
        content = sandbox["safefs"].read_file(
            sandbox["dir_a"] / "hello.txt", max_bytes=5
        )
        assert content == "hello"


class TestNoDeletePrimitive:
    def test_no_delete_method(self, sandbox):
        """SafeFS must not have any delete/remove method."""
        assert not hasattr(sandbox["safefs"], "delete")
        assert not hasattr(sandbox["safefs"], "remove")
        assert not hasattr(sandbox["safefs"], "unlink")
        assert not hasattr(sandbox["safefs"], "rmdir")
        assert not hasattr(sandbox["safefs"], "rmtree")
