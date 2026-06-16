"""SafeFS — sandboxed filesystem API for skills.

Skills never get raw ``os`` / ``shutil`` access.  They receive a ``SafeFS``
instance whose methods enforce:

* **Path boundaries** — every operation is checked against ``allowed_paths``.
* **Permission model** — only explicitly granted capabilities (read / move).
* **No deletion** — there is physically no ``delete()`` method.
* **No overwrite** — ``move_file`` appends a timestamp on collision.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

Permission = Literal["read", "move"]


class SafeFSError(PermissionError):
    """Raised when a filesystem operation violates sandbox rules."""


class SafeFS:
    """Sandboxed filesystem handle scoped to a set of allowed directories.

    Parameters
    ----------
    allowed_paths:
        Resolved, absolute ``Path`` objects that the skill may access.
    permissions:
        Set of granted capabilities (``"read"``, ``"move"``).
    """

    def __init__(
        self,
        allowed_paths: list[Path],
        permissions: set[Permission],
    ) -> None:
        # Resolve and store as absolute paths
        self._allowed: list[Path] = [p.resolve() for p in allowed_paths]
        self._permissions: set[Permission] = set(permissions)

    # -- boundary checks -----------------------------------------------------

    def _check_inside(self, path: Path) -> Path:
        """Resolve *path* and verify it falls within an allowed directory.

        Symlinks and ``..`` are resolved **before** the check, blocking
        traversal attacks.

        Returns the resolved path.
        """
        resolved = path.resolve()
        for allowed in self._allowed:
            try:
                resolved.relative_to(allowed)
                return resolved
            except ValueError:
                continue
        raise SafeFSError(
            f"Path {resolved} is outside allowed directories: "
            f"{[str(a) for a in self._allowed]}"
        )

    def _require(self, perm: Permission) -> None:
        if perm not in self._permissions:
            raise SafeFSError(
                f"Permission '{perm}' not granted. "
                f"Granted: {self._permissions}"
            )

    # -- read operations -----------------------------------------------------

    def list_dir(self, directory: Path) -> list[Path]:
        """List immediate children of *directory*."""
        self._require("read")
        resolved = self._check_inside(directory)
        if not resolved.is_dir():
            raise FileNotFoundError(f"Not a directory: {resolved}")
        return sorted(resolved.iterdir())

    def read_file(self, path: Path, max_bytes: int | None = None) -> str:
        """Read text content of *path*, optionally limited to *max_bytes*."""
        self._require("read")
        resolved = self._check_inside(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"Not a file: {resolved}")
        if max_bytes is not None:
            with open(resolved, "r", errors="replace") as f:
                return f.read(max_bytes)
        return resolved.read_text(errors="replace")

    def stat(self, path: Path) -> dict:
        """Return size, created, modified metadata for *path*."""
        self._require("read")
        resolved = self._check_inside(path)
        st = resolved.stat()
        return {
            "size": st.st_size,
            "created": datetime.fromtimestamp(st.st_birthtime, tz=timezone.utc),
            "modified": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc),
        }

    def exists(self, path: Path) -> bool:
        """Check whether *path* exists (within allowed boundaries)."""
        self._require("read")
        resolved = self._check_inside(path)
        return resolved.exists()

    def is_file(self, path: Path) -> bool:
        """Check whether *path* is a file."""
        self._require("read")
        resolved = self._check_inside(path)
        return resolved.is_file()

    def is_dir(self, path: Path) -> bool:
        """Check whether *path* is a directory."""
        self._require("read")
        resolved = self._check_inside(path)
        return resolved.is_dir()

    # -- write / move operations ---------------------------------------------

    def make_dir(self, path: Path) -> Path:
        """Create a directory (and parents) inside an allowed path."""
        self._require("move")
        resolved = self._check_inside(path)
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def move_file(self, src: Path, dst: Path) -> Path:
        """Move *src* to *dst*, both of which must be inside allowed paths.

        If *dst* already exists, a timestamp suffix is appended to avoid
        overwriting.  Returns the actual destination path used.
        """
        self._require("move")
        resolved_src = self._check_inside(src)
        resolved_dst = self._check_inside(dst)

        if not resolved_src.is_file():
            raise FileNotFoundError(f"Source is not a file: {resolved_src}")

        # Collision handling — never overwrite
        if resolved_dst.exists():
            ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
            stem = resolved_dst.stem
            suffix = resolved_dst.suffix
            resolved_dst = resolved_dst.with_name(f"{stem}_{ts}{suffix}")
            logger.info(
                "Destination exists, renamed to: %s", resolved_dst.name
            )

        # Ensure parent directory exists
        resolved_dst.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(str(resolved_src), str(resolved_dst))
        logger.info("Moved: %s -> %s", resolved_src, resolved_dst)
        return resolved_dst
