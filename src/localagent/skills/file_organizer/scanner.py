"""File scanner — builds a profile of every file in watched directories."""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from localagent.core.safefs import SafeFS

logger = logging.getLogger(__name__)

# Extensions we can safely read as text for content preview
_DEFAULT_TEXT_EXTENSIONS: frozenset[str] = frozenset({
    ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv",
    ".json", ".jsonl", ".yaml", ".yml", ".toml", ".xml", ".html", ".htm",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h",
    ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala",
    ".sh", ".bash", ".zsh", ".fish",
    ".css", ".scss", ".less",
    ".sql", ".r", ".R", ".m", ".tex", ".bib",
    ".log", ".ini", ".cfg", ".conf", ".env",
    ".gitignore", ".dockerignore",
    ".eml", ".msg",
    ".pem", ".crt", ".crl", ".cer", ".key",
    ".ics",
    ".svg",
    ".graphqls", ".graphql", ".gql",
})


@dataclass
class FileProfile:
    """Rich metadata about a single file for the categorizer."""

    name: str
    path: Path
    extension: str
    mime_type: str
    size_bytes: int
    created: datetime | None = None
    modified: datetime | None = None
    content_preview: str | None = None
    is_readable: bool = False

    def to_summary(self) -> dict[str, Any]:
        """Compact dict representation for LLM prompts."""
        d: dict[str, Any] = {
            "name": self.name,
            "extension": self.extension or "(none)",
            "mime_type": self.mime_type,
            "size": _human_size(self.size_bytes),
        }
        if self.content_preview:
            d["content_preview"] = self.content_preview
        return d

    def embedding_text(self) -> str:
        """Build a text string for embedding this file.

        Combines filename, extension, and a truncated content preview
        (when available) into a single string for the embedding model.
        """
        parts = [self.name]
        if self.extension:
            parts.append(self.extension)
        if self.content_preview:
            # Truncate preview to keep embedding input manageable
            parts.append(self.content_preview[:200])
        return " ".join(parts)


def _human_size(num_bytes: int) -> str:
    """Convert bytes to a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024  # type: ignore[assignment]
    return f"{num_bytes:.1f} TB"


def _get_mime_type(path: Path) -> str:
    """Get MIME type, with fallback if python-magic isn't available."""
    try:
        import magic

        return magic.from_file(str(path), mime=True)
    except Exception:
        # Fallback based on extension
        ext = path.suffix.lower()
        fallback = {
            ".pdf": "application/pdf",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".mp4": "video/mp4",
            ".mp3": "audio/mpeg",
            ".zip": "application/zip",
            ".gz": "application/gzip",
            ".tar": "application/x-tar",
            ".dmg": "application/x-apple-diskimage",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".xls": "application/vnd.ms-excel",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".ppt": "application/vnd.ms-powerpoint",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        }
        return fallback.get(ext, "application/octet-stream")


def _is_text_readable(path: Path, text_extensions: frozenset[str] | None = None) -> bool:
    """Heuristic: can we read this file as text?"""
    exts = text_extensions if text_extensions is not None else _DEFAULT_TEXT_EXTENSIONS
    return path.suffix.lower() in exts


def _extract_pdf_text(path: Path, max_bytes: int = 512) -> str | None:
    """Extract text from the first page(s) of a PDF for content preview.

    Uses ``pymupdf`` (``fitz``) if installed; returns ``None`` silently if the
    library is missing or extraction fails.
    """
    try:
        import fitz  # pymupdf
    except ImportError:
        return None

    try:
        doc = fitz.open(str(path))
        text_parts: list[str] = []
        collected = 0
        for page in doc:
            page_text = page.get_text().strip()
            if page_text:
                text_parts.append(page_text)
                collected += len(page_text)
                if collected >= max_bytes:
                    break
        doc.close()
        if not text_parts:
            return None
        full = "\n".join(text_parts)
        return full[:max_bytes] if len(full) > max_bytes else full
    except Exception as exc:
        logger.debug("Cannot extract PDF text from %s: %s", path.name, exc)
        return None


def _should_exclude(name: str, exclude_patterns: list[str], skip_hidden: bool) -> bool:
    """Check if a filename matches any exclusion rule."""
    if skip_hidden and name.startswith("."):
        return True
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(name, pattern):
            return True
    return False


def scan_directory(
    safefs: SafeFS,
    directory: Path,
    *,
    exclude_patterns: list[str] | None = None,
    skip_hidden: bool = True,
    content_preview_bytes: int = 512,
    extra_text_extensions: list[str] | None = None,
) -> list[FileProfile]:
    """Scan a directory and build FileProfile objects for each file.

    Only scans immediate children — does not recurse into subdirectories
    (those are likely already-organized category folders).
    """
    exclude_patterns = exclude_patterns or []
    profiles: list[FileProfile] = []

    # Build effective text extensions set
    text_extensions = _DEFAULT_TEXT_EXTENSIONS
    if extra_text_extensions:
        text_extensions = _DEFAULT_TEXT_EXTENSIONS | frozenset(extra_text_extensions)

    try:
        entries = safefs.list_dir(directory)
    except (FileNotFoundError, PermissionError) as exc:
        logger.warning("Cannot scan %s: %s", directory, exc)
        return profiles

    for entry in entries:
        # Only process files, skip directories
        if not safefs.is_file(entry):
            continue

        name = entry.name

        if _should_exclude(name, exclude_patterns, skip_hidden):
            logger.debug("Excluded: %s", name)
            continue

        # Gather metadata
        mime_type = _get_mime_type(entry)

        try:
            stat_info = safefs.stat(entry)
        except (OSError, PermissionError):
            logger.warning("Cannot stat %s, skipping", entry)
            continue

        # Content preview for readable files
        content_preview = None
        is_readable = _is_text_readable(entry, text_extensions)
        if is_readable:
            try:
                content_preview = safefs.read_file(
                    entry, max_bytes=content_preview_bytes
                )
            except Exception as exc:
                logger.debug("Cannot read %s for preview: %s", name, exc)
                content_preview = None

        # PDF text extraction (optional, requires pymupdf)
        if content_preview is None and entry.suffix.lower() == ".pdf":
            content_preview = _extract_pdf_text(entry, max_bytes=content_preview_bytes)

        profile = FileProfile(
            name=name,
            path=entry,
            extension=entry.suffix.lower(),
            mime_type=mime_type,
            size_bytes=stat_info["size"],
            created=stat_info.get("created"),
            modified=stat_info.get("modified"),
            content_preview=content_preview,
            is_readable=is_readable,
        )
        profiles.append(profile)

    logger.info("Scanned %s: found %d files", directory, len(profiles))
    return profiles


def scan_all(
    safefs: SafeFS,
    directories: list[Path],
    *,
    exclude_patterns: list[str] | None = None,
    skip_hidden: bool = True,
    content_preview_bytes: int = 512,
    extra_text_extensions: list[str] | None = None,
) -> list[FileProfile]:
    """Scan multiple directories and return combined file profiles."""
    all_profiles: list[FileProfile] = []
    for d in directories:
        all_profiles.extend(
            scan_directory(
                safefs,
                d,
                exclude_patterns=exclude_patterns,
                skip_hidden=skip_hidden,
                content_preview_bytes=content_preview_bytes,
                extra_text_extensions=extra_text_extensions,
            )
        )
    return all_profiles
