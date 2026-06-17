"""File scanner — builds a rich profile of every file in watched directories."""

from __future__ import annotations

import fnmatch
import logging
import re
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
    # Plain-text formats often missed: emails, certs, calendar, vector graphics
    ".eml", ".msg",
    ".pem", ".crt", ".crl", ".cer", ".key",
    ".ics",
    ".svg",
    ".graphqls", ".graphql", ".gql",
})


# ── Extension → suggested category ────────────────────────────────────────
# Deterministic pre-classification for unambiguous extensions.  Stored as
# hints["suggested_category"] so the LLM can still override when the
# filename or content suggests something more specific.

_EXTENSION_CATEGORIES: dict[str, str] = {
    # Spreadsheets (NOT documents — they have tabular data)
    ".csv": "Spreadsheets", ".tsv": "Spreadsheets",
    ".xlsx": "Spreadsheets", ".xls": "Spreadsheets", ".ods": "Spreadsheets",
    # Installers
    ".dmg": "Installers", ".pkg": "Installers", ".msi": "Installers",
    ".deb": "Installers", ".rpm": "Installers", ".snap": "Installers",
    ".appimage": "Installers",
    # Emails
    ".eml": "Emails", ".msg": "Emails",
    # Photos / Images
    ".heic": "Photos", ".jpg": "Photos", ".jpeg": "Photos",
    ".png": "Photos", ".gif": "Photos", ".bmp": "Photos",
    ".tiff": "Photos", ".tif": "Photos", ".webp": "Photos",
    ".svg": "Photos", ".eps": "Photos", ".raw": "Photos",
    # Videos
    ".mp4": "Videos", ".mov": "Videos", ".mkv": "Videos",
    ".avi": "Videos", ".wmv": "Videos", ".flv": "Videos",
    ".webm": "Videos",
    # Music / Audio
    ".mp3": "Music", ".wav": "Music", ".flac": "Music",
    ".aac": "Music", ".ogg": "Music", ".m4a": "Music", ".wma": "Music",
    # Archives
    ".zip": "Archives", ".tar": "Archives", ".gz": "Archives",
    ".bz2": "Archives", ".xz": "Archives", ".rar": "Archives",
    ".7z": "Archives",
    # Calendar
    ".ics": "Calendar Events",
    # Certificates / Security
    ".pem": "Certificates", ".crt": "Certificates", ".crl": "Certificates",
    ".cer": "Certificates", ".key": "Certificates", ".p12": "Certificates",
    ".pfx": "Certificates", ".pkpass": "Certificates",
    # Presentations
    ".pptx": "Presentations", ".ppt": "Presentations",
    ".odp": "Presentations", ".key": "Presentations",
    # Word-processor formats (specific name — "Documents" is too generic)
    ".doc": "Word Documents", ".docx": "Word Documents",
    ".odt": "Word Documents", ".rtf": "Word Documents",
    # PDFs (fallback when content-based classification fails)
    ".pdf": "PDF Documents",
    # Web pages
    ".html": "Web Pages", ".htm": "Web Pages",
}


# ── Filename pattern recognition ──────────────────────────────────────────
# Detects well-known naming conventions from OSes, apps, and document types
# to give the LLM richer signal — especially for binary files like PDFs
# where we can't read content.

_FILENAME_PATTERNS: list[tuple[re.Pattern[str], dict[str, str]]] = [
    # macOS screenshots: "Screenshot 2024-03-15 at 2.30.22 PM.png"
    (
        re.compile(r"^Screenshot \d{4}-\d{2}-\d{2} at .+\.\w+$"),
        {"source": "macos-screenshot"},
    ),
    # Linux screenshots: "Screenshot From 2025-07-10 15-39-27.png"
    (
        re.compile(r"^Screenshot From \d{4}-\d{2}-\d{2} .+\.\w+$"),
        {"source": "linux-screenshot"},
    ),
    # WhatsApp images: "WhatsApp Image 2024-12-09 at 19.42.33.jpeg"
    (
        re.compile(r"^WhatsApp (?:Image|Video) \d{4}-\d{2}-\d{2} at .+\.\w+$"),
        {"source": "whatsapp"},
    ),
    # iOS camera: "IMG_1234.HEIC" or "IMG_1234.jpg"
    (
        re.compile(r"^IMG_\d{4,5}\.\w+$", re.IGNORECASE),
        {"source": "ios-camera"},
    ),
    # Android camera: "IMG_20240315_142233.jpg"
    (
        re.compile(r"^IMG_\d{8}_\d{6}\.\w+$"),
        {"source": "android-camera"},
    ),
    # Canon/Nikon RAW: "20240315_142233.CR3" or "_DSC1234.NEF"
    (
        re.compile(r"^(?:\d{8}_\d{6}|_DSC\d{4,5})\.\w+$"),
        {"source": "dslr-camera"},
    ),
    # Zoom recordings: "GMT20250120-143022_Recording.m4a"
    (
        re.compile(r"^GMT\d{8}-\d{6}_Recording\.\w+$"),
        {"source": "zoom-recording"},
    ),
    # macOS installers: "AppName-1.2.3.dmg" or "AppName.dmg"
    (
        re.compile(r"^.+\.dmg$", re.IGNORECASE),
        {"type": "macos-installer"},
    ),
    # Browser duplicate downloads: "filename (1).ext", "filename (2).ext"
    (
        re.compile(r"^(.+?) \((\d+)\)(\.\w+)$"),
        {"download_duplicate": "true"},
    ),
    # Scanner-generated long names
    (
        re.compile(r"^Scan_.*_\d{8}_\d{6}.*\.\w+$"),
        {"source": "document-scanner"},
    ),
    # Payslips
    (
        re.compile(r"(?i)^payslip[_\s-]"),
        {"doc_type": "payslip"},
    ),
    # Rent receipts
    (
        re.compile(r"(?i)rent[_\s-]?rec(?:ei|ie)pt"),
        {"doc_type": "rent-receipt"},
    ),
    # Indian tax Form 16
    (
        re.compile(r"(?i)form\s*16"),
        {"doc_type": "tax-form-16"},
    ),
    # Aadhaar
    (
        re.compile(r"(?i)(?:e?aadhaar|EAadhaar)"),
        {"doc_type": "aadhaar-id"},
    ),
    # PAN Card
    (
        re.compile(r"(?i)^PAN[_\s-]?Card"),
        {"doc_type": "pan-card"},
    ),
    # Telecom / internet bills (common Indian providers)
    (
        re.compile(r"(?i)(?:VIL|Jio|Airtel|BSNL|ACT)[_\s-].*(?:bill|invoice)", re.IGNORECASE),
        {"doc_type": "telecom-bill"},
    ),
    # Credit card statements
    (
        re.compile(r"(?i)(?:CC|credit[_\s-]?card)[_\s-]?statement"),
        {"doc_type": "credit-card-statement"},
    ),
    # Mutual fund / investment statements
    (
        re.compile(r"(?i)(?:mutual\s*fund|CAS)[_\s-]?statement"),
        {"doc_type": "investment-statement"},
    ),
    # Insurance policies
    (
        re.compile(r"(?i)insurance[_\s-]?polic"),
        {"doc_type": "insurance-policy"},
    ),
    # Generic invoice / bill patterns
    (
        re.compile(r"(?i)(?:^|\b)invoice"),
        {"doc_type": "invoice"},
    ),
    (
        re.compile(r"(?i)(?:^|\b)receipt"),
        {"doc_type": "receipt"},
    ),
]


def detect_filename_hints(name: str) -> dict[str, str]:
    """Extract structured hints from well-known filename patterns.

    Returns a dict of hint key-value pairs, empty if no patterns match.
    Only the first matching pattern is used to avoid conflicting hints.
    """
    for pattern, hints in _FILENAME_PATTERNS:
        m = pattern.search(name)
        if m:
            result = dict(hints)
            # For duplicate downloads, also extract the base filename
            if "download_duplicate" in result:
                result["duplicate_of"] = m.group(1) + m.group(3)
            return result
    return {}


def find_duplicate_groups(
    profiles: list[FileProfile],
) -> dict[str, list[FileProfile]]:
    """Identify browser-duplicate groups: file.pdf, file (1).pdf, file (2).pdf.

    Returns a mapping of base filename → list of duplicate profiles.
    Only includes groups with at least one duplicate (2+ files).
    """
    dup_pattern = re.compile(r"^(.+?) \(\d+\)(\.\w+)$")
    # Map base name → list of profiles (base + duplicates)
    groups: dict[str, list[FileProfile]] = {}

    base_lookup: dict[str, FileProfile] = {}
    for p in profiles:
        base_lookup[p.name] = p

    for p in profiles:
        m = dup_pattern.match(p.name)
        if m:
            base_name = m.group(1) + m.group(2)
            if base_name not in groups:
                groups[base_name] = []
                # Include the base file if it exists
                if base_name in base_lookup:
                    groups[base_name].append(base_lookup[base_name])
            groups[base_name].append(p)

    # Only return groups that actually have duplicates
    return {k: v for k, v in groups.items() if len(v) > 1}


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
    hints: dict[str, str] = field(default_factory=dict)

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
        if self.hints:
            d["hints"] = self.hints
        return d


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
    library is missing or extraction fails.  This keeps PDF preview as a
    best-effort enhancement — the scanner works without it.
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

        # Detect filename hints (screenshots, duplicates, doc types, etc.)
        hints = detect_filename_hints(name)

        # Extension-based suggested category (deterministic pre-classification)
        ext_lower = entry.suffix.lower()
        suggested = _EXTENSION_CATEGORIES.get(ext_lower)
        if suggested and "suggested_category" not in hints:
            hints["suggested_category"] = suggested

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
            hints=hints,
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
