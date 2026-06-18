"""Adaptive LLM-powered file categorization.

No pre-defined categories.  The system evolves a taxonomy by inspecting
actual file types, names, and content — then persists it for future runs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from localagent.core.engine import Engine
from localagent.skills.file_organizer.scanner import FileProfile, find_duplicate_groups

logger = logging.getLogger(__name__)

TAXONOMY_FILE = "taxonomy.yaml"
MANIFEST_FILE = "last_run_manifest.yaml"

# ── Taxonomy I/O ────────────────────────────────────────────────────────────


def load_taxonomy(state_dir: Path) -> dict[str, Any] | None:
    """Load the learned taxonomy from disk, or return None on first run."""
    path = state_dir / TAXONOMY_FILE
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f) or None


def save_taxonomy(state_dir: Path, taxonomy: dict[str, Any]) -> Path:
    """Persist the taxonomy to disk."""
    path = state_dir / TAXONOMY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(taxonomy, f, default_flow_style=False, sort_keys=False)
    logger.info("Saved taxonomy to %s", path)
    return path


# ── Incremental run manifest ───────────────────────────────────────────────
# Tracks which files were categorized on the last run so we can skip
# unchanged files and only send new/modified ones to the LLM.


def _load_manifest(state_dir: Path) -> dict[str, dict[str, Any]]:
    """Load the last-run manifest: ``{filename: {category, mtime, size}}``."""
    path = state_dir / MANIFEST_FILE
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _save_manifest(
    state_dir: Path,
    assignments: dict[str, str],
    profiles: list[FileProfile],
) -> None:
    """Save manifest of categorized files for incremental runs."""
    manifest: dict[str, dict[str, Any]] = {}
    profiles_by_name = {p.name: p for p in profiles}
    for filename, category in assignments.items():
        p = profiles_by_name.get(filename)
        if p:
            manifest[filename] = {
                "category": category,
                "mtime": p.modified.isoformat() if p.modified else None,
                "size": p.size_bytes,
            }
    path = state_dir / MANIFEST_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
    logger.info("Saved manifest (%d entries) to %s", len(manifest), path)


def _partition_incremental(
    profiles: list[FileProfile],
    state_dir: Path,
) -> tuple[list[FileProfile], dict[str, str]]:
    """Separate profiles into new/changed files vs. already-categorized.

    Returns (profiles_to_categorize, cached_assignments).
    Already-categorized files are those whose name, mtime, and size match
    the last-run manifest.
    """
    manifest = _load_manifest(state_dir)
    if not manifest:
        return profiles, {}

    to_categorize: list[FileProfile] = []
    cached: dict[str, str] = {}

    for p in profiles:
        entry = manifest.get(p.name)
        if entry is None:
            # New file — needs categorization
            to_categorize.append(p)
            continue

        # Check if file has changed since last run
        prev_mtime = entry.get("mtime")
        curr_mtime = p.modified.isoformat() if p.modified else None
        prev_size = entry.get("size")

        if prev_mtime == curr_mtime and prev_size == p.size_bytes:
            prev_category = entry["category"]
            if prev_category == "Other Files":
                # Previously fell through to generic fallback — give the
                # LLM another chance with content previews on warm runs.
                to_categorize.append(p)
            else:
                # Unchanged — reuse cached category
                cached[p.name] = prev_category
        else:
            # Modified — re-categorize
            to_categorize.append(p)

    logger.info(
        "Incremental: %d cached, %d new/changed",
        len(cached),
        len(to_categorize),
    )
    return to_categorize, cached


# ── Prompt construction ─────────────────────────────────────────────────────

_COLD_START_SYSTEM = """\
You are a file organization assistant. You will be given a list of files with \
their names, extensions, MIME types, sizes, and (when available) content previews \
and hints about the file's origin or document type.

Your job:
1. Analyze the files and infer meaningful categories based on their actual \
content, purpose, and type — NOT just their extension.
2. Propose a taxonomy of folder categories that makes sense for THIS specific \
collection of files. Use clear, concise category names (e.g. "Receipts & Invoices", \
"Code Projects", "Research Papers", "Screenshots", "Music").
3. Assign every file to exactly one category.

Guidelines:
- Aim for 5–15 categories. Don't be too granular or too broad.
- Group related files (e.g. a .docx and its companion .pdf should go together).
- Think about the semantic purpose: a Python script about ML goes in a \
different category than a shell utility script.
- Pay attention to file extensions — they are strong signals for categorization:
  .dmg/.pkg → Installers, .eml/.msg → Emails, .csv/.xlsx → Spreadsheets, \
  .py/.js/.ts → Code, .jpg/.png/.heic → Photos, .mp4/.mov → Videos, \
  .mp3/.wav/.flac → Music, .zip/.tar.gz/.rar → Archives, \
  .pdf → (classify by content: Receipts, Tax Documents, Contracts, etc.), \
  .doc/.docx → (classify by content), .ics → Calendar.
  Use extensions as a starting point, then refine with content and context.
- Pay attention to the "hints" field — it tells you the file's likely source \
(e.g. screenshot, whatsapp, camera) or document type (e.g. payslip, invoice, \
tax form). Use these to make better categorization decisions.
- When a file has a "suggested_category" hint, USE that category as-is unless \
the filename or content preview strongly suggests a more specific one. The \
suggested_category is derived from the file extension and is reliable.
- NEVER use generic catch-all categories like "Documents", "Miscellaneous", \
"Other", or "General". Every file must go into a specific, descriptive \
category (e.g. "Resumes", "Tax Documents", "Receipts & Invoices", \
"Contracts", "Travel Documents", "Pay Statements").
- NEVER use a filename as a category name. Categories must be descriptive \
labels (e.g. "Photos", "Receipts & Invoices"), never specific filenames \
(e.g. "IMG_0019.HEIC", "report.pdf").

Each file has a short ID (like "f1", "f2").  Use these IDs — not filenames — \
in your assignments.

IMPORTANT: Every category name you use in "assignments" MUST also appear in \
"taxonomy" with a description. Do not assign a file to a category that is \
not listed in the taxonomy.

Respond with ONLY valid JSON in this exact format:
{
  "taxonomy": {
    "Category Name": "Brief description of what goes here",
    ...
  },
  "assignments": {
    "f1": "Category Name",
    "f2": "Category Name",
    ...
  }
}
"""

_WARM_SYSTEM = """\
You are a file organization assistant. You have an existing taxonomy of \
categories from previous runs. New files need to be categorized.

Your job:
1. Review the existing taxonomy and the new files.
2. Assign each new file to the MOST APPROPRIATE EXISTING category. \
You MUST reuse existing categories wherever possible.
3. Do NOT create renamed or rephrased versions of existing categories. \
For example, if "Photos" exists, do NOT create "Images" or \
"Photo Files" — use "Photos".
4. Only create a genuinely NEW category if the file does not fit ANY \
existing category at all — this should be rare.

IMPORTANT rules:
- Pay attention to file extensions — they are strong signals for categorization:
  .dmg/.pkg → Installers, .eml/.msg → Emails, .csv/.xlsx → Spreadsheets, \
  .py/.js/.ts → Code, .jpg/.png/.heic → Photos, .mp4/.mov → Videos, \
  .mp3/.wav/.flac → Music, .zip/.tar.gz/.rar → Archives, \
  .pdf → (classify by content: Receipts, Tax Documents, Contracts, etc.), \
  .doc/.docx → (classify by content), .ics → Calendar.
  Use extensions as a starting point, then refine with content and context.
- Pay attention to the "hints" field — it tells you the file's likely source \
(e.g. screenshot, whatsapp, camera) or document type (e.g. payslip, invoice, \
tax form). Use these to make better categorization decisions.
- When a file has a "suggested_category" hint, USE that category as-is unless \
the filename or content preview strongly suggests a more specific one. The \
suggested_category is derived from the file extension and is reliable. \
If the suggested category matches an existing category, use the existing one.
- NEVER use generic catch-all categories like "Documents", "Miscellaneous", \
"Other", or "General". Every file must go into a specific, descriptive \
category (e.g. "Resumes", "Tax Documents", "Receipts & Invoices", \
"Contracts", "Travel Documents", "Pay Statements").
- NEVER use a filename as a category name. Categories must be descriptive \
labels (e.g. "Photos", "Receipts & Invoices"), never specific filenames \
(e.g. "IMG_0019.HEIC", "report.pdf").
- Category names must be short, human-readable folder names \
(e.g. "Receipts & Invoices", "Code Projects", "Screenshots").
- Include ALL existing categories in the taxonomy, even if no new files \
are assigned to them.
- The taxonomy in your response must list every existing category plus \
any new ones you add.

Each file has a short ID (like "f1", "f2").  Use these IDs — not filenames — \
in your assignments.

IMPORTANT: Every category name you use in "assignments" MUST also appear in \
"taxonomy". Do not assign a file to a category that is not listed in the taxonomy.

Respond with ONLY valid JSON in this exact format:
{
  "taxonomy": {
    "Existing Category": "description (keep as-is)",
    "...": "(add only if truly needed)",
    ...
  },
  "assignments": {
    "f1": "Category Name",
    "f2": "Category Name",
    ...
  }
}
"""


def _build_file_inventory(
    profiles: list[FileProfile],
    *,
    include_previews: bool = True,
) -> tuple[str, dict[str, str]]:
    """Format file profiles into a text block for the LLM prompt.

    Each file gets a short ID (``f1``, ``f2``, …) to avoid asking the
    LLM to reproduce long filenames verbatim.

    When *include_previews* is False, content previews are stripped from
    the summaries to reduce token usage (useful for cold-start runs
    where batch sizes are large).

    Returns ``(inventory_text, id_to_filename)`` where *id_to_filename*
    maps the short IDs back to real filenames.
    """
    id_to_filename: dict[str, str] = {}
    summaries: list[dict[str, Any]] = []
    for idx, p in enumerate(profiles):
        fid = f"f{idx + 1}"
        id_to_filename[fid] = p.name
        summary = p.to_summary()
        if not include_previews:
            summary.pop("content_preview", None)
        summary["id"] = fid
        summaries.append(summary)
    return json.dumps(summaries, indent=2), id_to_filename


def _build_cold_messages(
    profiles: list[FileProfile],
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Build messages for a first-run (no existing taxonomy).

    Content previews are excluded to allow larger batch sizes.

    Returns ``(messages, id_to_filename)``.
    """
    inventory, id_map = _build_file_inventory(profiles, include_previews=False)
    messages = [
        {"role": "system", "content": _COLD_START_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Here are {len(profiles)} files to organize:\n\n{inventory}"
            ),
        },
    ]
    return messages, id_map


def _format_taxonomy_list(taxonomy: dict[str, Any]) -> str:
    """Format the taxonomy as a simple numbered list for the LLM prompt.

    Using a plain list instead of YAML prevents the model from
    reproducing YAML structural artifacts (like ``user_locked: true``)
    as category names.
    """
    tax_dict = taxonomy.get("taxonomy", taxonomy)
    lines: list[str] = []
    for i, (name, desc) in enumerate(tax_dict.items(), 1):
        # Only include the description string, not nested dicts
        if isinstance(desc, dict):
            desc_text = desc.get("description", str(desc))
        else:
            desc_text = str(desc)
        lines.append(f"{i}. {name} — {desc_text}")
    return "\n".join(lines)


def _build_warm_messages(
    profiles: list[FileProfile],
    taxonomy: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Build messages for a subsequent run with an existing taxonomy.

    Returns ``(messages, id_to_filename)``.
    """
    inventory, id_map = _build_file_inventory(profiles)
    taxonomy_str = _format_taxonomy_list(taxonomy)
    num_categories = len(taxonomy.get("taxonomy", taxonomy))

    cap_warning = ""
    if num_categories >= _MAX_TAXONOMY_SIZE:
        cap_warning = (
            f"\n\nIMPORTANT: There are already {num_categories} categories. "
            "Do NOT add new categories. Assign every file to the most "
            "appropriate existing category."
        )

    messages = [
        {"role": "system", "content": _WARM_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Existing categories ({num_categories}):\n{taxonomy_str}"
                f"{cap_warning}\n\n"
                f"New files to categorize ({len(profiles)}):\n\n{inventory}"
            ),
        },
    ]
    return messages, id_map


# ── Batching ────────────────────────────────────────────────────────────────

_COLD_BATCH_SIZE = 80   # cold start: no previews → more files per batch
_WARM_BATCH_SIZE = 25   # warm runs: with previews → fewer files per batch


def _batch_profiles(
    profiles: list[FileProfile],
    batch_size: int,
) -> list[list[FileProfile]]:
    """Split profiles into batches, grouping similar files together.

    Files are sorted by extension before batching so that the LLM sees
    related files in the same context window.  This produces finer-grained
    categories — e.g. when 14 PDFs (payslips, bills, tax docs) land in the
    same batch, the model distinguishes them better than if they were
    scattered across batches mixed with images and code.
    """
    if len(profiles) <= batch_size:
        return [profiles]

    # Sort by extension so similar files are grouped in the same batch
    sorted_profiles = sorted(profiles, key=lambda p: (p.extension, p.name))

    return [
        sorted_profiles[i : i + batch_size]
        for i in range(0, len(sorted_profiles), batch_size)
    ]


# ── Validation ──────────────────────────────────────────────────────────────

_YAML_ARTIFACTS = frozenset({
    "true", "false", "null", "yes", "no", "user_locked: true",
    "user_locked: false",
    # Prompt placeholders the model sometimes reproduces verbatim
    "new category if needed",
    "add only if truly needed",
    "existing category",
    "category name",
    # Overly generic catch-all categories that prevent real classification
    "documents",
    "pdf documents",
    "miscellaneous",
    "misc",
    "other",
    "general",
    "uncategorized",
})

_MAX_TAXONOMY_SIZE = 15


def _is_bad_category_name(
    name: str,
    known_filenames: set[str] | None = None,
) -> bool:
    """Return True if a category name looks like a filename or YAML artifact.

    Bad names include:
    - Exact matches of known filenames (e.g. ``report.pdf``)
    - Filename stems that are long enough to be specific (>= 6 chars),
      suggesting the LLM echoed a filename without its extension
    - YAML artifacts (e.g. ``true``, ``user_locked: true``)
    - Generic catch-all categories (e.g. ``Documents``, ``Miscellaneous``)
    - Very short or empty strings
    - Too-long strings (> 40 chars) — real categories are short labels
    - Strings with URL/path characters (``?``, ``=``, ``&``, ``/``, ``\\``)
    """
    stripped = name.strip()
    if len(stripped) < 2:
        return True
    # Too long to be a category name — likely a filename or URL
    if len(stripped) > 40:
        return True
    # URL/path characters don't belong in category names
    # Note: '&' is excluded — it appears in valid categories like
    # "Receipts & Invoices".
    if any(c in stripped for c in "?=/\\"):
        return True
    if stripped.lower() in _YAML_ARTIFACTS:
        return True
    if known_filenames:
        lower = stripped.lower()
        for filename in known_filenames:
            # Exact filename match (e.g. "report.pdf" as category)
            if lower == filename.lower():
                return True
            # Category matches filename stem (case-insensitive), and
            # stem is long enough to be a specific name rather than a
            # generic concept.  Short stems like "Code", "Data" are
            # legitimate category names; "Aadhaar", "VFS GLOBAL" are
            # too specific.
            stem = Path(filename).stem
            if len(stem) >= 6 and lower == stem.lower():
                return True
            # Category is a long substring of a filename — catches
            # partial filename echoes like "Logo-Red_Hat" appearing
            # inside "Logo-Red_Hat-Engineering.eps".  The length
            # threshold avoids false positives on short words.
            if len(stripped) >= 6 and lower in filename.lower():
                return True
            # Reverse: filename (with extension) is a substring of the
            # category — catches the LLM wrapping a filename in extra
            # text.  Only applies to filenames with extensions to avoid
            # false positives on bare words like "Code" or "Research".
            if (
                len(filename) >= 5
                and "." in filename
                and filename.lower() in lower
            ):
                return True
    return False


def _normalize_categories(
    taxonomy: dict[str, Any],
    assignments: dict[str, str],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Merge near-duplicate category names in the taxonomy.

    Detects duplicates via:
    - Case-insensitive match (``Code`` vs ``code``)
    - One name is a substring of another (``Documents`` vs ``Text Documents``)

    The first-seen (canonical) name wins.  Assignments are updated to
    point to the canonical name.

    Returns ``(cleaned_taxonomy, updated_assignments)``.
    """
    if not taxonomy:
        return taxonomy, assignments

    # Build canonical mapping: lowercased → first-seen original name
    canonical_map: dict[str, str] = {}  # lower_name → canonical_name
    canonical_names: list[str] = []  # ordered list of canonical names

    for name in taxonomy:
        lower = name.strip().lower()
        if lower in canonical_map:
            # Exact case-insensitive duplicate
            logger.info(
                "Merging duplicate category '%s' → '%s'",
                name,
                canonical_map[lower],
            )
            continue
        # Check substring containment with existing canonical names
        merged = False
        for existing_lower, existing_name in canonical_map.items():
            if lower in existing_lower or existing_lower in lower:
                logger.info(
                    "Merging similar category '%s' → '%s'",
                    name,
                    existing_name,
                )
                canonical_map[lower] = existing_name
                merged = True
                break
        if not merged:
            canonical_map[lower] = name
            canonical_names.append(name)

    # Rebuild taxonomy with only canonical names
    cleaned_taxonomy: dict[str, Any] = {}
    for name in canonical_names:
        cleaned_taxonomy[name] = taxonomy.get(name, "")

    # Build reverse map: any name → canonical name
    name_to_canonical: dict[str, str] = {}
    for name in taxonomy:
        lower = name.strip().lower()
        name_to_canonical[name] = canonical_map.get(lower, name)

    # Update assignments to use canonical names
    updated_assignments: dict[str, str] = {}
    for filename, category in assignments.items():
        updated_assignments[filename] = name_to_canonical.get(category, category)

    if len(cleaned_taxonomy) < len(taxonomy):
        logger.info(
            "Normalized taxonomy: %d → %d categories",
            len(taxonomy),
            len(cleaned_taxonomy),
        )

    return cleaned_taxonomy, updated_assignments


def _validate_response(
    result: dict[str, Any],
    id_to_filename: dict[str, str],
    all_filenames: set[str] | None = None,
    profiles_by_name: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate LLM output and map short IDs back to real filenames.

    - Unknown IDs are dropped (the model hallucinated an ID).
    - Unknown categories are auto-added to the taxonomy rather than
      dropping the file, so no file is silently lost.
    - Bad category names (filenames, YAML artifacts, generic catch-alls)
      are stripped from the taxonomy.  Files from bad categories are
      reassigned to their ``suggested_category`` hint if available, or
      to a category derived from their file extension.

    ``all_filenames`` is the full set of filenames being processed
    (across all batches) so we can detect the LLM echoing back a
    filename as a category name.

    ``profiles_by_name`` maps filenames to their FileProfile objects
    so we can look up ``suggested_category`` hints for fallback.
    """
    taxonomy = result.get("taxonomy", {})
    assignments = result.get("assignments", {})
    profiles_by_name = profiles_by_name or {}

    # Build filename set: current batch + any extras passed in
    filenames = set(id_to_filename.values())
    if all_filenames:
        filenames |= all_filenames

    # Strip bad taxonomy entries
    bad_names: set[str] = set()
    for cat_name in list(taxonomy):
        if _is_bad_category_name(cat_name, filenames):
            logger.warning(
                "Removing bad category name: '%s'", cat_name,
            )
            bad_names.add(cat_name)
            del taxonomy[cat_name]

    valid_assignments: dict[str, str] = {}
    for key, category in assignments.items():
        # Map ID → filename
        filename = id_to_filename.get(key)
        if filename is None:
            logger.warning("LLM returned unknown ID: '%s' — skipping", key)
            continue
        # Reassign files from bad categories using hints or extension
        if category in bad_names or _is_bad_category_name(category, filenames):
            # Try suggested_category from hints first
            profile = profiles_by_name.get(filename)
            suggested = None
            if profile and hasattr(profile, "hints"):
                suggested = profile.hints.get("suggested_category")
            if suggested and not _is_bad_category_name(suggested):
                logger.info(
                    "Reassigning '%s' from bad category '%s' → '%s' (from hint)",
                    filename, category, suggested,
                )
                category = suggested
            else:
                # Derive a category from file extension
                ext = Path(filename).suffix.lower()
                from localagent.skills.file_organizer.scanner import (
                    _EXTENSION_CATEGORIES,
                )
                fallback = _EXTENSION_CATEGORIES.get(ext, "Other Files")
                logger.info(
                    "Reassigning '%s' from bad category '%s' → '%s' (from extension)",
                    filename, category, fallback,
                )
                category = fallback
            if category not in taxonomy:
                taxonomy[category] = f"Files categorized as {category}"
        if category not in taxonomy:
            # Auto-add the category instead of dropping the file
            logger.info(
                "Auto-adding category '%s' (used by '%s' but not in taxonomy)",
                category,
                filename,
            )
            taxonomy[category] = f"Files categorized as {category}"
        valid_assignments[filename] = category

    return {
        "taxonomy": taxonomy,
        "assignments": valid_assignments,
    }


def _merge_taxonomy(
    existing: dict[str, Any],
    new_result: dict[str, Any],
    all_filenames: set[str] | None = None,
) -> dict[str, Any]:
    """Merge new LLM results into the existing taxonomy.

    User-locked categories are preserved unconditionally.
    Bad category names are stripped from the merged result.
    """
    existing_tax = dict(existing.get("taxonomy", {}))
    new_tax = new_result.get("taxonomy", {})

    # Preserve user-locked categories
    for cat_name, cat_info in existing_tax.items():
        if isinstance(cat_info, dict) and cat_info.get("user_locked"):
            new_tax[cat_name] = cat_info

    # Merge: new taxonomy wins for non-locked categories
    merged = {**existing_tax, **new_tax}

    # Strip any bad category names that slipped through
    for cat_name in list(merged):
        if _is_bad_category_name(cat_name, all_filenames):
            logger.warning("Stripping bad category from taxonomy: '%s'", cat_name)
            del merged[cat_name]

    return {"taxonomy": merged}


# ── Main entry point ────────────────────────────────────────────────────────


def categorize(
    engine: Engine,
    profiles: list[FileProfile],
    state_dir: Path,
) -> dict[str, Any]:
    """Categorize files using the LLM, evolving the taxonomy over time.

    Duplicate downloads (``file.pdf``, ``file (1).pdf``) are detected
    before the LLM call.  Only the base file is sent for categorization
    and duplicates are auto-assigned to the same category afterward.

    Returns ``{"taxonomy": {...}, "assignments": {"file": "category", ...}}``.
    """
    if not profiles:
        return {"taxonomy": {}, "assignments": {}}

    # ── Incremental: skip unchanged files from previous run ──────────
    profiles_to_process, cached_assignments = _partition_incremental(
        profiles, state_dir,
    )

    if not profiles_to_process:
        logger.info("All files unchanged since last run — nothing to categorize")
        return {
            "taxonomy": (load_taxonomy(state_dir) or {}).get("taxonomy", {}),
            "assignments": cached_assignments,
        }

    # ── Detect and strip duplicate downloads ─────────────────────────
    dup_groups = find_duplicate_groups(profiles_to_process)
    dup_filenames: set[str] = set()
    for base_name, group in dup_groups.items():
        # Mark all non-base files as duplicates (skip them in LLM call)
        for p in group:
            if p.name != base_name:
                dup_filenames.add(p.name)
        logger.info(
            "Duplicate group for '%s': %d copies will auto-assign",
            base_name,
            len(group) - 1,
        )

    # Only send unique files to the LLM (filter from profiles_to_process,
    # not the full profiles list, to avoid re-sending cached files)
    unique_profiles = [
        p for p in profiles_to_process if p.name not in dup_filenames
    ]
    logger.info(
        "Sending %d unique files to LLM (%d duplicates will auto-assign)",
        len(unique_profiles),
        len(dup_filenames),
    )

    existing_taxonomy = load_taxonomy(state_dir)
    is_cold = existing_taxonomy is None

    # Collect all filenames so validation can detect filename-as-category
    all_known_filenames = {p.name for p in profiles}
    all_profiles_by_name = {p.name: p for p in profiles}

    all_assignments: dict[str, str] = {}
    latest_taxonomy: dict[str, Any] = existing_taxonomy or {}

    batch_size = _COLD_BATCH_SIZE if is_cold else _WARM_BATCH_SIZE
    batches = _batch_profiles(unique_profiles, batch_size)
    batch_errors = 0

    for i, batch in enumerate(batches):
        logger.info(
            "Categorizing batch %d/%d (%d files)",
            i + 1,
            len(batches),
            len(batch),
        )

        if is_cold and i == 0:
            messages, id_map = _build_cold_messages(batch)
        else:
            messages, id_map = _build_warm_messages(batch, latest_taxonomy)

        try:
            result = engine.generate_json(messages, max_tokens=4096)
        except ValueError as exc:
            batch_errors += 1
            logger.error(
                "Batch %d/%d failed: %s — skipping %d files",
                i + 1,
                len(batches),
                exc,
                len(batch),
            )
            continue

        result = _validate_response(
            result, id_map, all_known_filenames, all_profiles_by_name,
        )

        all_assignments.update(result.get("assignments", {}))

        if is_cold and i == 0:
            latest_taxonomy = result
        else:
            latest_taxonomy = _merge_taxonomy(
                latest_taxonomy, result, all_known_filenames,
            )

    if batch_errors:
        logger.warning(
            "%d/%d batches failed — %d files were not categorized",
            batch_errors,
            len(batches),
            sum(len(b) for b in batches) - len(all_assignments),
        )

    # ── Normalize near-duplicate categories across batches ───────────
    raw_taxonomy = latest_taxonomy.get("taxonomy", {})
    normalized_tax, all_assignments = _normalize_categories(
        raw_taxonomy, all_assignments,
    )
    latest_taxonomy = {"taxonomy": normalized_tax}

    # ── Auto-assign duplicates to the same category as their base ────
    for base_name, group in dup_groups.items():
        base_category = all_assignments.get(base_name)
        if base_category:
            for p in group:
                if p.name != base_name:
                    all_assignments[p.name] = base_category
                    logger.debug(
                        "Auto-assigned duplicate '%s' → '%s'",
                        p.name,
                        base_category,
                    )

    # Merge cached assignments from previous run
    all_assignments.update(cached_assignments)

    # Save evolved taxonomy
    save_taxonomy(state_dir, latest_taxonomy)

    # Save manifest for incremental runs
    _save_manifest(state_dir, all_assignments, profiles)

    return {
        "taxonomy": latest_taxonomy.get("taxonomy", {}),
        "assignments": all_assignments,
    }
