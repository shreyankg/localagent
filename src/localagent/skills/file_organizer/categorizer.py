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
            # Unchanged — reuse cached category
            cached[p.name] = entry["category"]
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
- Pay attention to the "hints" field — it tells you the file's likely source \
(e.g. screenshot, whatsapp, camera) or document type (e.g. payslip, invoice, \
tax form). Use these to make better categorization decisions.
- If a file's purpose is unclear, use a general category like "Miscellaneous".

Respond with ONLY valid JSON in this exact format:
{
  "taxonomy": {
    "Category Name": "Brief description of what goes here",
    ...
  },
  "assignments": {
    "filename.ext": "Category Name",
    ...
  }
}
"""

_WARM_SYSTEM = """\
You are a file organization assistant. You have an existing taxonomy of \
categories from previous runs. New files need to be categorized.

Your job:
1. Review the existing taxonomy and the new files.
2. Assign each new file to the most appropriate existing category.
3. If a file doesn't fit any existing category well, you may propose a NEW \
category — but only if it's genuinely distinct.
4. Categories marked with "user_locked: true" must NOT be renamed or removed.

Respond with ONLY valid JSON in this exact format:
{
  "taxonomy": {
    "Existing Category": "description (keep as-is or update)",
    "New Category If Needed": "description",
    ...
  },
  "assignments": {
    "new_filename.ext": "Category Name",
    ...
  }
}
"""


def _build_file_inventory(profiles: list[FileProfile]) -> str:
    """Format file profiles into a text block for the LLM prompt."""
    summaries = [p.to_summary() for p in profiles]
    return json.dumps(summaries, indent=2)


def _build_cold_messages(profiles: list[FileProfile]) -> list[dict[str, str]]:
    """Build messages for a first-run (no existing taxonomy)."""
    inventory = _build_file_inventory(profiles)
    return [
        {"role": "system", "content": _COLD_START_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Here are {len(profiles)} files to organize:\n\n{inventory}"
            ),
        },
    ]


def _build_warm_messages(
    profiles: list[FileProfile],
    taxonomy: dict[str, Any],
) -> list[dict[str, str]]:
    """Build messages for a subsequent run with an existing taxonomy."""
    inventory = _build_file_inventory(profiles)
    taxonomy_str = yaml.dump(
        taxonomy.get("taxonomy", taxonomy),
        default_flow_style=False,
    )
    return [
        {"role": "system", "content": _WARM_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Existing taxonomy:\n```yaml\n{taxonomy_str}```\n\n"
                f"New files to categorize ({len(profiles)}):\n\n{inventory}"
            ),
        },
    ]


# ── Batching ────────────────────────────────────────────────────────────────

_BATCH_SIZE = 80  # max files per LLM call


def _batch_profiles(
    profiles: list[FileProfile],
) -> list[list[FileProfile]]:
    """Split profiles into batches, grouping similar files together.

    Files are sorted by extension before batching so that the LLM sees
    related files in the same context window.  This produces finer-grained
    categories — e.g. when 14 PDFs (payslips, bills, tax docs) land in the
    same batch, the model distinguishes them better than if they were
    scattered across batches mixed with images and code.
    """
    if len(profiles) <= _BATCH_SIZE:
        return [profiles]

    # Sort by extension so similar files are grouped in the same batch
    sorted_profiles = sorted(profiles, key=lambda p: (p.extension, p.name))

    return [
        sorted_profiles[i : i + _BATCH_SIZE]
        for i in range(0, len(sorted_profiles), _BATCH_SIZE)
    ]


# ── Validation ──────────────────────────────────────────────────────────────


def _validate_response(
    result: dict[str, Any],
    known_files: set[str],
) -> dict[str, Any]:
    """Validate LLM output: drop hallucinated filenames, warn on mismatches."""
    taxonomy = result.get("taxonomy", {})
    assignments = result.get("assignments", {})

    # Drop assignments for files that don't exist
    valid_assignments: dict[str, str] = {}
    for filename, category in assignments.items():
        if filename not in known_files:
            logger.warning("LLM hallucinated file: '%s' — skipping", filename)
            continue
        if category not in taxonomy:
            logger.warning(
                "File '%s' assigned to unknown category '%s' — skipping",
                filename,
                category,
            )
            continue
        valid_assignments[filename] = category

    return {
        "taxonomy": taxonomy,
        "assignments": valid_assignments,
    }


def _merge_taxonomy(
    existing: dict[str, Any],
    new_result: dict[str, Any],
) -> dict[str, Any]:
    """Merge new LLM results into the existing taxonomy.

    User-locked categories are preserved unconditionally.
    """
    existing_tax = dict(existing.get("taxonomy", {}))
    new_tax = new_result.get("taxonomy", {})

    # Preserve user-locked categories
    for cat_name, cat_info in existing_tax.items():
        if isinstance(cat_info, dict) and cat_info.get("user_locked"):
            new_tax[cat_name] = cat_info

    # Merge: new taxonomy wins for non-locked categories
    merged = {**existing_tax, **new_tax}

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

    # Only send unique files to the LLM
    unique_profiles = [p for p in profiles if p.name not in dup_filenames]
    logger.info(
        "Sending %d unique files to LLM (%d duplicates will auto-assign)",
        len(unique_profiles),
        len(dup_filenames),
    )

    known_files = {p.name for p in unique_profiles}
    existing_taxonomy = load_taxonomy(state_dir)
    is_cold = existing_taxonomy is None

    all_assignments: dict[str, str] = {}
    latest_taxonomy: dict[str, Any] = existing_taxonomy or {}

    batches = _batch_profiles(unique_profiles)

    for i, batch in enumerate(batches):
        logger.info(
            "Categorizing batch %d/%d (%d files)",
            i + 1,
            len(batches),
            len(batch),
        )

        if is_cold and i == 0:
            messages = _build_cold_messages(batch)
        else:
            messages = _build_warm_messages(batch, latest_taxonomy)

        result = engine.generate_json(messages, max_tokens=4096)
        result = _validate_response(result, known_files)

        all_assignments.update(result.get("assignments", {}))

        if is_cold and i == 0:
            latest_taxonomy = result
        else:
            latest_taxonomy = _merge_taxonomy(latest_taxonomy, result)

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
