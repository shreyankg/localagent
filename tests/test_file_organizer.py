"""Tests for the file organizer skill — scanner, categorizer, mover."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from localagent.core.safefs import SafeFS
from localagent.core.skill import Action
from localagent.skills.file_organizer.scanner import (
    FileProfile,
    scan_all,
    scan_directory,
)
from localagent.skills.file_organizer.categorizer import (
    _is_bad_category_name,
    _normalize_categories,
    _validate_response,
    categorize,
    load_taxonomy,
    save_taxonomy,
)
from localagent.skills.file_organizer.mover import (
    MoveRecord,
    build_actions,
    execute_moves,
)


# ── Scanner tests ───────────────────────────────────────────────────────────


@pytest.fixture
def sample_dir(tmp_path):
    """Create a directory with various test files."""
    # Text files
    (tmp_path / "readme.md").write_text("# My Project\nA cool project")
    (tmp_path / "notes.txt").write_text("Remember to buy milk")
    (tmp_path / "script.py").write_text("import pandas as pd\ndf = pd.read_csv('data.csv')")

    # Binary-ish files (just by name)
    (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8\xff\xe0")
    (tmp_path / "archive.zip").write_bytes(b"PK\x03\x04")

    # Hidden file
    (tmp_path / ".hidden").write_text("hidden")

    # Excluded file
    (tmp_path / ".DS_Store").write_bytes(b"\x00\x00")

    # A subdirectory (should be skipped by scanner)
    subdir = tmp_path / "existing_folder"
    subdir.mkdir()
    (subdir / "nested.txt").write_text("nested")

    return tmp_path


class TestScanner:
    def test_scans_files_not_dirs(self, sample_dir):
        safefs = SafeFS(allowed_paths=[sample_dir], permissions={"read"})
        profiles = scan_directory(safefs, sample_dir)
        names = [p.name for p in profiles]

        # Files should be present
        assert "readme.md" in names
        assert "notes.txt" in names
        assert "script.py" in names
        assert "photo.jpg" in names
        assert "archive.zip" in names

        # Dirs and hidden/excluded files should not
        assert "existing_folder" not in names
        assert ".hidden" not in names
        assert ".DS_Store" not in names

    def test_content_preview_for_text_files(self, sample_dir):
        safefs = SafeFS(allowed_paths=[sample_dir], permissions={"read"})
        profiles = scan_directory(safefs, sample_dir, content_preview_bytes=100)
        by_name = {p.name: p for p in profiles}

        assert by_name["readme.md"].content_preview is not None
        assert "My Project" in by_name["readme.md"].content_preview
        assert by_name["script.py"].content_preview is not None
        assert "pandas" in by_name["script.py"].content_preview

    def test_no_content_preview_for_binary(self, sample_dir):
        safefs = SafeFS(allowed_paths=[sample_dir], permissions={"read"})
        profiles = scan_directory(safefs, sample_dir)
        by_name = {p.name: p for p in profiles}

        assert by_name["photo.jpg"].content_preview is None
        assert by_name["archive.zip"].content_preview is None

    def test_exclude_patterns(self, sample_dir):
        safefs = SafeFS(allowed_paths=[sample_dir], permissions={"read"})
        profiles = scan_directory(
            safefs, sample_dir, exclude_patterns=["*.zip", "*.jpg"]
        )
        names = [p.name for p in profiles]
        assert "archive.zip" not in names
        assert "photo.jpg" not in names

    def test_scan_all_multiple_dirs(self, tmp_path):
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()
        (dir1 / "a.txt").write_text("a")
        (dir2 / "b.txt").write_text("b")

        safefs = SafeFS(allowed_paths=[dir1, dir2], permissions={"read"})
        profiles = scan_all(safefs, [dir1, dir2])
        names = [p.name for p in profiles]
        assert "a.txt" in names
        assert "b.txt" in names

    def test_file_profile_to_summary(self, sample_dir):
        safefs = SafeFS(allowed_paths=[sample_dir], permissions={"read"})
        profiles = scan_directory(safefs, sample_dir)
        for p in profiles:
            summary = p.to_summary()
            assert "name" in summary
            assert "extension" in summary
            assert "mime_type" in summary
            assert "size" in summary


# ── Categorizer tests ──────────────────────────────────────────────────────


class TestValidateResponse:
    def test_drops_unknown_ids(self):
        id_map = {"f1": "real.txt"}
        result = {
            "taxonomy": {"Docs": "documents"},
            "assignments": {
                "f1": "Docs",
                "f99": "Docs",  # unknown ID
            },
        }
        validated = _validate_response(result, id_map)
        assert "real.txt" in validated["assignments"]
        assert len(validated["assignments"]) == 1

    def test_auto_adds_unknown_categories(self):
        id_map = {"f1": "file.txt"}
        result = {
            "taxonomy": {"Docs": "documents"},
            "assignments": {
                "f1": "NonexistentCategory",
            },
        }
        validated = _validate_response(result, id_map)
        # File should NOT be dropped; category auto-added instead
        assert "file.txt" in validated["assignments"]
        assert "NonexistentCategory" in validated["taxonomy"]

    def test_valid_assignments_kept(self):
        id_map = {"f1": "readme.md", "f2": "main.py"}
        result = {
            "taxonomy": {"Docs": "documents", "Code": "source code"},
            "assignments": {
                "f1": "Docs",
                "f2": "Code",
            },
        }
        validated = _validate_response(result, id_map)
        assert len(validated["assignments"]) == 2
        assert validated["assignments"]["readme.md"] == "Docs"
        assert validated["assignments"]["main.py"] == "Code"

    def test_bad_category_names_stripped(self):
        id_map = {"f1": "readme.md", "f2": "data.csv"}
        all_filenames = {"readme.md", "data.csv", "report.pdf"}
        result = {
            "taxonomy": {
                "Docs": "documents",
                "report.pdf": "a bad filename category",
                "true": "a YAML artifact",
            },
            "assignments": {
                "f1": "report.pdf",
                "f2": "true",
            },
        }
        validated = _validate_response(result, id_map, all_filenames)
        # Bad categories should be removed, files reassigned to Miscellaneous
        assert "report.pdf" not in validated["taxonomy"]
        assert "true" not in validated["taxonomy"]
        assert validated["assignments"]["readme.md"] == "Miscellaneous"
        assert validated["assignments"]["data.csv"] == "Miscellaneous"
        assert "Miscellaneous" in validated["taxonomy"]

    def test_filename_as_category_in_assignments_caught(self):
        """LLM returns a filename as category in assignments (not taxonomy)."""
        id_map = {"f1": "IMG_0019.HEIC", "f2": "notes.txt"}
        all_filenames = {"IMG_0019.HEIC", "notes.txt", "schema.graphqls"}
        result = {
            "taxonomy": {"Photos": "photos"},
            "assignments": {
                "f1": "IMG_0019.HEIC",  # filename echoed back as category
                "f2": "schema.graphqls",  # filename from another batch
            },
        }
        validated = _validate_response(result, id_map, all_filenames)
        assert validated["assignments"]["IMG_0019.HEIC"] == "Miscellaneous"
        assert validated["assignments"]["notes.txt"] == "Miscellaneous"


class TestBadCategoryName:
    def test_rejects_exact_filename_match(self):
        filenames = {"report.pdf", "IMG_2034.jpg", "script.py"}
        assert _is_bad_category_name("report.pdf", filenames)
        assert _is_bad_category_name("IMG_2034.jpg", filenames)
        assert _is_bad_category_name("script.py", filenames)

    def test_rejects_category_thats_substring_of_filename(self):
        filenames = {"Logo-Red_Hat-Engineering.eps"}
        # Category is a substring of a filename (truncated filename)
        assert _is_bad_category_name("Logo-Red_Hat-Engineering", filenames)
        assert _is_bad_category_name("Logo-Red_Hat", filenames)

    def test_does_not_reject_category_containing_filename(self):
        filenames = {"Code", "Research"}
        # Filename appears inside a valid category — should NOT be rejected
        assert not _is_bad_category_name("Code Projects", filenames)
        assert not _is_bad_category_name("Research Papers", filenames)

    def test_rejects_yaml_artifacts(self):
        assert _is_bad_category_name("true")
        assert _is_bad_category_name("false")
        assert _is_bad_category_name("user_locked: true")

    def test_rejects_short_strings(self):
        assert _is_bad_category_name("")
        assert _is_bad_category_name("a")

    def test_accepts_valid_categories(self):
        filenames = {"report.pdf", "photo.jpg"}
        assert not _is_bad_category_name("Documents", filenames)
        assert not _is_bad_category_name("Receipts & Invoices", filenames)
        assert not _is_bad_category_name("Code Projects", filenames)
        assert not _is_bad_category_name("Screenshots", filenames)


class TestNormalizeCategories:
    def test_merges_case_duplicates(self):
        taxonomy = {"Documents": "docs", "documents": "also docs"}
        assignments = {"a.txt": "Documents", "b.txt": "documents"}
        norm_tax, norm_assign = _normalize_categories(taxonomy, assignments)
        assert len(norm_tax) == 1
        assert "Documents" in norm_tax
        assert norm_assign["a.txt"] == "Documents"
        assert norm_assign["b.txt"] == "Documents"

    def test_merges_substring_duplicates(self):
        taxonomy = {
            "Documents": "text files",
            "Text Documents": "also text files",
        }
        assignments = {"a.txt": "Documents", "b.txt": "Text Documents"}
        norm_tax, norm_assign = _normalize_categories(taxonomy, assignments)
        assert len(norm_tax) == 1
        assert "Documents" in norm_tax
        assert norm_assign["b.txt"] == "Documents"

    def test_keeps_distinct_categories(self):
        taxonomy = {
            "Documents": "text files",
            "Code": "source code",
            "Screenshots": "screen captures",
        }
        assignments = {"a.txt": "Documents", "b.py": "Code"}
        norm_tax, norm_assign = _normalize_categories(taxonomy, assignments)
        assert len(norm_tax) == 3
        assert norm_assign == assignments

    def test_empty_taxonomy(self):
        norm_tax, norm_assign = _normalize_categories({}, {"a.txt": "X"})
        assert norm_tax == {}


class TestTaxonomyIO:
    def test_save_and_load(self, tmp_path):
        taxonomy = {
            "taxonomy": {
                "Documents": "Text files and docs",
                "Code": "Source code files",
            }
        }
        save_taxonomy(tmp_path, taxonomy)
        loaded = load_taxonomy(tmp_path)
        assert loaded == taxonomy

    def test_load_nonexistent_returns_none(self, tmp_path):
        assert load_taxonomy(tmp_path) is None


class TestCategorize:
    @patch("localagent.skills.file_organizer.categorizer.Engine")
    def test_cold_start_categorization(self, MockEngine, tmp_path):
        """Test first-run categorization with mocked LLM."""
        engine = MagicMock()
        # LLM returns short IDs (f1, f2) — the categorizer maps them back
        engine.generate_json.return_value = {
            "taxonomy": {
                "Documents": "Text documents",
                "Data Science": "ML and data files",
            },
            "assignments": {
                "f1": "Documents",
                "f2": "Data Science",
            },
        }

        profiles = [
            FileProfile(
                name="readme.md",
                path=tmp_path / "readme.md",
                extension=".md",
                mime_type="text/markdown",
                size_bytes=100,
                content_preview="# Readme",
            ),
            FileProfile(
                name="train.py",
                path=tmp_path / "train.py",
                extension=".py",
                mime_type="text/x-python",
                size_bytes=500,
                content_preview="import torch",
            ),
        ]

        result = categorize(engine, profiles, tmp_path)

        assert "taxonomy" in result
        assert "assignments" in result
        assert result["assignments"]["readme.md"] == "Documents"
        assert result["assignments"]["train.py"] == "Data Science"

        # Taxonomy should be saved
        assert load_taxonomy(tmp_path) is not None

    def test_empty_profiles_returns_empty(self, tmp_path):
        engine = MagicMock()
        result = categorize(engine, [], tmp_path)
        assert result["assignments"] == {}
        assert result["taxonomy"] == {}


# ── Mover tests ─────────────────────────────────────────────────────────────


class TestBuildActions:
    def test_builds_correct_actions(self, tmp_path):
        (tmp_path / "doc.pdf").write_bytes(b"pdf content")

        profile = FileProfile(
            name="doc.pdf",
            path=tmp_path / "doc.pdf",
            extension=".pdf",
            mime_type="application/pdf",
            size_bytes=100,
        )

        actions = build_actions(
            assignments={"doc.pdf": "Documents"},
            profiles_by_name={"doc.pdf": profile},
            watch_directories=[tmp_path],
        )

        assert len(actions) == 1
        assert actions[0].action_type == "move"
        assert "doc.pdf" in actions[0].source
        assert "Documents" in actions[0].destination


class TestExecuteMoves:
    def test_executes_moves_successfully(self, tmp_path):
        (tmp_path / "file.txt").write_text("content")

        safefs = SafeFS(allowed_paths=[tmp_path], permissions={"read", "move"})
        actions = [
            Action(
                action_type="move",
                source=str(tmp_path / "file.txt"),
                destination=str(tmp_path / "TextFiles" / "file.txt"),
            )
        ]

        performed, skipped, errors = execute_moves(actions, safefs)
        assert performed == 1
        assert skipped == 0
        assert len(errors) == 0
        assert (tmp_path / "TextFiles" / "file.txt").exists()
        assert not (tmp_path / "file.txt").exists()

    def test_handles_missing_source(self, tmp_path):
        safefs = SafeFS(allowed_paths=[tmp_path], permissions={"read", "move"})
        actions = [
            Action(
                action_type="move",
                source=str(tmp_path / "nonexistent.txt"),
                destination=str(tmp_path / "Misc" / "nonexistent.txt"),
            )
        ]

        performed, skipped, errors = execute_moves(actions, safefs)
        assert performed == 0
        assert skipped == 1
        assert len(errors) == 1


class TestMoveRecord:
    def test_serialization_roundtrip(self):
        record = MoveRecord(
            filename="test.txt",
            source="/a/test.txt",
            destination="/a/Docs/test.txt",
            category="Docs",
            timestamp="2026-01-01T00:00:00",
        )
        json_str = record.to_json()
        restored = MoveRecord.from_json(json_str)
        assert restored.filename == record.filename
        assert restored.source == record.source
        assert restored.destination == record.destination
