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
    _extract_pdf_text,
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


# ── PDF extraction tests ──────────────────────────────────────────────────


class TestPdfExtraction:
    def test_returns_none_when_pymupdf_missing(self, tmp_path):
        """Without pymupdf installed, extraction gracefully returns None."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake content")
        with patch.dict("sys.modules", {"fitz": None}):
            result = _extract_pdf_text(pdf, max_bytes=512)
        assert result is None

    def test_returns_none_for_corrupt_pdf(self, tmp_path):
        """Corrupt/non-PDF files should return None without raising."""
        bad_pdf = tmp_path / "corrupt.pdf"
        bad_pdf.write_bytes(b"this is not a real PDF at all")
        result = _extract_pdf_text(bad_pdf, max_bytes=512)
        # Either None (no pymupdf) or None (extraction fails on garbage)
        assert result is None

    def test_extracts_text_when_pymupdf_available(self, tmp_path):
        """If pymupdf is installed, verify it extracts text from a real PDF."""
        try:
            import fitz
        except ImportError:
            pytest.skip("pymupdf not installed")

        # Create a minimal real PDF with text
        pdf_path = tmp_path / "sample.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Invoice #12345\nAmount: $500.00")
        doc.save(str(pdf_path))
        doc.close()

        result = _extract_pdf_text(pdf_path, max_bytes=512)
        assert result is not None
        assert "Invoice" in result
        assert "12345" in result

    def test_respects_max_bytes(self, tmp_path):
        """Extracted text should be capped at max_bytes."""
        try:
            import fitz
        except ImportError:
            pytest.skip("pymupdf not installed")

        pdf_path = tmp_path / "long.pdf"
        doc = fitz.open()
        page = doc.new_page()
        long_text = "A" * 2000
        page.insert_text((72, 72), long_text)
        doc.save(str(pdf_path))
        doc.close()

        result = _extract_pdf_text(pdf_path, max_bytes=100)
        assert result is not None
        assert len(result) <= 100

    def test_scan_directory_uses_pdf_extraction(self, tmp_path):
        """PDF files in scan_directory should get content_preview when pymupdf is available."""
        try:
            import fitz
        except ImportError:
            pytest.skip("pymupdf not installed")

        pdf_path = tmp_path / "invoice.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Payment receipt for services")
        doc.save(str(pdf_path))
        doc.close()

        safefs = SafeFS(allowed_paths=[tmp_path], permissions={"read"})
        profiles = scan_directory(safefs, tmp_path)
        by_name = {p.name: p for p in profiles}

        assert "invoice.pdf" in by_name
        assert by_name["invoice.pdf"].content_preview is not None
        assert "Payment" in by_name["invoice.pdf"].content_preview


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
                "Code": "source code",
                "report.pdf": "a bad filename category",
                "true": "a YAML artifact",
            },
            "assignments": {
                "f1": "report.pdf",
                "f2": "true",
            },
        }
        validated = _validate_response(result, id_map, all_filenames)
        # Bad categories should be removed, files reassigned via extension
        assert "report.pdf" not in validated["taxonomy"]
        assert "true" not in validated["taxonomy"]
        # .md has no extension category → "Other Files"; .csv → "Spreadsheets"
        assert validated["assignments"]["readme.md"] == "Other Files"
        assert validated["assignments"]["data.csv"] == "Spreadsheets"

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
        # .HEIC → "Photos" (from extension), .txt → "Other Files" (no ext category)
        assert validated["assignments"]["IMG_0019.HEIC"] == "Photos"
        assert validated["assignments"]["notes.txt"] == "Other Files"


class TestBadCategoryName:
    def test_rejects_exact_filename_match(self):
        filenames = {"report.pdf", "IMG_2034.jpg", "script.py"}
        assert _is_bad_category_name("report.pdf", filenames)
        assert _is_bad_category_name("IMG_2034.jpg", filenames)
        assert _is_bad_category_name("script.py", filenames)

    def test_rejects_long_filename_stem_as_category(self):
        filenames = {"Logo-Red_Hat-Engineering.eps", "Data Literacy Learning Paths.pdf"}
        # Long filename stems (>= 8 chars) used as categories are rejected
        assert _is_bad_category_name("Logo-Red_Hat-Engineering", filenames)
        assert _is_bad_category_name("Data Literacy Learning Paths", filenames)
        # Case-insensitive stem matching
        assert _is_bad_category_name("data literacy learning paths", filenames)

    def test_rejects_long_substring_of_filename(self):
        filenames = {"Logo-Red_Hat-Engineering.eps", "Airtel Black Statement.pdf"}
        # Category (>= 6 chars) that is a substring of a filename
        assert _is_bad_category_name("Logo-Red_Hat", filenames)
        assert _is_bad_category_name("Airtel Black", filenames)

    def test_rejects_medium_filename_stem_as_category(self):
        filenames = {"Aadhaar.pdf", "VFS GLOBAL.pdf"}
        # Stems >= 6 chars that match a filename are too specific
        assert _is_bad_category_name("Aadhaar", filenames)
        assert _is_bad_category_name("VFS GLOBAL", filenames)

    def test_allows_short_filename_stem_as_category(self):
        filenames = {"Code.zip", "Data.csv", "Music.tar"}
        # Short stems (< 6 chars) are legitimate category names
        assert not _is_bad_category_name("Code", filenames)
        assert not _is_bad_category_name("Data", filenames)
        assert not _is_bad_category_name("Music", filenames)

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

    def test_rejects_generic_catchall_categories(self):
        assert _is_bad_category_name("Documents")
        assert _is_bad_category_name("Miscellaneous")
        assert _is_bad_category_name("Other")
        assert _is_bad_category_name("General")
        assert _is_bad_category_name("Uncategorized")

    def test_accepts_valid_categories(self):
        filenames = {"report.pdf", "photo.jpg"}
        assert not _is_bad_category_name("Receipts & Invoices", filenames)
        assert not _is_bad_category_name("Code Projects", filenames)
        assert not _is_bad_category_name("Screenshots", filenames)
        assert not _is_bad_category_name("Tax Documents", filenames)


class TestNormalizeCategories:
    def test_merges_case_duplicates(self):
        taxonomy = {"Receipts": "payment records", "receipts": "also payment records"}
        assignments = {"a.pdf": "Receipts", "b.pdf": "receipts"}
        norm_tax, norm_assign = _normalize_categories(taxonomy, assignments)
        assert len(norm_tax) == 1
        assert "Receipts" in norm_tax
        assert norm_assign["a.pdf"] == "Receipts"
        assert norm_assign["b.pdf"] == "Receipts"

    def test_merges_substring_duplicates(self):
        taxonomy = {
            "Invoices": "billing documents",
            "Tax Invoices": "also billing documents",
        }
        assignments = {"a.pdf": "Invoices", "b.pdf": "Tax Invoices"}
        norm_tax, norm_assign = _normalize_categories(taxonomy, assignments)
        assert len(norm_tax) == 1
        assert "Invoices" in norm_tax
        assert norm_assign["b.pdf"] == "Invoices"

    def test_keeps_distinct_categories(self):
        taxonomy = {
            "Receipts": "payment records",
            "Code": "source code",
            "Screenshots": "screen captures",
        }
        assignments = {"a.pdf": "Receipts", "b.py": "Code"}
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
                "Receipts": "Payment records and invoices",
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
                "Project Notes": "Markdown notes and READMEs",
                "Data Science": "ML and data files",
            },
            "assignments": {
                "f1": "Project Notes",
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
        assert result["assignments"]["readme.md"] == "Project Notes"
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
