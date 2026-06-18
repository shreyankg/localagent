"""Tests for the file organizer skill — scanner, categorizer, mover."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
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
    _assign_to_existing,
    _build_naming_prompt,
    _cluster_embeddings,
    _compute_centroids,
    _sample_clusters,
    _validate_names,
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

    def test_embedding_text_with_preview(self):
        p = FileProfile(
            name="script.py",
            path=Path("/tmp/script.py"),
            extension=".py",
            mime_type="text/x-python",
            size_bytes=100,
            content_preview="import torch\nmodel = torch.nn.Linear(10, 5)",
        )
        text = p.embedding_text()
        assert "script.py" in text
        assert ".py" in text
        assert "import torch" in text

    def test_embedding_text_without_preview(self):
        p = FileProfile(
            name="photo.jpg",
            path=Path("/tmp/photo.jpg"),
            extension=".jpg",
            mime_type="image/jpeg",
            size_bytes=5000,
        )
        text = p.embedding_text()
        assert "photo.jpg" in text
        assert ".jpg" in text


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


class TestClustering:
    def test_few_files_get_individual_clusters(self):
        """When there are <= 3 files, each gets its own cluster."""
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        labels = _cluster_embeddings(embeddings, distance_threshold=0.4)
        assert len(labels) == 2
        assert labels[0] == 0
        assert labels[1] == 1

    def test_similar_files_cluster_together(self):
        """Very similar embeddings should end up in the same cluster."""
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.98, 0.02, 0.0],
            [0.0, 1.0, 0.0],
            [0.01, 0.99, 0.0],
            [0.02, 0.98, 0.0],
        ], dtype=np.float32)
        # Normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        labels = _cluster_embeddings(embeddings, distance_threshold=0.3)
        # Should produce 2 clusters (group of x-axis vs group of y-axis)
        assert len(set(labels)) == 2
        # First three should be in same cluster
        assert labels[0] == labels[1] == labels[2]
        # Last three should be in same cluster
        assert labels[3] == labels[4] == labels[5]

    def test_compute_centroids(self):
        embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ], dtype=np.float32)
        labels = np.array([0, 1, 0])
        centroids = _compute_centroids(embeddings, labels)
        assert centroids.shape == (2, 2)
        # Cluster 0 centroid should be [1, 0] (normalised)
        np.testing.assert_allclose(centroids[0], [1.0, 0.0], atol=0.01)
        # Cluster 1 centroid should be [0, 1] (normalised)
        np.testing.assert_allclose(centroids[1], [0.0, 1.0], atol=0.01)


class TestAssignToExisting:
    def test_matches_close_embeddings(self):
        existing_centroids = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        existing_names = ["Code Files", "Documents"]
        new_embeddings = np.array([
            [0.95, 0.05, 0.0],  # close to Code Files
            [0.05, 0.95, 0.0],  # close to Documents
        ], dtype=np.float32)
        # Normalise
        norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        new_embeddings = new_embeddings / norms

        _, matched_names = _assign_to_existing(
            new_embeddings, existing_centroids, existing_names,
        )
        assert matched_names[0] == "Code Files"
        assert matched_names[1] == "Documents"

    def test_rejects_distant_embeddings(self):
        existing_centroids = np.array([
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)
        existing_names = ["Code Files"]
        # Orthogonal = cosine distance 1.0, well above cutoff
        new_embeddings = np.array([
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)

        _, matched_names = _assign_to_existing(
            new_embeddings, existing_centroids, existing_names,
        )
        assert matched_names[0] is None


class TestValidateNames:
    def test_valid_names_with_descriptions(self):
        raw = {
            "0": {"name": "Code Projects", "description": "Source code and dev files"},
            "1": {"name": "Tax Documents", "description": "Tax returns and related forms"},
        }
        result = _validate_names(raw, n_clusters=2)
        assert result[0] == ("Code Projects", "Source code and dev files")
        assert result[1] == ("Tax Documents", "Tax returns and related forms")

    def test_plain_string_fallback(self):
        raw = {"0": "Code Projects", "1": "Tax Documents"}
        result = _validate_names(raw, n_clusters=2)
        assert result[0][0] == "Code Projects"
        assert result[1][0] == "Tax Documents"

    def test_generic_names_rejected(self):
        raw = {
            "0": {"name": "Miscellaneous", "description": "misc stuff"},
            "1": {"name": "Documents", "description": "docs"},
            "2": {"name": "Code Projects", "description": "Source code files"},
        }
        result = _validate_names(raw, n_clusters=3)
        assert result[0][0] == "Group 0"  # fallback
        assert result[1][0] == "Group 1"  # fallback
        assert result[2][0] == "Code Projects"  # kept
        assert result[2][1] == "Source code files"

    def test_non_integer_keys_skipped(self):
        raw = {"zero": "Something", "1": {"name": "Code Projects", "description": "Source code"}}
        result = _validate_names(raw, n_clusters=2)
        assert 1 in result
        assert len(result) == 1

    def test_empty_names_rejected(self):
        raw = {
            "0": {"name": "", "description": "empty"},
            "1": {"name": "a", "description": "too short"},
            "2": {"name": "Valid Name", "description": "A real category"},
        }
        result = _validate_names(raw, n_clusters=3)
        assert result[0][0] == "Group 0"
        assert result[1][0] == "Group 1"
        assert result[2] == ("Valid Name", "A real category")


class TestNamingPrompt:
    def test_cold_run_prompt(self):
        samples = {
            0: [{"name": "script.py", "extension": ".py"}],
            1: [{"name": "photo.jpg", "extension": ".jpg"}],
        }
        messages = _build_naming_prompt(samples)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "2-to-3 word" in messages[0]["content"]
        assert "Group 0" in messages[1]["content"]

    def test_warm_run_prompt_includes_existing_names(self):
        samples = {
            0: [{"name": "new_file.rs", "extension": ".rs"}],
        }
        messages = _build_naming_prompt(samples, existing_names=["Code Projects", "Photos"])
        user_content = messages[1]["content"]
        assert "Code Projects" in user_content
        assert "Photos" in user_content
        assert "Reuse" in user_content


class TestSampleClusters:
    def test_limits_samples_per_cluster(self):
        profiles = [
            FileProfile(
                name=f"file{i}.txt",
                path=Path(f"/tmp/file{i}.txt"),
                extension=".txt",
                mime_type="text/plain",
                size_bytes=100,
            )
            for i in range(20)
        ]
        labels = np.array([0] * 10 + [1] * 10)
        samples = _sample_clusters(labels, profiles, max_samples=3)
        assert len(samples[0]) == 3
        assert len(samples[1]) == 3

    def test_prefers_files_with_previews(self):
        profiles = [
            FileProfile(
                name="no_preview.txt",
                path=Path("/tmp/no_preview.txt"),
                extension=".txt",
                mime_type="text/plain",
                size_bytes=100,
            ),
            FileProfile(
                name="has_preview.txt",
                path=Path("/tmp/has_preview.txt"),
                extension=".txt",
                mime_type="text/plain",
                size_bytes=100,
                content_preview="some content",
            ),
        ]
        labels = np.array([0, 0])
        samples = _sample_clusters(labels, profiles, max_samples=1)
        # Should pick the file with preview first
        assert samples[0][0]["name"] == "has_preview.txt"


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
    def test_cold_start_categorization(self, tmp_path):
        """Test first-run categorization with mocked embedder and LLM."""
        engine = MagicMock()
        engine.generate_json.return_value = {
            "names": {
                "0": {"name": "Project Notes", "description": "Markdown notes and READMEs"},
                "1": {"name": "Data Science", "description": "ML and data analysis scripts"},
            },
        }

        embedder = MagicMock()
        # Return embeddings that will cluster into 2 groups
        embedder.embed.return_value = np.array([
            [1.0, 0.0, 0.0],  # readme.md
            [0.0, 1.0, 0.0],  # train.py
        ], dtype=np.float32)

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

        result = categorize(engine, embedder, profiles, tmp_path)

        assert "taxonomy" in result
        assert "assignments" in result
        assert result["assignments"]["readme.md"] == "Project Notes"
        assert result["assignments"]["train.py"] == "Data Science"

        # Taxonomy descriptions should come from the LLM, not be placeholders
        assert result["taxonomy"]["Project Notes"] == "Markdown notes and READMEs"
        assert result["taxonomy"]["Data Science"] == "ML and data analysis scripts"

        # Taxonomy should be saved
        assert load_taxonomy(tmp_path) is not None

        # Embedder should have been called
        embedder.embed.assert_called_once()

    def test_empty_profiles_returns_empty(self, tmp_path):
        engine = MagicMock()
        embedder = MagicMock()
        result = categorize(engine, embedder, [], tmp_path)
        assert result["assignments"] == {}
        assert result["taxonomy"] == {}

    def test_warm_run_reuses_existing_clusters(self, tmp_path):
        """On warm run, files close to existing centroids get assigned without LLM."""
        # Set up saved centroids from a previous run
        existing_centroids = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        existing_names = ["Code Files", "Documents"]
        np.savez(tmp_path / "centroids.npz", centroids=existing_centroids)
        with open(tmp_path / "centroid_names.yaml", "w") as f:
            yaml.dump(existing_names, f)
        save_taxonomy(tmp_path, {"taxonomy": {
            "Code Files": "Source code",
            "Documents": "Text documents",
        }})

        engine = MagicMock()
        embedder = MagicMock()
        # Return embeddings very close to existing centroids
        embedder.embed.return_value = np.array([
            [0.99, 0.01, 0.0],  # close to Code Files
            [0.01, 0.99, 0.0],  # close to Documents
        ], dtype=np.float32)

        profiles = [
            FileProfile(
                name="new_script.py",
                path=tmp_path / "new_script.py",
                extension=".py",
                mime_type="text/x-python",
                size_bytes=200,
            ),
            FileProfile(
                name="report.txt",
                path=tmp_path / "report.txt",
                extension=".txt",
                mime_type="text/plain",
                size_bytes=150,
            ),
        ]

        result = categorize(engine, embedder, profiles, tmp_path)

        assert result["assignments"]["new_script.py"] == "Code Files"
        assert result["assignments"]["report.txt"] == "Documents"

        # LLM should NOT have been called (all matched existing)
        engine.generate_json.assert_not_called()


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
            assignments={"doc.pdf": "Tax Documents"},
            profiles_by_name={"doc.pdf": profile},
            watch_directories=[tmp_path],
        )

        assert len(actions) == 1
        assert actions[0].action_type == "move"
        assert "doc.pdf" in actions[0].source
        assert "Tax Documents" in actions[0].destination


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
