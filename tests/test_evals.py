"""Tests for the eval framework — scenarios, scoring, runner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localagent.evals.scenario import EvalResult, EvalScore
from localagent.evals.runner import run_scenario, run_eval, save_results
from localagent.skills.file_organizer.evals.scenarios import (
    ALL_SCENARIOS,
    BasicFileTypes,
    ContentAwareCategorization,
    AmbiguousFiles,
    LargeFileset,
    SemanticNuance,
    get_scenarios,
)


# ── Scenario registry ──────────────────────────────────────────────────────


class TestScenarioRegistry:
    def test_all_scenarios_registered(self):
        assert len(ALL_SCENARIOS) == 5

    def test_get_all_scenarios(self):
        scenarios = get_scenarios()
        assert len(scenarios) == 5

    def test_get_scenarios_by_name(self):
        scenarios = get_scenarios(["basic-file-types", "content-aware"])
        assert len(scenarios) == 2
        assert scenarios[0].name == "basic-file-types"
        assert scenarios[1].name == "content-aware"

    def test_get_scenarios_unknown_name(self):
        scenarios = get_scenarios(["nonexistent"])
        assert len(scenarios) == 0

    def test_all_scenarios_have_profiles(self):
        for s in ALL_SCENARIOS:
            profiles = s.get_file_profiles()
            assert len(profiles) > 0, f"{s.name} has no profiles"
            for p in profiles:
                assert "name" in p, f"{s.name}: profile missing 'name'"
                assert "extension" in p, f"{s.name}: profile missing 'extension'"


# ── Scoring tests (with known-good LLM outputs) ────────────────────────────


class TestBasicFileTypesScoring:
    def test_perfect_categorization(self):
        scenario = BasicFileTypes()
        result = {
            "taxonomy": {
                "Documents": "Text documents and reports",
                "Images": "Photos and graphics",
                "Code": "Source code files",
                "Data": "Datasets and spreadsheets",
                "Media": "Audio and video",
                "Archives": "Compressed files",
                "Presentations": "Slide decks",
            },
            "assignments": {
                "report.pdf": "Documents",
                "notes.txt": "Documents",
                "vacation.jpg": "Images",
                "sunset.png": "Images",
                "backup.zip": "Archives",
                "app.py": "Code",
                "style.css": "Code",
                "data.csv": "Data",
                "presentation.pptx": "Presentations",
                "song.mp3": "Media",
            },
        }
        score = scenario.score(result)
        assert score.passed
        assert score.score >= 0.8
        assert score.percentage >= 80

    def test_all_in_one_category_fails(self):
        scenario = BasicFileTypes()
        result = {
            "taxonomy": {"Stuff": "everything"},
            "assignments": {p["name"]: "Stuff" for p in scenario.get_file_profiles()},
        }
        score = scenario.score(result)
        # Taxonomy too small + no separation = low score, but coverage + grouping
        # still contribute, so it lands around 0.6 — should not score well
        assert score.score < 0.7

    def test_missing_files_penalized(self):
        scenario = BasicFileTypes()
        result = {
            "taxonomy": {"Docs": "documents"},
            "assignments": {"report.pdf": "Docs"},  # only 1 of 10
        }
        score = scenario.score(result)
        assert score.score < 0.5


class TestContentAwareScoring:
    def test_semantic_separation(self):
        scenario = ContentAwareCategorization()
        result = {
            "taxonomy": {
                "Machine Learning": "ML and AI code",
                "DevOps": "Deployment and infrastructure",
                "Data Analysis": "Analytics and business data",
                "Web Development": "Web apps and APIs",
                "Testing": "Test suites",
                "Finance": "Financial records",
                "Documentation": "Project docs",
            },
            "assignments": {
                "train.py": "Machine Learning",
                "deploy.sh": "DevOps",
                "analyze_sales.py": "Data Analysis",
                "server.js": "Web Development",
                "test_auth.py": "Testing",
                "invoice_q4.csv": "Finance",
                "README.md": "Documentation",
            },
        }
        score = scenario.score(result)
        assert score.passed
        assert score.score >= 0.6


class TestAmbiguousFilesScoring:
    def test_photos_grouped(self):
        scenario = AmbiguousFiles()
        result = {
            "taxonomy": {
                "Photos": "Image files",
                "Documents": "PDFs and docs",
                "Notes": "Text files",
                "Screenshots": "Screen captures",
                "Misc": "Other files",
            },
            "assignments": {
                "document": "Misc",
                "IMG_20240315_142233.jpg": "Photos",
                "IMG_20240315_142234.jpg": "Photos",
                "IMG_20240315_142235.jpg": "Photos",
                "download.pdf": "Documents",
                "download (1).pdf": "Documents",
                "Untitled.txt": "Notes",
                "Screenshot 2024-03-15 at 2.30.22 PM.png": "Screenshots",
                "final_final_v2_FINAL.docx": "Documents",
                "asdf.txt": "Notes",
            },
        }
        score = scenario.score(result)
        assert score.passed
        assert score.score >= 0.7


# ── EvalScore / EvalResult data tests ──────────────────────────────────────


class TestEvalScore:
    def test_percentage(self):
        s = EvalScore(scenario_name="test", passed=True, score=0.75, max_score=1.0)
        assert s.percentage == 75.0

    def test_zero_max_score(self):
        s = EvalScore(scenario_name="test", passed=True, score=0, max_score=0)
        assert s.percentage == 0.0


class TestEvalResult:
    def test_aggregation(self):
        r = EvalResult(
            model_name="test-model",
            skill_name="test-skill",
            scores=[
                EvalScore(scenario_name="a", passed=True, score=0.8, max_score=1.0),
                EvalScore(scenario_name="b", passed=False, score=0.3, max_score=1.0),
                EvalScore(scenario_name="c", passed=True, score=1.0, max_score=1.0),
            ],
        )
        assert r.total_score == pytest.approx(2.1)
        assert r.max_total_score == 3.0
        assert r.passed == 2
        assert r.failed == 1
        assert r.percentage == pytest.approx(70.0)


# ── Runner tests (with mocked engine) ──────────────────────────────────────


class TestRunner:
    @patch("localagent.evals.runner.Engine")
    def test_run_scenario_success(self, MockEngine):
        engine = MagicMock()
        engine.generate_json.return_value = {
            "taxonomy": {
                "Documents": "docs",
                "Images": "pics",
                "Code": "source",
                "Data": "datasets",
                "Media": "audio/video",
                "Archives": "compressed",
            },
            "assignments": {
                "report.pdf": "Documents",
                "notes.txt": "Documents",
                "vacation.jpg": "Images",
                "sunset.png": "Images",
                "backup.zip": "Archives",
                "app.py": "Code",
                "style.css": "Code",
                "data.csv": "Data",
                "presentation.pptx": "Documents",
                "song.mp3": "Media",
            },
        }

        scenario = BasicFileTypes()
        score = run_scenario(engine, scenario)
        assert isinstance(score, EvalScore)
        assert score.scenario_name == "basic-file-types"
        assert score.error is None

    @patch("localagent.evals.runner.Engine")
    def test_run_scenario_llm_failure(self, MockEngine):
        engine = MagicMock()
        engine.generate_json.side_effect = ValueError("JSON parse failed")

        scenario = BasicFileTypes()
        score = run_scenario(engine, scenario)
        assert not score.passed
        assert score.score == 0.0
        assert "JSON parse failed" in score.error


class TestSaveResults:
    def test_save_and_read(self, tmp_path):
        results = [
            EvalResult(
                model_name="test-model",
                skill_name="file-organizer",
                scores=[
                    EvalScore(scenario_name="test", passed=True, score=0.9, max_score=1.0),
                ],
                total_time_seconds=5.0,
            )
        ]
        output = tmp_path / "results.yaml"
        save_results(results, output)
        assert output.exists()

        import yaml
        with open(output) as f:
            data = yaml.safe_load(f)
        assert len(data) == 1
        assert data[0]["model"] == "test-model"
        assert data[0]["percentage"] == 90.0
