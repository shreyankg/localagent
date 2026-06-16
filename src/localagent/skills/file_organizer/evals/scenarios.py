"""Eval scenarios for the file organizer skill.

Each scenario tests a specific aspect of categorization quality:
- Can the LLM distinguish file types by content, not just extension?
- Does it create sensible semantic categories?
- Does it handle ambiguous or unusual files?
- Does it assign every file?
- Does it avoid too many or too few categories?
"""

from __future__ import annotations

from typing import Any

from localagent.evals.scenario import EvalScenario, EvalScore


# ── Scoring helpers ─────────────────────────────────────────────────────────


def _check_all_assigned(
    assignments: dict[str, str],
    expected_files: set[str],
) -> tuple[float, dict[str, Any]]:
    """Check that every expected file got assigned. Returns (score, details)."""
    assigned = set(assignments.keys())
    missing = expected_files - assigned
    hallucinated = assigned - expected_files
    coverage = len(assigned & expected_files) / len(expected_files) if expected_files else 1.0
    return coverage, {
        "coverage": round(coverage, 3),
        "missing_files": sorted(missing) if missing else [],
        "hallucinated_files": sorted(hallucinated) if hallucinated else [],
    }


def _check_taxonomy_size(
    taxonomy: dict[str, str],
    min_cats: int = 2,
    max_cats: int = 15,
) -> tuple[float, dict[str, Any]]:
    """Check that the taxonomy has a reasonable number of categories."""
    n = len(taxonomy)
    if min_cats <= n <= max_cats:
        score = 1.0
    elif n < min_cats:
        score = n / min_cats
    else:
        score = max(0.0, 1.0 - (n - max_cats) / max_cats)
    return score, {"num_categories": n, "range": f"{min_cats}-{max_cats}"}


def _check_grouping(
    assignments: dict[str, str],
    expected_groups: list[set[str]],
) -> tuple[float, dict[str, Any]]:
    """Check that files expected to be in the same category are grouped together.

    ``expected_groups`` is a list of sets of filenames that should share a category.
    """
    total = len(expected_groups)
    matched = 0
    group_details: list[dict[str, Any]] = []

    for group in expected_groups:
        cats = set()
        for f in group:
            if f in assignments:
                cats.add(assignments[f])
        grouped = len(cats) == 1 and len(cats) > 0
        if grouped:
            matched += 1
        group_details.append({
            "files": sorted(group),
            "categories_assigned": sorted(cats),
            "grouped": grouped,
        })

    score = matched / total if total > 0 else 1.0
    return score, {"groups": group_details}


def _check_separation(
    assignments: dict[str, str],
    must_separate: list[tuple[str, str]],
) -> tuple[float, dict[str, Any]]:
    """Check that files expected to be in different categories are separated."""
    total = len(must_separate)
    separated = 0

    for f1, f2 in must_separate:
        c1 = assignments.get(f1)
        c2 = assignments.get(f2)
        if c1 and c2 and c1 != c2:
            separated += 1

    score = separated / total if total > 0 else 1.0
    return score, {"separated": separated, "total": total}


# ── Scenarios ───────────────────────────────────────────────────────────────


class BasicFileTypes(EvalScenario):
    """Can the model distinguish common file types and create sensible categories?"""

    @property
    def name(self) -> str:
        return "basic-file-types"

    @property
    def description(self) -> str:
        return "Categorize a simple mix of documents, images, code, and archives"

    def get_file_profiles(self) -> list[dict[str, Any]]:
        return [
            {"name": "report.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "2.4 MB"},
            {"name": "notes.txt", "extension": ".txt", "mime_type": "text/plain", "size": "1.2 KB", "content_preview": "Meeting notes from Q4 planning session\n- Budget review\n- Hiring plan for engineering"},
            {"name": "vacation.jpg", "extension": ".jpg", "mime_type": "image/jpeg", "size": "3.8 MB"},
            {"name": "sunset.png", "extension": ".png", "mime_type": "image/png", "size": "5.1 MB"},
            {"name": "backup.zip", "extension": ".zip", "mime_type": "application/zip", "size": "150 MB"},
            {"name": "app.py", "extension": ".py", "mime_type": "text/x-python", "size": "4.5 KB", "content_preview": "from flask import Flask\napp = Flask(__name__)\n\n@app.route('/')\ndef index():\n    return 'Hello World'"},
            {"name": "style.css", "extension": ".css", "mime_type": "text/css", "size": "2.1 KB", "content_preview": "body { font-family: sans-serif; }\n.header { background: #333; color: white; }"},
            {"name": "data.csv", "extension": ".csv", "mime_type": "text/csv", "size": "45 KB", "content_preview": "date,revenue,expenses,profit\n2024-01-01,50000,30000,20000\n2024-02-01,55000,32000,23000"},
            {"name": "presentation.pptx", "extension": ".pptx", "mime_type": "application/vnd.openxmlformats-officedocument.presentationml.presentation", "size": "8.2 MB"},
            {"name": "song.mp3", "extension": ".mp3", "mime_type": "audio/mpeg", "size": "4.5 MB"},
        ]

    def score(self, result: dict[str, Any]) -> EvalScore:
        taxonomy = result.get("taxonomy", {})
        assignments = result.get("assignments", {})
        expected_files = {p["name"] for p in self.get_file_profiles()}

        coverage_score, coverage_details = _check_all_assigned(assignments, expected_files)
        taxonomy_score, taxonomy_details = _check_taxonomy_size(taxonomy, min_cats=3, max_cats=10)

        # Images should be grouped
        grouping_score, grouping_details = _check_grouping(
            assignments,
            [{"vacation.jpg", "sunset.png"}],
        )

        # Code files should be separate from documents
        sep_score, sep_details = _check_separation(
            assignments,
            [("app.py", "report.pdf"), ("app.py", "notes.txt")],
        )

        total = (coverage_score * 0.3) + (taxonomy_score * 0.2) + (grouping_score * 0.25) + (sep_score * 0.25)
        return EvalScore(
            scenario_name=self.name,
            passed=total >= 0.6,
            score=round(total, 3),
            max_score=1.0,
            details={
                "coverage": coverage_details,
                "taxonomy": taxonomy_details,
                "grouping": grouping_details,
                "separation": sep_details,
            },
        )


class ContentAwareCategorization(EvalScenario):
    """Can the model use content previews to categorize beyond just extensions?"""

    @property
    def name(self) -> str:
        return "content-aware"

    @property
    def description(self) -> str:
        return "Categorize Python files by their purpose using content, not just extension"

    def get_file_profiles(self) -> list[dict[str, Any]]:
        return [
            {"name": "train.py", "extension": ".py", "mime_type": "text/x-python", "size": "8.2 KB", "content_preview": "import torch\nimport torch.nn as nn\nfrom transformers import AutoModel\n\nclass FineTuner:\n    def train(self, dataset):"},
            {"name": "deploy.sh", "extension": ".sh", "mime_type": "text/x-shellscript", "size": "1.1 KB", "content_preview": "#!/bin/bash\nset -e\ndocker build -t myapp .\ndocker push myapp:latest\nkubectl rollout restart deployment/myapp"},
            {"name": "analyze_sales.py", "extension": ".py", "mime_type": "text/x-python", "size": "3.4 KB", "content_preview": "import pandas as pd\nimport matplotlib.pyplot as plt\n\ndf = pd.read_csv('sales_2024.csv')\nmonthly = df.groupby('month').sum()\nmonthly.plot(kind='bar')"},
            {"name": "server.js", "extension": ".js", "mime_type": "application/javascript", "size": "2.8 KB", "content_preview": "const express = require('express');\nconst cors = require('cors');\nconst app = express();\napp.use(cors());\napp.get('/api/users', async (req, res) => {"},
            {"name": "test_auth.py", "extension": ".py", "mime_type": "text/x-python", "size": "2.1 KB", "content_preview": "import pytest\nfrom app.auth import login, verify_token\n\ndef test_login_success():\n    result = login('user@test.com', 'password123')\n    assert result.status == 200"},
            {"name": "invoice_q4.csv", "extension": ".csv", "mime_type": "text/csv", "size": "12 KB", "content_preview": "invoice_id,client,amount,date,status\nINV-001,Acme Corp,5000.00,2024-10-15,paid\nINV-002,Beta Inc,3200.00,2024-11-01,pending"},
            {"name": "README.md", "extension": ".md", "mime_type": "text/markdown", "size": "3.5 KB", "content_preview": "# MyApp\n\nA web application for managing customer relationships.\n\n## Installation\n\n```bash\nnpm install\nnpm start\n```"},
        ]

    def score(self, result: dict[str, Any]) -> EvalScore:
        taxonomy = result.get("taxonomy", {})
        assignments = result.get("assignments", {})
        expected_files = {p["name"] for p in self.get_file_profiles()}

        coverage_score, coverage_details = _check_all_assigned(assignments, expected_files)

        # ML script and data analysis script should NOT be in the same category as web server code
        sep_score, sep_details = _check_separation(
            assignments,
            [
                ("train.py", "server.js"),
                ("analyze_sales.py", "deploy.sh"),
                ("test_auth.py", "invoice_q4.csv"),
            ],
        )

        # Financial files could reasonably be grouped
        grouping_score, grouping_details = _check_grouping(
            assignments,
            [{"invoice_q4.csv", "analyze_sales.py"}],  # Both finance-related
        )

        total = (coverage_score * 0.3) + (sep_score * 0.4) + (grouping_score * 0.3)
        return EvalScore(
            scenario_name=self.name,
            passed=total >= 0.5,
            score=round(total, 3),
            max_score=1.0,
            details={
                "coverage": coverage_details,
                "separation": sep_details,
                "grouping": grouping_details,
            },
        )


class AmbiguousFiles(EvalScenario):
    """Can the model handle files with misleading extensions or unclear purpose?"""

    @property
    def name(self) -> str:
        return "ambiguous-files"

    @property
    def description(self) -> str:
        return "Handle files with misleading names, missing extensions, or unclear purpose"

    def get_file_profiles(self) -> list[dict[str, Any]]:
        return [
            {"name": "document", "extension": "(none)", "mime_type": "application/octet-stream", "size": "45 KB"},
            {"name": "IMG_20240315_142233.jpg", "extension": ".jpg", "mime_type": "image/jpeg", "size": "4.2 MB"},
            {"name": "IMG_20240315_142234.jpg", "extension": ".jpg", "mime_type": "image/jpeg", "size": "3.9 MB"},
            {"name": "IMG_20240315_142235.jpg", "extension": ".jpg", "mime_type": "image/jpeg", "size": "4.1 MB"},
            {"name": "download.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "1.2 MB"},
            {"name": "download (1).pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "800 KB"},
            {"name": "Untitled.txt", "extension": ".txt", "mime_type": "text/plain", "size": "256 B", "content_preview": "grocery list\n- milk\n- eggs\n- bread\n- butter"},
            {"name": "Screenshot 2024-03-15 at 2.30.22 PM.png", "extension": ".png", "mime_type": "image/png", "size": "1.8 MB"},
            {"name": "final_final_v2_FINAL.docx", "extension": ".docx", "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "size": "3.2 MB"},
            {"name": "asdf.txt", "extension": ".txt", "mime_type": "text/plain", "size": "12 B", "content_preview": "test test test"},
        ]

    def score(self, result: dict[str, Any]) -> EvalScore:
        taxonomy = result.get("taxonomy", {})
        assignments = result.get("assignments", {})
        expected_files = {p["name"] for p in self.get_file_profiles()}

        coverage_score, coverage_details = _check_all_assigned(assignments, expected_files)
        taxonomy_score, taxonomy_details = _check_taxonomy_size(taxonomy, min_cats=2, max_cats=8)

        # The three IMG_ photos should be grouped together
        grouping_score, grouping_details = _check_grouping(
            assignments,
            [
                {"IMG_20240315_142233.jpg", "IMG_20240315_142234.jpg", "IMG_20240315_142235.jpg"},
            ],
        )

        total = (coverage_score * 0.4) + (taxonomy_score * 0.2) + (grouping_score * 0.4)
        return EvalScore(
            scenario_name=self.name,
            passed=total >= 0.6,
            score=round(total, 3),
            max_score=1.0,
            details={
                "coverage": coverage_details,
                "taxonomy": taxonomy_details,
                "grouping": grouping_details,
            },
        )


class LargeFileset(EvalScenario):
    """Can the model handle a larger, more realistic set of files?"""

    @property
    def name(self) -> str:
        return "large-fileset"

    @property
    def description(self) -> str:
        return "Categorize 25 files spanning many types — tests scale and taxonomy quality"

    def get_file_profiles(self) -> list[dict[str, Any]]:
        return [
            # Documents
            {"name": "resume_2024.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "200 KB"},
            {"name": "cover_letter.docx", "extension": ".docx", "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "size": "45 KB"},
            {"name": "tax_return_2023.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "1.5 MB"},
            {"name": "lease_agreement.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "3.2 MB"},
            # Finances
            {"name": "expenses_march.xlsx", "extension": ".xlsx", "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "size": "85 KB"},
            {"name": "bank_statement_q1.csv", "extension": ".csv", "mime_type": "text/csv", "size": "120 KB", "content_preview": "date,description,amount,balance\n2024-01-02,SALARY DEPOSIT,5000.00,12500.00\n2024-01-03,RENT PAYMENT,-1800.00,10700.00"},
            {"name": "receipt_amazon_0315.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "150 KB"},
            # Photos
            {"name": "family_dinner.jpg", "extension": ".jpg", "mime_type": "image/jpeg", "size": "4.5 MB"},
            {"name": "passport_scan.png", "extension": ".png", "mime_type": "image/png", "size": "2.1 MB"},
            {"name": "product_mockup.psd", "extension": ".psd", "mime_type": "image/vnd.adobe.photoshop", "size": "45 MB"},
            # Code
            {"name": "main.py", "extension": ".py", "mime_type": "text/x-python", "size": "6 KB", "content_preview": "from fastapi import FastAPI\nfrom sqlalchemy import create_engine\n\napp = FastAPI(title='Inventory API')"},
            {"name": "schema.sql", "extension": ".sql", "mime_type": "text/plain", "size": "3 KB", "content_preview": "CREATE TABLE products (\n  id SERIAL PRIMARY KEY,\n  name VARCHAR(255),\n  price DECIMAL(10,2)\n);"},
            {"name": "Dockerfile", "extension": "(none)", "mime_type": "text/plain", "size": "800 B", "content_preview": "FROM python:3.12-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install -r requirements.txt"},
            # Media
            {"name": "podcast_ep42.mp3", "extension": ".mp3", "mime_type": "audio/mpeg", "size": "65 MB"},
            {"name": "tutorial_react.mp4", "extension": ".mp4", "mime_type": "video/mp4", "size": "250 MB"},
            # Archives
            {"name": "project_backup_2024.tar.gz", "extension": ".gz", "mime_type": "application/gzip", "size": "500 MB"},
            {"name": "fonts.zip", "extension": ".zip", "mime_type": "application/zip", "size": "12 MB"},
            # Misc
            {"name": "todo.md", "extension": ".md", "mime_type": "text/markdown", "size": "1 KB", "content_preview": "# TODO\n- [ ] Finish project proposal\n- [ ] Book flight for conference\n- [x] Submit expense report"},
            {"name": "bookmarks.html", "extension": ".html", "mime_type": "text/html", "size": "45 KB", "content_preview": "<!DOCTYPE NETSCAPE-Bookmark-file-1>\n<DL><DT><A HREF=\"https://news.ycombinator.com\">Hacker News</A>"},
            {"name": "id_rsa.pub", "extension": ".pub", "mime_type": "text/plain", "size": "600 B", "content_preview": "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQ..."},
            {"name": "meeting_recording.m4a", "extension": ".m4a", "mime_type": "audio/mp4", "size": "18 MB"},
            {"name": "wireframes.fig", "extension": ".fig", "mime_type": "application/octet-stream", "size": "8 MB"},
            {"name": "analytics_dashboard.ipynb", "extension": ".ipynb", "mime_type": "application/json", "size": "2.5 MB", "content_preview": "{\n \"cells\": [\n  {\"cell_type\": \"code\", \"source\": [\"import pandas as pd\\nimport plotly.express as px\"]}"},
            {"name": "wallpaper.heic", "extension": ".heic", "mime_type": "image/heic", "size": "6 MB"},
            {"name": "contract_signed.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "2 MB"},
        ]

    def score(self, result: dict[str, Any]) -> EvalScore:
        taxonomy = result.get("taxonomy", {})
        assignments = result.get("assignments", {})
        expected_files = {p["name"] for p in self.get_file_profiles()}

        coverage_score, coverage_details = _check_all_assigned(assignments, expected_files)
        taxonomy_score, taxonomy_details = _check_taxonomy_size(taxonomy, min_cats=5, max_cats=15)

        # Expected groupings
        grouping_score, grouping_details = _check_grouping(
            assignments,
            [
                {"resume_2024.pdf", "cover_letter.docx"},  # Job-related
                {"expenses_march.xlsx", "bank_statement_q1.csv", "receipt_amazon_0315.pdf"},  # Financial
                {"main.py", "schema.sql", "Dockerfile"},  # Code/dev
                {"podcast_ep42.mp3", "meeting_recording.m4a"},  # Audio
            ],
        )

        # Expected separations
        sep_score, sep_details = _check_separation(
            assignments,
            [
                ("main.py", "resume_2024.pdf"),
                ("family_dinner.jpg", "bank_statement_q1.csv"),
                ("podcast_ep42.mp3", "main.py"),
            ],
        )

        total = (coverage_score * 0.25) + (taxonomy_score * 0.15) + (grouping_score * 0.35) + (sep_score * 0.25)
        return EvalScore(
            scenario_name=self.name,
            passed=total >= 0.5,
            score=round(total, 3),
            max_score=1.0,
            details={
                "coverage": coverage_details,
                "taxonomy": taxonomy_details,
                "grouping": grouping_details,
                "separation": sep_details,
            },
        )


class SemanticNuance(EvalScenario):
    """Can the model pick up on semantic nuance from content previews?"""

    @property
    def name(self) -> str:
        return "semantic-nuance"

    @property
    def description(self) -> str:
        return "Distinguish files that share the same extension but have very different purposes"

    def get_file_profiles(self) -> list[dict[str, Any]]:
        return [
            {"name": "model_weights.py", "extension": ".py", "mime_type": "text/x-python", "size": "12 KB", "content_preview": "import torch\nfrom transformers import GPT2LMHeadModel\n\ndef load_weights(path):\n    model = GPT2LMHeadModel.from_pretrained(path)\n    return model"},
            {"name": "scrape_recipes.py", "extension": ".py", "mime_type": "text/x-python", "size": "3 KB", "content_preview": "import requests\nfrom bs4 import BeautifulSoup\n\ndef get_recipes(url):\n    soup = BeautifulSoup(requests.get(url).text)\n    return [r.text for r in soup.select('.recipe-title')]"},
            {"name": "budget_tracker.py", "extension": ".py", "mime_type": "text/x-python", "size": "5 KB", "content_preview": "import csv\nfrom datetime import date\n\ndef add_expense(category, amount):\n    with open('expenses.csv', 'a') as f:\n        csv.writer(f).writerow([date.today(), category, amount])"},
            {"name": "game_physics.py", "extension": ".py", "mime_type": "text/x-python", "size": "8 KB", "content_preview": "import pygame\nimport numpy as np\n\nclass RigidBody:\n    def __init__(self, mass, position):\n        self.velocity = np.zeros(2)\n        self.apply_gravity()"},
            {"name": "migrate_db.py", "extension": ".py", "mime_type": "text/x-python", "size": "2 KB", "content_preview": "from alembic import op\nimport sqlalchemy as sa\n\ndef upgrade():\n    op.add_column('users', sa.Column('email_verified', sa.Boolean))"},
            {"name": "personal_diary.md", "extension": ".md", "mime_type": "text/markdown", "size": "15 KB", "content_preview": "# March 15, 2024\n\nHad a great day at the park with the kids. Weather was perfect.\nTrying a new pasta recipe tonight."},
            {"name": "api_docs.md", "extension": ".md", "mime_type": "text/markdown", "size": "8 KB", "content_preview": "# API Reference\n\n## POST /api/v1/users\n\nCreates a new user account.\n\n### Parameters\n| Field | Type | Required |"},
        ]

    def score(self, result: dict[str, Any]) -> EvalScore:
        taxonomy = result.get("taxonomy", {})
        assignments = result.get("assignments", {})
        expected_files = {p["name"] for p in self.get_file_profiles()}

        coverage_score, coverage_details = _check_all_assigned(assignments, expected_files)

        # The model should NOT dump all .py files into one "Code" bucket
        # At minimum, ML script and game physics should be separate from budget tracker
        sep_score, sep_details = _check_separation(
            assignments,
            [
                ("model_weights.py", "budget_tracker.py"),
                ("game_physics.py", "migrate_db.py"),
                ("personal_diary.md", "api_docs.md"),
            ],
        )

        # Dev-related code should group
        grouping_score, grouping_details = _check_grouping(
            assignments,
            [
                {"migrate_db.py", "api_docs.md"},  # Both dev/infrastructure
            ],
        )

        total = (coverage_score * 0.3) + (sep_score * 0.5) + (grouping_score * 0.2)
        return EvalScore(
            scenario_name=self.name,
            passed=total >= 0.4,
            score=round(total, 3),
            max_score=1.0,
            details={
                "coverage": coverage_details,
                "separation": sep_details,
                "grouping": grouping_details,
            },
        )


# ── Registry ────────────────────────────────────────────────────────────────


ALL_SCENARIOS: list[EvalScenario] = [
    BasicFileTypes(),
    ContentAwareCategorization(),
    AmbiguousFiles(),
    LargeFileset(),
    SemanticNuance(),
]


def get_scenarios(names: list[str] | None = None) -> list[EvalScenario]:
    """Return scenarios filtered by name, or all if names is None."""
    if names is None:
        return list(ALL_SCENARIOS)
    by_name = {s.name: s for s in ALL_SCENARIOS}
    return [by_name[n] for n in names if n in by_name]
