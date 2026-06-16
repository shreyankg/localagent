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


# ── Real-world scenarios ────────────────────────────────────────────────────
# Based on representative file patterns from actual Documents/Downloads usage.


class DownloadsChaos(EvalScenario):
    """Can the model tame a messy Downloads folder with duplicates, installers, and mixed junk?

    Modeled on real ~/Downloads patterns: duplicate browser downloads with (1) suffixes,
    macOS DMGs, scattered PDFs, WhatsApp images, and random data files.
    """

    @property
    def name(self) -> str:
        return "downloads-chaos"

    @property
    def description(self) -> str:
        return "Organize a realistic messy Downloads folder with duplicates, installers, and mixed files"

    def get_file_profiles(self) -> list[dict[str, Any]]:
        return [
            # Duplicate browser downloads
            {"name": "invoice_march_2024.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "145 KB"},
            {"name": "invoice_march_2024 (1).pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "145 KB"},
            {"name": "invoice_march_2024 (2).pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "145 KB"},
            # macOS installers
            {"name": "Obsidian-1.5.8-universal.dmg", "extension": ".dmg", "mime_type": "application/x-apple-diskimage", "size": "165 MB"},
            {"name": "AnyDesk.dmg", "extension": ".dmg", "mime_type": "application/x-apple-diskimage", "size": "12 MB"},
            {"name": "KalturaCapture-3.29.12.dmg", "extension": ".dmg", "mime_type": "application/x-apple-diskimage", "size": "89 MB"},
            # Resume from job application
            {"name": "2476207_UmerQayam_JM_3184289_Resume.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "380 KB"},
            # WhatsApp image dump
            {"name": "WhatsApp Image 2024-12-09 at 19.42.33.jpeg", "extension": ".jpeg", "mime_type": "image/jpeg", "size": "2.1 MB"},
            {"name": "WhatsApp Image 2024-12-09 at 19.42.34.jpeg", "extension": ".jpeg", "mime_type": "image/jpeg", "size": "1.8 MB"},
            {"name": "WhatsApp Image 2024-12-10 at 10.15.22.jpeg", "extension": ".jpeg", "mime_type": "image/jpeg", "size": "3.2 MB"},
            # Data files
            {"name": "maintenance-charges-2024.csv", "extension": ".csv", "mime_type": "text/csv", "size": "28 KB", "content_preview": "flat_no,owner,month,amount,status\nA-101,Sharma,Jan,4500,paid\nA-102,Patel,Jan,4500,pending"},
            {"name": "credit_card_statement_dec.csv", "extension": ".csv", "mime_type": "text/csv", "size": "65 KB", "content_preview": "date,merchant,amount,category\n2024-12-01,Swiggy,450.00,Food\n2024-12-02,Amazon,2300.00,Shopping"},
            # Random downloaded archive
            {"name": "fonts-collection.zip", "extension": ".zip", "mime_type": "application/zip", "size": "45 MB"},
            # Screenshot from browser
            {"name": "Screenshot From 2025-07-10 15-39-27.png", "extension": ".png", "mime_type": "image/png", "size": "1.4 MB"},
            # Video recording
            {"name": "2025-06-09 10-04-25.mkv", "extension": ".mkv", "mime_type": "video/x-matroska", "size": "850 MB"},
        ]

    def score(self, result: dict[str, Any]) -> EvalScore:
        taxonomy = result.get("taxonomy", {})
        assignments = result.get("assignments", {})
        expected_files = {p["name"] for p in self.get_file_profiles()}

        coverage_score, coverage_details = _check_all_assigned(assignments, expected_files)
        taxonomy_score, taxonomy_details = _check_taxonomy_size(taxonomy, min_cats=4, max_cats=10)

        # Key groupings
        grouping_score, grouping_details = _check_grouping(
            assignments,
            [
                # Duplicate downloads should stay together
                {"invoice_march_2024.pdf", "invoice_march_2024 (1).pdf", "invoice_march_2024 (2).pdf"},
                # Installers should be grouped
                {"Obsidian-1.5.8-universal.dmg", "AnyDesk.dmg", "KalturaCapture-3.29.12.dmg"},
                # WhatsApp images together
                {
                    "WhatsApp Image 2024-12-09 at 19.42.33.jpeg",
                    "WhatsApp Image 2024-12-09 at 19.42.34.jpeg",
                    "WhatsApp Image 2024-12-10 at 10.15.22.jpeg",
                },
            ],
        )

        # Key separations
        sep_score, sep_details = _check_separation(
            assignments,
            [
                ("Obsidian-1.5.8-universal.dmg", "invoice_march_2024.pdf"),
                ("WhatsApp Image 2024-12-09 at 19.42.33.jpeg", "maintenance-charges-2024.csv"),
                ("2476207_UmerQayam_JM_3184289_Resume.pdf", "AnyDesk.dmg"),
            ],
        )

        total = (coverage_score * 0.2) + (taxonomy_score * 0.15) + (grouping_score * 0.4) + (sep_score * 0.25)
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


class PersonalFinanceDocs(EvalScenario):
    """Can the model intelligently separate financial documents by purpose?

    Tests whether the model can distinguish bills, pay stubs, tax documents,
    rent receipts, and credit card statements — all PDFs with varying naming
    conventions, as found in a real Indian professional's Downloads.
    """

    @property
    def name(self) -> str:
        return "personal-finance"

    @property
    def description(self) -> str:
        return "Organize financial documents: payslips, bills, tax forms, rent receipts, statements"

    def get_file_profiles(self) -> list[dict[str, Any]]:
        return [
            # Payslips
            {"name": "Payslip_2025-01-31.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "95 KB"},
            {"name": "Payslip_2025-02-28.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "97 KB"},
            {"name": "Payslip_2025-03-31.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "94 KB"},
            # Telecom bills
            {"name": "VIL_bill_9604587789_2024-08-14.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "220 KB"},
            {"name": "Jio_bill_sept_2024.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "180 KB"},
            # Internet bill
            {"name": "ACT_Fibernet_Invoice_Oct2024.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "150 KB"},
            # Rent receipts
            {"name": "rent-receipt-april-june-2024.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "85 KB"},
            {"name": "rent-receipt-july-sept-2024.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "82 KB"},
            # Tax documents
            {"name": "Form 16 - Part A.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "350 KB"},
            {"name": "Form 16 - Part B.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "280 KB"},
            {"name": "AIS_Annual_Information_Statement_2024.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "1.2 MB"},
            # Credit card statement
            {"name": "HDFC_CC_Statement_Nov2024.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "410 KB"},
            # Insurance
            {"name": "health_insurance_policy_2024.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "2.5 MB"},
            # Investment
            {"name": "MutualFund_CAS_Statement_Dec2024.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "320 KB"},
        ]

    def score(self, result: dict[str, Any]) -> EvalScore:
        taxonomy = result.get("taxonomy", {})
        assignments = result.get("assignments", {})
        expected_files = {p["name"] for p in self.get_file_profiles()}

        coverage_score, coverage_details = _check_all_assigned(assignments, expected_files)
        taxonomy_score, taxonomy_details = _check_taxonomy_size(taxonomy, min_cats=4, max_cats=10)

        # Key groupings — model must understand financial subcategories
        grouping_score, grouping_details = _check_grouping(
            assignments,
            [
                # Payslips together
                {"Payslip_2025-01-31.pdf", "Payslip_2025-02-28.pdf", "Payslip_2025-03-31.pdf"},
                # Telecom/internet bills together
                {"VIL_bill_9604587789_2024-08-14.pdf", "Jio_bill_sept_2024.pdf", "ACT_Fibernet_Invoice_Oct2024.pdf"},
                # Rent receipts together
                {"rent-receipt-april-june-2024.pdf", "rent-receipt-july-sept-2024.pdf"},
                # Tax documents together
                {"Form 16 - Part A.pdf", "Form 16 - Part B.pdf", "AIS_Annual_Information_Statement_2024.pdf"},
            ],
        )

        # Key separations — don't lump everything into "Documents"
        sep_score, sep_details = _check_separation(
            assignments,
            [
                ("Payslip_2025-01-31.pdf", "VIL_bill_9604587789_2024-08-14.pdf"),
                ("Form 16 - Part A.pdf", "rent-receipt-april-june-2024.pdf"),
                ("health_insurance_policy_2024.pdf", "HDFC_CC_Statement_Nov2024.pdf"),
                ("MutualFund_CAS_Statement_Dec2024.pdf", "Jio_bill_sept_2024.pdf"),
            ],
        )

        total = (coverage_score * 0.2) + (taxonomy_score * 0.1) + (grouping_score * 0.45) + (sep_score * 0.25)
        return EvalScore(
            scenario_name=self.name,
            passed=total >= 0.45,
            score=round(total, 3),
            max_score=1.0,
            details={
                "coverage": coverage_details,
                "taxonomy": taxonomy_details,
                "grouping": grouping_details,
                "separation": sep_details,
            },
        )


class ScreenshotOverload(EvalScenario):
    """Can the model handle a folder dominated by screenshots from multiple platforms?

    Tests whether the model can recognize screenshot naming conventions from macOS,
    Linux, and iOS, and group them separately from actual photos and documents.
    """

    @property
    def name(self) -> str:
        return "screenshot-overload"

    @property
    def description(self) -> str:
        return "Sort screenshots from different OSes alongside real photos and documents"

    def get_file_profiles(self) -> list[dict[str, Any]]:
        return [
            # macOS screenshots (different years)
            {"name": "Screenshot 2024-03-15 at 2.30.22 PM.png", "extension": ".png", "mime_type": "image/png", "size": "1.8 MB"},
            {"name": "Screenshot 2024-11-20 at 10.15.44 AM.png", "extension": ".png", "mime_type": "image/png", "size": "2.1 MB"},
            {"name": "Screenshot 2025-01-08 at 4.52.11 PM.png", "extension": ".png", "mime_type": "image/png", "size": "1.5 MB"},
            # Linux screenshots
            {"name": "Screenshot From 2025-07-10 15-39-27.png", "extension": ".png", "mime_type": "image/png", "size": "1.4 MB"},
            {"name": "Screenshot From 2025-06-22 09-12-03.png", "extension": ".png", "mime_type": "image/png", "size": "1.9 MB"},
            # iOS photo naming
            {"name": "IMG_4523.HEIC", "extension": ".HEIC", "mime_type": "image/heic", "size": "3.5 MB"},
            {"name": "IMG_4524.HEIC", "extension": ".HEIC", "mime_type": "image/heic", "size": "4.1 MB"},
            {"name": "IMG_4525.HEIC", "extension": ".HEIC", "mime_type": "image/heic", "size": "3.8 MB"},
            # Actual named photos
            {"name": "family_dinner_diwali_2024.jpg", "extension": ".jpg", "mime_type": "image/jpeg", "size": "5.2 MB"},
            {"name": "office_team_photo.jpg", "extension": ".jpg", "mime_type": "image/jpeg", "size": "4.8 MB"},
            # A scanned document (image but really a document)
            {"name": "passport_scan_page1.png", "extension": ".png", "mime_type": "image/png", "size": "2.8 MB"},
            {"name": "aadhaar_card_scan.png", "extension": ".png", "mime_type": "image/png", "size": "1.9 MB"},
            # Design asset (image but for work)
            {"name": "logo_final_v3.svg", "extension": ".svg", "mime_type": "image/svg+xml", "size": "45 KB"},
            {"name": "banner_homepage.png", "extension": ".png", "mime_type": "image/png", "size": "850 KB"},
        ]

    def score(self, result: dict[str, Any]) -> EvalScore:
        taxonomy = result.get("taxonomy", {})
        assignments = result.get("assignments", {})
        expected_files = {p["name"] for p in self.get_file_profiles()}

        coverage_score, coverage_details = _check_all_assigned(assignments, expected_files)
        taxonomy_score, taxonomy_details = _check_taxonomy_size(taxonomy, min_cats=3, max_cats=8)

        # Screenshots from both OSes should be grouped
        grouping_score, grouping_details = _check_grouping(
            assignments,
            [
                # All screenshots together (regardless of OS)
                {
                    "Screenshot 2024-03-15 at 2.30.22 PM.png",
                    "Screenshot 2024-11-20 at 10.15.44 AM.png",
                    "Screenshot 2025-01-08 at 4.52.11 PM.png",
                    "Screenshot From 2025-07-10 15-39-27.png",
                    "Screenshot From 2025-06-22 09-12-03.png",
                },
                # iOS photos together
                {"IMG_4523.HEIC", "IMG_4524.HEIC", "IMG_4525.HEIC"},
                # Scanned ID documents together
                {"passport_scan_page1.png", "aadhaar_card_scan.png"},
            ],
        )

        # Key separations
        sep_score, sep_details = _check_separation(
            assignments,
            [
                # Screenshots should not be mixed with real photos
                ("Screenshot 2024-03-15 at 2.30.22 PM.png", "family_dinner_diwali_2024.jpg"),
                # Scanned documents should not be mixed with screenshots
                ("passport_scan_page1.png", "Screenshot 2024-03-15 at 2.30.22 PM.png"),
                # Design assets should not be mixed with personal photos
                ("logo_final_v3.svg", "family_dinner_diwali_2024.jpg"),
            ],
        )

        total = (coverage_score * 0.2) + (taxonomy_score * 0.1) + (grouping_score * 0.4) + (sep_score * 0.3)
        return EvalScore(
            scenario_name=self.name,
            passed=total >= 0.45,
            score=round(total, 3),
            max_score=1.0,
            details={
                "coverage": coverage_details,
                "taxonomy": taxonomy_details,
                "grouping": grouping_details,
                "separation": sep_details,
            },
        )


class WorkPersonalBlend(EvalScenario):
    """Can the model separate work and personal files that are interleaved?

    A realistic mix of meeting recordings, company onboarding docs, personal
    travel plans, resumes, and hobby projects all in one folder.
    """

    @property
    def name(self) -> str:
        return "work-personal-blend"

    @property
    def description(self) -> str:
        return "Separate interleaved work and personal files in a shared folder"

    def get_file_profiles(self) -> list[dict[str, Any]]:
        return [
            # Work — meeting recordings
            {"name": "zoom_recording_standup_2025-01-15.mp4", "extension": ".mp4", "mime_type": "video/mp4", "size": "120 MB"},
            {"name": "GMT20250120-143022_Recording.m4a", "extension": ".m4a", "mime_type": "audio/mp4", "size": "35 MB"},
            # Work — onboarding / company docs
            {"name": "Employee_Handbook_2025.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "4.5 MB"},
            {"name": "Q4_OKRs_Engineering.pptx", "extension": ".pptx", "mime_type": "application/vnd.openxmlformats-officedocument.presentationml.presentation", "size": "6.2 MB"},
            {"name": "sprint_retro_notes.md", "extension": ".md", "mime_type": "text/markdown", "size": "3 KB", "content_preview": "# Sprint 42 Retro\n\n## What went well\n- Shipped auth migration on time\n- Zero downtime deployment\n\n## What to improve\n- Better test coverage on edge cases"},
            # Work — code review
            {"name": "PR_review_feedback.txt", "extension": ".txt", "mime_type": "text/plain", "size": "2 KB", "content_preview": "PR #1847 - API rate limiting\n\nComments:\n- Consider using token bucket algorithm\n- Missing unit tests for edge cases\n- LGTM on the middleware approach"},
            # Personal — travel
            {"name": "Brno_trip_itinerary_2025.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "1.2 MB"},
            {"name": "flight_booking_DEL_PRG.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "350 KB"},
            {"name": "hotel_confirmation_hilton.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "280 KB"},
            # Personal — resume and career
            {"name": "Gupta_Shreyank_Resume_2025.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "200 KB"},
            {"name": "cover_letter_google.docx", "extension": ".docx", "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "size": "45 KB"},
            # Personal — hobby project
            {"name": "raspberry_pi_home_server.md", "extension": ".md", "mime_type": "text/markdown", "size": "5 KB", "content_preview": "# Home Server Setup\n\n## Parts list\n- Raspberry Pi 5 8GB\n- 1TB NVMe SSD\n- Argon ONE case\n\n## Software\n- Pi-hole for DNS\n- Nextcloud for file sync"},
            # Ambiguous — could be either
            {"name": "meeting_agenda_monday.txt", "extension": ".txt", "mime_type": "text/plain", "size": "800 B", "content_preview": "Monday Sync\n1. Review PRs from last week\n2. Discuss production incident\n3. Plan for next sprint"},
        ]

    def score(self, result: dict[str, Any]) -> EvalScore:
        taxonomy = result.get("taxonomy", {})
        assignments = result.get("assignments", {})
        expected_files = {p["name"] for p in self.get_file_profiles()}

        coverage_score, coverage_details = _check_all_assigned(assignments, expected_files)
        taxonomy_score, taxonomy_details = _check_taxonomy_size(taxonomy, min_cats=4, max_cats=10)

        grouping_score, grouping_details = _check_grouping(
            assignments,
            [
                # Travel documents together
                {"Brno_trip_itinerary_2025.pdf", "flight_booking_DEL_PRG.pdf", "hotel_confirmation_hilton.pdf"},
                # Career documents together
                {"Gupta_Shreyank_Resume_2025.pdf", "cover_letter_google.docx"},
                # Meeting recordings together
                {"zoom_recording_standup_2025-01-15.mp4", "GMT20250120-143022_Recording.m4a"},
            ],
        )

        sep_score, sep_details = _check_separation(
            assignments,
            [
                # Work docs should not mix with travel
                ("Employee_Handbook_2025.pdf", "Brno_trip_itinerary_2025.pdf"),
                # Resume should not mix with work docs
                ("Gupta_Shreyank_Resume_2025.pdf", "Q4_OKRs_Engineering.pptx"),
                # Hobby project should not mix with work notes
                ("raspberry_pi_home_server.md", "sprint_retro_notes.md"),
                # Travel should not mix with career
                ("flight_booking_DEL_PRG.pdf", "cover_letter_google.docx"),
            ],
        )

        total = (coverage_score * 0.2) + (taxonomy_score * 0.1) + (grouping_score * 0.4) + (sep_score * 0.3)
        return EvalScore(
            scenario_name=self.name,
            passed=total >= 0.45,
            score=round(total, 3),
            max_score=1.0,
            details={
                "coverage": coverage_details,
                "taxonomy": taxonomy_details,
                "grouping": grouping_details,
                "separation": sep_details,
            },
        )


class IdentityDocuments(EvalScenario):
    """Can the model recognize and group identity/government documents?

    A mix of scanned IDs, visa paperwork, Aadhaar downloads, and personal
    certificates alongside unrelated files. Tests whether the model understands
    the sensitivity and purpose of identity documents.
    """

    @property
    def name(self) -> str:
        return "identity-documents"

    @property
    def description(self) -> str:
        return "Identify and group government IDs, visa docs, and certificates among other files"

    def get_file_profiles(self) -> list[dict[str, Any]]:
        return [
            # Government ID downloads
            {"name": "EAadhaar_952783784057.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "1.1 MB"},
            {"name": "PAN_Card_ABCPG1234H.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "450 KB"},
            # Visa documents
            {"name": "Schengen_Visa_Application.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "2.3 MB"},
            {"name": "visa_appointment_confirmation.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "180 KB"},
            {"name": "passport_photo_35x45mm.jpg", "extension": ".jpg", "mime_type": "image/jpeg", "size": "250 KB"},
            # Education certificates
            {"name": "BTech_Degree_Certificate.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "3.8 MB"},
            {"name": "semester8_marksheet.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "1.5 MB"},
            # Unrelated files mixed in
            {"name": "grocery_list.txt", "extension": ".txt", "mime_type": "text/plain", "size": "200 B", "content_preview": "Weekly groceries:\n- Rice 5kg\n- Dal 2kg\n- Onions 2kg\n- Tomatoes 1kg"},
            {"name": "workout_plan.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "800 KB"},
            {"name": "spotify_wrapped_2024.png", "extension": ".png", "mime_type": "image/png", "size": "1.2 MB"},
            {"name": "recipe_butter_chicken.txt", "extension": ".txt", "mime_type": "text/plain", "size": "1.5 KB", "content_preview": "Butter Chicken Recipe\n\nIngredients:\n- 500g chicken thighs\n- 2 cups tomato puree\n- 1 cup cream\n- Spices: garam masala, turmeric, chili"},
            {"name": "apartment_lease_2024.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "5.2 MB"},
        ]

    def score(self, result: dict[str, Any]) -> EvalScore:
        taxonomy = result.get("taxonomy", {})
        assignments = result.get("assignments", {})
        expected_files = {p["name"] for p in self.get_file_profiles()}

        coverage_score, coverage_details = _check_all_assigned(assignments, expected_files)
        taxonomy_score, taxonomy_details = _check_taxonomy_size(taxonomy, min_cats=4, max_cats=10)

        grouping_score, grouping_details = _check_grouping(
            assignments,
            [
                # Government IDs together
                {"EAadhaar_952783784057.pdf", "PAN_Card_ABCPG1234H.pdf"},
                # Visa documents together
                {"Schengen_Visa_Application.pdf", "visa_appointment_confirmation.pdf", "passport_photo_35x45mm.jpg"},
                # Education certificates together
                {"BTech_Degree_Certificate.pdf", "semester8_marksheet.pdf"},
            ],
        )

        sep_score, sep_details = _check_separation(
            assignments,
            [
                # IDs separate from groceries
                ("EAadhaar_952783784057.pdf", "grocery_list.txt"),
                # Visa docs separate from recipes
                ("Schengen_Visa_Application.pdf", "recipe_butter_chicken.txt"),
                # Education certs separate from workout
                ("BTech_Degree_Certificate.pdf", "workout_plan.pdf"),
                # Passport photo separate from spotify screenshot
                ("passport_photo_35x45mm.jpg", "spotify_wrapped_2024.png"),
            ],
        )

        total = (coverage_score * 0.2) + (taxonomy_score * 0.1) + (grouping_score * 0.4) + (sep_score * 0.3)
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


class MixedMediaProject(EvalScenario):
    """Can the model organize a folder with a mix of project assets and personal media?

    Simulates a creative project folder where design files, exported assets,
    reference materials, and meeting notes are all mixed together — a common
    pattern for freelancers and side projects.
    """

    @property
    def name(self) -> str:
        return "mixed-media-project"

    @property
    def description(self) -> str:
        return "Organize a mixed bag of design files, exports, references, and project docs"

    def get_file_profiles(self) -> list[dict[str, Any]]:
        return [
            # Design source files
            {"name": "app_redesign_v2.fig", "extension": ".fig", "mime_type": "application/octet-stream", "size": "12 MB"},
            {"name": "brand_guidelines.sketch", "extension": ".sketch", "mime_type": "application/octet-stream", "size": "8.5 MB"},
            {"name": "icon_set_draft.xcf", "extension": ".xcf", "mime_type": "image/x-xcf", "size": "15 MB"},
            # Exported assets
            {"name": "logo_dark_512x512.png", "extension": ".png", "mime_type": "image/png", "size": "45 KB"},
            {"name": "logo_light_512x512.png", "extension": ".png", "mime_type": "image/png", "size": "42 KB"},
            {"name": "hero_banner_1920x1080.jpg", "extension": ".jpg", "mime_type": "image/jpeg", "size": "320 KB"},
            {"name": "app_icon.svg", "extension": ".svg", "mime_type": "image/svg+xml", "size": "8 KB"},
            # Reference / inspiration
            {"name": "competitor_analysis.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "2.1 MB"},
            {"name": "color_palette_reference.png", "extension": ".png", "mime_type": "image/png", "size": "180 KB"},
            # Project management
            {"name": "project_timeline.xlsx", "extension": ".xlsx", "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "size": "65 KB"},
            {"name": "client_feedback_round2.docx", "extension": ".docx", "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "size": "120 KB"},
            {"name": "meeting_notes_kickoff.md", "extension": ".md", "mime_type": "text/markdown", "size": "2 KB", "content_preview": "# Project Kickoff\n\nClient: FreshCart\nTimeline: 6 weeks\nScope: Full app redesign\n\n## Deliverables\n- New UI kit\n- 5 key screen mockups\n- Prototype"},
            # Deliverable
            {"name": "final_presentation_client.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "18 MB"},
            # Font files
            {"name": "Inter-Regular.woff2", "extension": ".woff2", "mime_type": "font/woff2", "size": "98 KB"},
            {"name": "Inter-Bold.woff2", "extension": ".woff2", "mime_type": "font/woff2", "size": "102 KB"},
        ]

    def score(self, result: dict[str, Any]) -> EvalScore:
        taxonomy = result.get("taxonomy", {})
        assignments = result.get("assignments", {})
        expected_files = {p["name"] for p in self.get_file_profiles()}

        coverage_score, coverage_details = _check_all_assigned(assignments, expected_files)
        taxonomy_score, taxonomy_details = _check_taxonomy_size(taxonomy, min_cats=4, max_cats=10)

        grouping_score, grouping_details = _check_grouping(
            assignments,
            [
                # Design source files together
                {"app_redesign_v2.fig", "brand_guidelines.sketch", "icon_set_draft.xcf"},
                # Exported logo assets together
                {"logo_dark_512x512.png", "logo_light_512x512.png", "app_icon.svg"},
                # Font files together
                {"Inter-Regular.woff2", "Inter-Bold.woff2"},
                # Project docs together
                {"project_timeline.xlsx", "client_feedback_round2.docx", "meeting_notes_kickoff.md"},
            ],
        )

        sep_score, sep_details = _check_separation(
            assignments,
            [
                # Source files should not be mixed with exported assets
                ("app_redesign_v2.fig", "logo_dark_512x512.png"),
                # Fonts should not be mixed with design source files
                ("Inter-Regular.woff2", "brand_guidelines.sketch"),
                # Reference should not be mixed with project docs
                ("color_palette_reference.png", "project_timeline.xlsx"),
            ],
        )

        total = (coverage_score * 0.2) + (taxonomy_score * 0.1) + (grouping_score * 0.4) + (sep_score * 0.3)
        return EvalScore(
            scenario_name=self.name,
            passed=total >= 0.45,
            score=round(total, 3),
            max_score=1.0,
            details={
                "coverage": coverage_details,
                "taxonomy": taxonomy_details,
                "grouping": grouping_details,
                "separation": sep_details,
            },
        )


class MultilingualFilenames(EvalScenario):
    """Can the model handle files with non-ASCII characters and mixed naming conventions?

    Tests robustness with filenames containing accented characters, Devanagari,
    CJK characters, and emoji — along with extremely long names and special characters
    commonly generated by scanners, cameras, and web downloads.
    """

    @property
    def name(self) -> str:
        return "multilingual-filenames"

    @property
    def description(self) -> str:
        return "Handle non-ASCII filenames, long names, and special characters without breaking"

    def get_file_profiles(self) -> list[dict[str, Any]]:
        return [
            # Devanagari / Hindi filename
            {"name": "निवेश_रिपोर्ट_2024.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "1.2 MB"},
            # Accented European
            {"name": "Résumé_François_Müller.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "300 KB"},
            # Japanese
            {"name": "会議メモ_2024年3月.txt", "extension": ".txt", "mime_type": "text/plain", "size": "4 KB", "content_preview": "会議メモ\n参加者: 田中、鈴木\n議題: Q1の売上について"},
            # Very long scanner-generated name
            {"name": "Scan_HP_LaserJet_Pro_MFP_M428fdw_20240315_143022_001_document_high_quality.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "8.5 MB"},
            # Name with lots of special chars
            {"name": "2024 Tax Return (Final) [Reviewed] {v3}.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "2.1 MB"},
            # Standard English files for baseline
            {"name": "quarterly_report.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "1.5 MB"},
            {"name": "family_photo_christmas.jpg", "extension": ".jpg", "mime_type": "image/jpeg", "size": "4.2 MB"},
            {"name": "budget_2024.xlsx", "extension": ".xlsx", "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "size": "85 KB"},
            # Korean
            {"name": "이력서_김민수.pdf", "extension": ".pdf", "mime_type": "application/pdf", "size": "280 KB"},
            # Filename with dots (often confuses extension detection)
            {"name": "project.plan.v2.final.docx", "extension": ".docx", "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "size": "120 KB"},
            # Numeric-only name (camera export)
            {"name": "20240315_142233.CR3", "extension": ".CR3", "mime_type": "image/x-canon-cr3", "size": "25 MB"},
            {"name": "20240315_142234.CR3", "extension": ".CR3", "mime_type": "image/x-canon-cr3", "size": "24 MB"},
        ]

    def score(self, result: dict[str, Any]) -> EvalScore:
        taxonomy = result.get("taxonomy", {})
        assignments = result.get("assignments", {})
        expected_files = {p["name"] for p in self.get_file_profiles()}

        coverage_score, coverage_details = _check_all_assigned(assignments, expected_files)
        taxonomy_score, taxonomy_details = _check_taxonomy_size(taxonomy, min_cats=3, max_cats=10)

        # Camera RAW photos should be grouped
        grouping_score, grouping_details = _check_grouping(
            assignments,
            [
                {"20240315_142233.CR3", "20240315_142234.CR3"},
                # Resumes across languages should ideally group
                {"Résumé_François_Müller.pdf", "이력서_김민수.pdf"},
            ],
        )

        # Key separations
        sep_score, sep_details = _check_separation(
            assignments,
            [
                ("family_photo_christmas.jpg", "quarterly_report.pdf"),
                ("20240315_142233.CR3", "budget_2024.xlsx"),
                ("निवेश_रिपोर्ट_2024.pdf", "family_photo_christmas.jpg"),
            ],
        )

        total = (coverage_score * 0.35) + (taxonomy_score * 0.1) + (grouping_score * 0.25) + (sep_score * 0.3)
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


# ── Registry ────────────────────────────────────────────────────────────────


ALL_SCENARIOS: list[EvalScenario] = [
    # Original 5
    BasicFileTypes(),
    ContentAwareCategorization(),
    AmbiguousFiles(),
    LargeFileset(),
    SemanticNuance(),
    # Real-world scenarios (based on actual Documents/Downloads patterns)
    DownloadsChaos(),
    PersonalFinanceDocs(),
    ScreenshotOverload(),
    WorkPersonalBlend(),
    IdentityDocuments(),
    MixedMediaProject(),
    MultilingualFilenames(),
]


def get_scenarios(names: list[str] | None = None) -> list[EvalScenario]:
    """Return scenarios filtered by name, or all if names is None."""
    if names is None:
        return list(ALL_SCENARIOS)
    by_name = {s.name: s for s in ALL_SCENARIOS}
    return [by_name[n] for n in names if n in by_name]
