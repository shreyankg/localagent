"""Eval runner — executes scenarios against one or more models and collects scores."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import yaml

from localagent.core.engine import Engine
from localagent.evals.scenario import EvalResult, EvalScenario, EvalScore

logger = logging.getLogger(__name__)


# ── Prompt construction (mirrors categorizer but uses synthetic profiles) ───

_EVAL_SYSTEM_PROMPT = """\
You are a file organization assistant. You will be given a list of files with \
their names, extensions, MIME types, sizes, and (when available) content previews.

Your job:
1. Analyze the files and infer meaningful categories based on their actual \
content, purpose, and type — NOT just their extension.
2. Propose a taxonomy of folder categories that makes sense for THIS specific \
collection of files. Use clear, concise category names.
3. Assign every file to exactly one category.

Guidelines:
- Aim for 5–15 categories. Don't be too granular or too broad.
- Group related files together.
- Think about the semantic purpose of each file.
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


def _build_eval_messages(profiles: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build chat messages for an eval scenario."""
    inventory = json.dumps(profiles, indent=2)
    return [
        {"role": "system", "content": _EVAL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Here are {len(profiles)} files to organize:\n\n{inventory}",
        },
    ]


def run_scenario(
    engine: Engine,
    scenario: EvalScenario,
) -> EvalScore:
    """Run a single eval scenario against a loaded engine.

    Returns the EvalScore (never raises — captures errors in the score).
    """
    logger.info("Running scenario: %s", scenario.name)

    profiles = scenario.get_file_profiles()
    messages = _build_eval_messages(profiles)

    try:
        result = engine.generate_json(messages, max_tokens=4096)
    except Exception as exc:
        logger.error("Scenario '%s' failed: %s", scenario.name, exc)
        return EvalScore(
            scenario_name=scenario.name,
            passed=False,
            score=0.0,
            max_score=1.0,
            error=str(exc),
        )

    try:
        return scenario.score(result)
    except Exception as exc:
        logger.error("Scoring failed for '%s': %s", scenario.name, exc)
        return EvalScore(
            scenario_name=scenario.name,
            passed=False,
            score=0.0,
            max_score=1.0,
            error=f"Scoring error: {exc}",
        )


def run_eval(
    model_path: str,
    scenarios: list[EvalScenario],
    skill_name: str,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> EvalResult:
    """Run all scenarios against a single model.

    Loads the model once, runs all scenarios, returns aggregated results.
    """
    logger.info("Starting eval: model=%s, scenarios=%d", model_path, len(scenarios))

    engine = Engine(
        model_path=model_path,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    result = EvalResult(
        model_name=model_path,
        skill_name=skill_name,
    )

    start = time.time()

    for scenario in scenarios:
        t0 = time.time()
        score = run_scenario(engine, scenario)
        elapsed = time.time() - t0
        score.details["time_seconds"] = round(elapsed, 2)
        result.scores.append(score)
        logger.info(
            "  %s: %s (%.1f%%, %.1fs)",
            scenario.name,
            "PASS" if score.passed else "FAIL",
            score.percentage,
            elapsed,
        )

    result.total_time_seconds = round(time.time() - start, 2)

    logger.info(
        "Eval complete: %d/%d passed, %.1f%% overall, %.1fs total",
        result.passed,
        len(scenarios),
        result.percentage,
        result.total_time_seconds,
    )

    return result


def run_benchmark(
    model_paths: list[str],
    scenarios: list[EvalScenario],
    skill_name: str,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> list[EvalResult]:
    """Run all scenarios against multiple models for comparison.

    Returns a list of EvalResult, one per model.
    """
    results: list[EvalResult] = []

    for model_path in model_paths:
        logger.info("=" * 60)
        logger.info("Benchmarking model: %s", model_path)
        logger.info("=" * 60)

        result = run_eval(
            model_path,
            scenarios,
            skill_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        results.append(result)

    return results


# ── Results persistence ─────────────────────────────────────────────────────


def save_results(results: list[EvalResult], output_path: Path) -> Path:
    """Save benchmark results to a YAML file."""
    data = []
    for r in results:
        data.append({
            "model": r.model_name,
            "skill": r.skill_name,
            "total_score": r.total_score,
            "max_score": r.max_total_score,
            "percentage": round(r.percentage, 1),
            "passed": r.passed,
            "failed": r.failed,
            "time_seconds": r.total_time_seconds,
            "scenarios": [
                {
                    "name": s.scenario_name,
                    "passed": s.passed,
                    "score": s.score,
                    "max_score": s.max_score,
                    "percentage": round(s.percentage, 1),
                    "error": s.error,
                    "details": s.details,
                }
                for s in r.scores
            ],
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info("Results saved to %s", output_path)
    return output_path
