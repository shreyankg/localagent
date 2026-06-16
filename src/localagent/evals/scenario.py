"""Eval scenario definitions — the contract every skill eval must follow."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvalScore:
    """Score for a single eval scenario."""

    scenario_name: str
    passed: bool
    score: float  # 0.0 – 1.0
    max_score: float  # typically 1.0
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def percentage(self) -> float:
        return (self.score / self.max_score * 100) if self.max_score > 0 else 0.0


@dataclass
class EvalResult:
    """Aggregated result across all scenarios for one model."""

    model_name: str
    skill_name: str
    scores: list[EvalScore] = field(default_factory=list)
    total_time_seconds: float = 0.0

    @property
    def total_score(self) -> float:
        return sum(s.score for s in self.scores)

    @property
    def max_total_score(self) -> float:
        return sum(s.max_score for s in self.scores)

    @property
    def percentage(self) -> float:
        return (self.total_score / self.max_total_score * 100) if self.max_total_score > 0 else 0.0

    @property
    def passed(self) -> int:
        return sum(1 for s in self.scores if s.passed)

    @property
    def failed(self) -> int:
        return sum(1 for s in self.scores if not s.passed)


class EvalScenario(abc.ABC):
    """Abstract base for a single eval scenario.

    A scenario defines:
    - A set of input files (virtual, no real filesystem needed)
    - Expected categorization behavior
    - A scoring function
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique name for this scenario."""
        ...

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Human-readable description of what this scenario tests."""
        ...

    @abc.abstractmethod
    def get_file_profiles(self) -> list[dict[str, Any]]:
        """Return synthetic file profiles for the LLM to categorize.

        Each dict should match the FileProfile.to_summary() format:
        {"name": ..., "extension": ..., "mime_type": ..., "size": ..., "content_preview": ...}
        """
        ...

    @abc.abstractmethod
    def score(self, result: dict[str, Any]) -> EvalScore:
        """Score the LLM's categorization output.

        ``result`` has the shape:
        {"taxonomy": {"cat": "desc", ...}, "assignments": {"file": "cat", ...}}

        Returns an EvalScore.
        """
        ...
