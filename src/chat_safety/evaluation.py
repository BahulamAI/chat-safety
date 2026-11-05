"""Offline policy metrics over human-labeled, non-sensitive cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .models import Action, AssessmentScope, Category, SafetyDecision
from .policy import PolicyBundle


@dataclass(frozen=True)
class LabeledCase:
    request_id: str
    expected_action: Action
    severe: bool = False
    slice_name: str = "all"
    expected_categories: tuple[Category, ...] = ()


@dataclass(frozen=True)
class EvaluationReport:
    total: int
    exact_matches: int
    severe_misses: int
    benign_false_refusals: int
    review_count: int

    @property
    def exact_match_rate(self) -> float:
        return self.exact_matches / self.total

    @property
    def review_rate(self) -> float:
        return self.review_count / self.total


def evaluate_decisions(
    cases: Sequence[LabeledCase], decisions: Mapping[str, SafetyDecision]
) -> EvaluationReport:
    if not cases:
        raise ValueError("evaluation requires at least one labeled case")
    ids = [case.request_id for case in cases]
    if len(set(ids)) != len(ids) or set(ids) != set(decisions):
        raise ValueError("case and decision IDs must match exactly and be unique")
    observed = [(case, decisions[case.request_id].recommended_action) for case in cases]
    return EvaluationReport(
        total=len(cases),
        exact_matches=sum(actual is case.expected_action for case, actual in observed),
        severe_misses=sum(case.severe and case.expected_action is Action.REFUSE
                          and actual in {Action.ALLOW, Action.CONSTRAIN}
                          for case, actual in observed),
        benign_false_refusals=sum(case.expected_action is Action.ALLOW
                                  and actual is Action.REFUSE
                                  for case, actual in observed),
        review_count=sum(actual is Action.REVIEW for _, actual in observed),
    )


def evaluate_categories(
    cases: Sequence[LabeledCase],
    decisions: Mapping[str, SafetyDecision],
    policy: PolicyBundle,
) -> dict[str, dict[str, int | float | None]]:
    """Measure category detection on non-degraded model decisions only."""
    if len({case.request_id for case in cases}) != len(cases) or {
        case.request_id for case in cases
    } != set(decisions):
        raise ValueError("case and decision IDs must match exactly and be unique")
    result = {}
    eligible = [case for case in cases if not decisions[case.request_id].degraded
                and decisions[case.request_id].signals]
    for category in sorted(policy.required_categories | policy.optional_categories,
                           key=lambda item: item.value):
        true_positive = false_positive = false_negative = support = predicted = 0
        for case in eligible:
            decision = decisions[case.request_id]
            if (category in policy.optional_categories
                    and not any(signal.category is category for signal in decision.signals)):
                continue
            expected = category in case.expected_categories
            detected = any(signal.category is category
                           and signal.probability >= (
                               policy.session_thresholds if signal.scope is AssessmentScope.SESSION
                               else policy.thresholds
                           )[decision.stage][category].review_at
                           for signal in decision.signals)
            support += int(expected)
            predicted += int(detected)
            true_positive += int(expected and detected)
            false_positive += int(not expected and detected)
            false_negative += int(expected and not detected)
        result[category.value] = {
            "support": support,
            "predicted": predicted,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": round(true_positive / predicted, 4) if predicted else None,
            "recall": round(true_positive / support, 4) if support else None,
        }
    return result
