"""Offline JSONL replay of recorded category signals through SafetyGate."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from .engine import SafetyGate
from .config import load_policy_json
from .evaluation import LabeledCase, evaluate_categories, evaluate_decisions
from .models import (
    Action, AssessmentScope, Category, Evaluation, Mode, RecentTurn, RiskTier, SafetyInput,
    ScannerStatus, Signal, Stage,
)
from .policy import PolicyBundle, ThresholdBand


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _parse_bands(raw: dict[str, Any]) -> dict[Stage, dict[Category, ThresholdBand]]:
    return {
        Stage(stage): {
            Category(category): ThresholdBand(
                review_at=float(spec["review_at"]),
                refuse_at=float(spec["refuse_at"]),
                middle_action=Action(spec.get("middle_action", "review")),
                upper_action=Action(spec.get("upper_action", "refuse")),
            )
            for category, spec in categories.items()
        }
        for stage, categories in raw.items()
    }


def _load_policy(path: Path) -> PolicyBundle:
    raw = _read_object(path)
    if "schema" in raw:
        return load_policy_json(path).policy
    required = frozenset(Category(value) for value in raw["required_categories"])
    return PolicyBundle(
        version=str(raw["version"]),
        mode=Mode(raw.get("mode", "enforce")),
        thresholds=_parse_bands(raw["thresholds"]),
        required_categories=required,
        session_thresholds=_parse_bands(raw.get("session_thresholds", {})),
        optional_categories=frozenset(Category(value) for value in raw.get("optional_categories", [])),
    )


def _load_cases(path: Path, policy: PolicyBundle):
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        request_id = str(raw["id"])
        request = SafetyInput(
            request_id=request_id,
            policy_version=policy.version,
            stage=Stage(raw.get("stage", "input")),
            surface=str(raw.get("surface", "offline-evaluation")),
            actor_ref="offline-evaluation",
            content=str(raw["content"]),
            recent_turns=tuple(RecentTurn(turn["role"], turn["text"])
                               for turn in raw.get("recent_turns", [])),
            safety_summary=str(raw.get("safety_summary", "")),
            risk_tier=RiskTier(raw.get("risk_tier", "low")),
            attachment_kinds=tuple(raw.get("attachment_kinds", [])),
            scanner_status=ScannerStatus(raw.get("scanner_status", "not_applicable")),
            assessment_scopes=tuple(AssessmentScope(value) for value in
                                    raw.get("assessment_scopes", ["message"])),
            focus_categories=tuple(Category(value) for value in
                                   raw.get("focus_categories", [])),
            session_ref=raw.get("session_ref"),
            session_revision=raw.get("session_revision"),
        )
        label = LabeledCase(
            request_id=request_id,
            expected_action=Action(raw["expected_action"]),
            severe=raw.get("severe", False),
            slice_name=str(raw.get("slice", "all")),
            expected_categories=tuple(Category(value)
                                      for value in raw.get("expected_categories", [])),
        )
        if not isinstance(label.severe, bool):
            raise ValueError(f"{path}:{line_number}: severe must be a boolean")
        scores = raw.get("signals")
        if scores is not None and not isinstance(scores, dict):
            raise ValueError(f"{path}:{line_number}: signals must be an object")
        session_scores = raw.get("session_signals")
        if session_scores is not None and not isinstance(session_scores, dict):
            raise ValueError(f"{path}:{line_number}: session_signals must be an object")
        rows.append((request, label, (scores, session_scores)))
    if not rows:
        raise ValueError("evaluation dataset is empty")
    if len({request.request_id for request, _, _ in rows}) != len(rows):
        raise ValueError("evaluation case IDs must be unique")
    return rows


class _RecordedEvaluator:
    def __init__(self, scores: tuple[dict[str, Any] | None, dict[str, Any] | None]) -> None:
        self.scores = scores

    async def evaluate(self, request: SafetyInput) -> Evaluation:
        message_scores, session_scores = self.scores
        if message_scores is None:
            raise RuntimeError("recorded evaluator unavailable")
        signals = tuple(Signal(Category(category), float(probability))
                        for category, probability in message_scores.items())
        if AssessmentScope.SESSION in request.assessment_scopes and session_scores is not None:
            signals += tuple(
                Signal(Category(category), float(probability), scope=AssessmentScope.SESSION)
                for category, probability in session_scores.items()
            )
        return Evaluation(signals, evaluator="recorded", model_version="recorded/1")


async def _replay(policy: PolicyBundle, rows):
    decisions = {}
    for request, _, scores in rows:
        decisions[request.request_id] = await SafetyGate(
            policy, _RecordedEvaluator(scores)
        ).evaluate(request)
    return decisions


def _report(policy: PolicyBundle, rows, decisions, *, list_decisions: bool = False) -> dict[str, Any]:
    labels = [label for _, label, _ in rows]
    overall = evaluate_decisions(labels, decisions)
    slices = sorted({label.slice_name for label in labels})

    def summary(report):
        return {
            "total": report.total,
            "exact_matches": report.exact_matches,
            "severe_misses": report.severe_misses,
            "benign_false_refusals": report.benign_false_refusals,
            "review_count": report.review_count,
            "review_rate": round(report.review_rate, 4),
        }

    result = {
        "policy_version": policy.version,
        "mode": policy.mode.value,
        "overall": summary(overall),
        "slices": {
            name: summary(evaluate_decisions(
                [label for label in labels if label.slice_name == name],
                {label.request_id: decisions[label.request_id]
                 for label in labels if label.slice_name == name},
            ))
            for name in slices
        },
        "degraded_count": sum(decision.degraded for decision in decisions.values()),
        "category_metrics": evaluate_categories(labels, decisions, policy),
    }
    if list_decisions:
        result["case_decisions"] = [
            {
                "id": request.request_id,
                "slice": label.slice_name,
                "expected": label.expected_action.value,
                "recommended": decisions[request.request_id].recommended_action.value,
                "reason_code": decisions[request.request_id].reason_code,
                "degraded": decisions[request.request_id].degraded,
            }
            for request, label, _ in rows
        ]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True, help="Versioned threshold JSON")
    parser.add_argument("--cases", type=Path, required=True, help="Labeled JSONL with recorded signals")
    parser.add_argument("--list-decisions", action="store_true",
                        help="Show case IDs and actions, never case text")
    parser.add_argument("--fail-on-mismatch", action="store_true",
                        help="Exit nonzero if any action differs from its label")
    args = parser.parse_args(argv)
    policy = _load_policy(args.policy)
    rows = _load_cases(args.cases, policy)
    decisions = asyncio.run(_replay(policy, rows))
    report = _report(policy, rows, decisions, list_decisions=args.list_decisions)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.fail_on_mismatch and report["overall"]["exact_matches"] != len(rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
