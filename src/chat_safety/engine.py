"""Combine hard rules, evaluator evidence, policy bands, and safe fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .context import ContextTooLarge
from .models import (
    Action, AssessmentScope, Evaluation, Mode, RiskTier, SafetyDecision, SafetyInput, ScannerStatus,
    Signal, Stage,
)
from .policy import PolicyBundle


class Evaluator(Protocol):
    async def evaluate(self, request: SafetyInput) -> Evaluation: ...


@dataclass(frozen=True)
class RuleResult:
    action: Action
    reason_code: str

    def __post_init__(self) -> None:
        if self.action is Action.ALLOW:
            raise ValueError("hard rules cannot bypass the evaluator by allowing")
        if not self.reason_code:
            raise ValueError("hard rule reason code is required")


class HardRule(Protocol):
    def check(self, request: SafetyInput) -> RuleResult | None: ...


_ACTION_PRIORITY = {
    Action.ALLOW: 0,
    Action.CONSTRAIN: 1,
    Action.REVIEW: 2,
    Action.REFUSE: 3,
}


class SafetyGate:
    def __init__(
        self,
        policy: PolicyBundle,
        evaluator: Evaluator,
        hard_rules: Sequence[HardRule] = (),
    ) -> None:
        self.policy = policy
        self.evaluator = evaluator
        self.hard_rules = tuple(hard_rules)

    async def evaluate(self, request: SafetyInput) -> SafetyDecision:
        if request.policy_version != self.policy.version:
            return self._decision(request, Action.REVIEW, "POLICY_VERSION_MISMATCH", degraded=True)
        enabled_categories = self.policy.required_categories | self.policy.optional_categories
        if not set(request.focus_categories).issubset(enabled_categories):
            return self._decision(request, Action.REVIEW, "FOCUS_CATEGORY_UNAVAILABLE", degraded=True)
        if request.stage not in self.policy.thresholds:
            return self._decision(request, Action.REVIEW, "STAGE_POLICY_UNAVAILABLE", degraded=True)
        if (AssessmentScope.SESSION in request.assessment_scopes
                and request.stage not in self.policy.session_thresholds):
            return self._decision(request, Action.REVIEW, "SESSION_POLICY_UNAVAILABLE", degraded=True)
        try:
            self.policy.limits.validate(request)
        except ContextTooLarge:
            return self._decision(request, Action.REVIEW, "CONTEXT_LIMIT_EXCEEDED", degraded=True)
        if request.attachment_kinds:
            if request.scanner_status is ScannerStatus.BLOCKED:
                return self._decision(request, Action.REFUSE, "ATTACHMENT_BLOCKED")
            if request.scanner_status is not ScannerStatus.CLEAN:
                return self._decision(request, Action.REVIEW, "ATTACHMENT_NOT_SCANNED")

        try:
            rule_results = [result for rule in self.hard_rules
                            if (result := rule.check(request)) is not None]
        except Exception:
            return self._decision(request, Action.REVIEW, "RULE_UNAVAILABLE", degraded=True)
        if rule_results:
            strongest = max(rule_results, key=lambda item: _ACTION_PRIORITY[item.action])
            return self._decision(request, strongest.action, strongest.reason_code)

        try:
            evaluation = await self.evaluator.evaluate(request)
            active_categories = self.policy.required_categories | set(request.focus_categories)
            expected = {
                (scope, category)
                for scope in request.assessment_scopes
                for category in active_categories
            }
            observed = [(signal.scope, signal.category) for signal in evaluation.signals]
            if set(observed) != expected or len(observed) != len(expected):
                raise ValueError("evaluation scope/category coverage mismatch")
            actions = []
            for signal in evaluation.signals:
                bands = (self.policy.session_thresholds if signal.scope is AssessmentScope.SESSION
                         else self.policy.thresholds)
                actions.append(bands[request.stage][signal.category].action_for(signal.probability))
            action = max(actions, key=_ACTION_PRIORITY.__getitem__)
        except Exception:
            return self._fallback(request, "EVALUATOR_OR_POLICY_UNAVAILABLE")

        return self._decision(
            request, action, "CATEGORY_THRESHOLD", signals=evaluation.signals,
            evaluator=evaluation.evaluator, model_version=evaluation.model_version,
        )

    def _fallback(self, request: SafetyInput, reason_code: str) -> SafetyDecision:
        action = (Action.CONSTRAIN if request.stage is Stage.INPUT
                  and request.risk_tier is RiskTier.LOW
                  and AssessmentScope.SESSION not in request.assessment_scopes
                  else Action.REVIEW)
        return self._decision(request, action, reason_code, degraded=True)

    def _decision(
        self,
        request: SafetyInput,
        recommended: Action,
        reason_code: str,
        *,
        signals: tuple[Signal, ...] = (),
        evaluator: str | None = None,
        model_version: str | None = None,
        degraded: bool = False,
    ) -> SafetyDecision:
        effective = Action.ALLOW if self.policy.mode is Mode.SHADOW else recommended
        return SafetyDecision(
            request_id=request.request_id,
            policy_version=self.policy.version,
            stage=request.stage,
            mode=self.policy.mode,
            recommended_action=recommended,
            effective_action=effective,
            reason_code=reason_code,
            public_reason_code=(None if self.policy.mode is Mode.SHADOW
                                else "SAFETY_REVIEW" if recommended is Action.REVIEW
                                else "SAFETY_LIMIT" if recommended in {Action.CONSTRAIN, Action.REFUSE}
                                else None),
            signals=signals,
            evaluator=evaluator,
            model_version=model_version,
            degraded=degraded,
            tool_eligible=(self.policy.mode is Mode.ENFORCE
                           and effective is Action.ALLOW
                           and request.stage is Stage.INPUT),
        )
