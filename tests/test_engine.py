import unittest

from chat_safety import (
    Action, AssessmentScope, Category, ContextLimits, Evaluation, Mode, PolicyBundle, RecentTurn,
    RiskTier, SafetyGate, SafetyInput, ScannerStatus, Signal, Stage, ThresholdBand,
)
from chat_safety.engine import RuleResult


CATEGORY = Category.FRAUD_THEFT


class FakeEvaluator:
    def __init__(self, probability=0.0, error=None, signals=None):
        self.probability = probability
        self.error = error
        self.signals = signals
        self.calls = 0

    async def evaluate(self, request):
        self.calls += 1
        if self.error:
            raise self.error
        signals = self.signals if self.signals is not None else (
            Signal(CATEGORY, self.probability),
        )
        return Evaluation(signals, evaluator="fake", model_version="fake/1")


class RefuseRule:
    def check(self, request):
        return RuleResult(Action.REFUSE, "LOCAL_HARD_STOP")


def policy(mode=Mode.ENFORCE, *, limits=None, session=False):
    return PolicyBundle(
        version="test/1",
        mode=mode,
        thresholds={
            Stage.INPUT: {CATEGORY: ThresholdBand(0.35, 0.85)},
            Stage.OUTPUT_RELEASE: {CATEGORY: ThresholdBand(0.35, 0.85)},
            **({Stage.TOOL_PROPOSAL: {CATEGORY: ThresholdBand(0.35, 0.85)}} if session else {}),
        },
        required_categories=frozenset({CATEGORY}),
        limits=limits or ContextLimits(),
        session_thresholds={
            Stage.INPUT: {CATEGORY: ThresholdBand(0.25, 0.8)},
            Stage.TOOL_PROPOSAL: {CATEGORY: ThresholdBand(0.25, 0.8)},
            Stage.OUTPUT_RELEASE: {CATEGORY: ThresholdBand(0.25, 0.8)},
        } if session else {},
    )


def request(stage=Stage.INPUT, risk=RiskTier.LOW, **kwargs):
    return SafetyInput(
        request_id="r1", policy_version="test/1", stage=stage,
        surface="chat", actor_ref="opaque-actor", content="synthetic test input",
        risk_tier=risk, **kwargs,
    )


class SafetyGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_thresholds_map_to_distinct_actions(self):
        for probability, expected in [
            (0.1, Action.ALLOW), (0.4, Action.REVIEW), (0.9, Action.REFUSE),
        ]:
            with self.subTest(probability=probability):
                result = await SafetyGate(policy(), FakeEvaluator(probability)).evaluate(request())
                self.assertEqual(result.recommended_action, expected)
                self.assertEqual(result.effective_action, expected)

    async def test_hard_rule_precedes_model(self):
        evaluator = FakeEvaluator(0)
        result = await SafetyGate(policy(), evaluator, [RefuseRule()]).evaluate(request())
        self.assertEqual(result.effective_action, Action.REFUSE)
        self.assertEqual(evaluator.calls, 0)

    async def test_provider_failure_uses_stage_and_risk_fallback(self):
        gate = SafetyGate(policy(), FakeEvaluator(error=TimeoutError()))
        low_input = await gate.evaluate(request())
        high_input = await gate.evaluate(request(risk=RiskTier.HIGH))
        output = await gate.evaluate(request(stage=Stage.OUTPUT_RELEASE))
        self.assertEqual(low_input.effective_action, Action.CONSTRAIN)
        self.assertEqual(high_input.effective_action, Action.REVIEW)
        self.assertEqual(output.effective_action, Action.REVIEW)
        self.assertTrue(all(item.degraded for item in (low_input, high_input, output)))
        self.assertFalse(low_input.tool_eligible)

    async def test_missing_category_signal_falls_back(self):
        gate = SafetyGate(policy(), FakeEvaluator(signals=(
            Signal(Category.VIOLENCE_WEAPONS, 0.0),
        )))
        result = await gate.evaluate(request())
        self.assertEqual(result.effective_action, Action.CONSTRAIN)
        self.assertTrue(result.degraded)

    async def test_harmless_latest_message_is_held_for_session_trajectory(self):
        evaluator = FakeEvaluator(signals=(
            Signal(CATEGORY, 0.02, scope=AssessmentScope.MESSAGE),
            Signal(CATEGORY, 0.92, scope=AssessmentScope.SESSION),
        ))
        item = request(
            assessment_scopes=(AssessmentScope.MESSAGE, AssessmentScope.SESSION),
            focus_categories=(CATEGORY,),
            session_ref="opaque-session", session_revision=4,
            recent_turns=(
                RecentTurn("user", "Earlier concerning request"),
                RecentTurn("assistant", "Earlier agent answer"),
                RecentTurn("tool_proposal", "Proposed external action"),
                RecentTurn("tool_result", "Tool result summary"),
            ),
        )
        result = await SafetyGate(policy(session=True), evaluator).evaluate(item)
        self.assertEqual(result.effective_action, Action.REFUSE)
        self.assertFalse(result.tool_eligible)
        self.assertEqual({signal.scope for signal in result.signals},
                         {AssessmentScope.MESSAGE, AssessmentScope.SESSION})

    async def test_session_scope_requires_its_own_signal_and_policy(self):
        item = request(
            assessment_scopes=(AssessmentScope.MESSAGE, AssessmentScope.SESSION),
            session_ref="opaque-session", session_revision=1,
            safety_summary="Server summary of prior turns",
        )
        no_policy = FakeEvaluator(0.01)
        result = await SafetyGate(policy(), no_policy).evaluate(item)
        self.assertEqual(result.reason_code, "SESSION_POLICY_UNAVAILABLE")
        self.assertEqual(no_policy.calls, 0)

        missing_signal = await SafetyGate(policy(session=True), FakeEvaluator(0.01)).evaluate(item)
        self.assertEqual(missing_signal.effective_action, Action.REVIEW)
        self.assertTrue(missing_signal.degraded)

    async def test_caller_focus_cannot_replace_mandatory_categories(self):
        item = request(focus_categories=(CATEGORY,))
        missing = FakeEvaluator(signals=(Signal(Category.MEDICAL_ADVICE, 0.01),))
        result = await SafetyGate(policy(), missing).evaluate(item)
        self.assertEqual(result.effective_action, Action.CONSTRAIN)
        self.assertTrue(result.degraded)

        unsupported = FakeEvaluator(0.01)
        result = await SafetyGate(policy(), unsupported).evaluate(
            request(focus_categories=(Category.MEDICAL_ADVICE,))
        )
        self.assertEqual(result.reason_code, "FOCUS_CATEGORY_UNAVAILABLE")
        self.assertEqual(unsupported.calls, 0)

    async def test_optional_focus_adds_category_without_making_it_global(self):
        bundle = PolicyBundle(
            version="test/1", mode=Mode.ENFORCE,
            thresholds={Stage.INPUT: {
                CATEGORY: ThresholdBand(0.35, 0.85),
                Category.MEDICAL_ADVICE: ThresholdBand(0.25, 0.75),
            }},
            required_categories=frozenset({CATEGORY}),
            optional_categories=frozenset({Category.MEDICAL_ADVICE}),
        )
        focused = FakeEvaluator(signals=(
            Signal(CATEGORY, 0.01),
            Signal(Category.MEDICAL_ADVICE, 0.8),
        ))
        result = await SafetyGate(bundle, focused).evaluate(
            request(focus_categories=(Category.MEDICAL_ADVICE,))
        )
        self.assertEqual(result.effective_action, Action.REFUSE)

        ordinary = await SafetyGate(bundle, FakeEvaluator(0.01)).evaluate(request())
        self.assertEqual(ordinary.effective_action, Action.ALLOW)

    async def test_tool_proposal_session_risk_is_checked_before_dispatch(self):
        evaluator = FakeEvaluator(signals=(
            Signal(CATEGORY, 0.01, scope=AssessmentScope.MESSAGE),
            Signal(CATEGORY, 0.9, scope=AssessmentScope.SESSION),
        ))
        item = request(
            stage=Stage.TOOL_PROPOSAL,
            assessment_scopes=(AssessmentScope.MESSAGE, AssessmentScope.SESSION),
            session_ref="opaque-session", session_revision=3,
            safety_summary="Cumulative sequence of earlier requests and agent outputs",
        )
        result = await SafetyGate(policy(session=True), evaluator).evaluate(item)
        self.assertEqual(result.effective_action, Action.REFUSE)

    async def test_agent_output_can_raise_session_risk(self):
        evaluator = FakeEvaluator(signals=(
            Signal(CATEGORY, 0.04, scope=AssessmentScope.MESSAGE),
            Signal(CATEGORY, 0.6, scope=AssessmentScope.SESSION),
        ))
        item = request(
            stage=Stage.OUTPUT_RELEASE,
            assessment_scopes=(AssessmentScope.MESSAGE, AssessmentScope.SESSION),
            session_ref="opaque-session", session_revision=6,
            recent_turns=(
                RecentTurn("user", "Benign follow-up"),
                RecentTurn("assistant", "Earlier answer that changed the trajectory"),
            ),
        )
        result = await SafetyGate(policy(session=True), evaluator).evaluate(item)
        self.assertEqual(result.effective_action, Action.REVIEW)

    async def test_oversize_content_is_not_truncated_or_sent(self):
        evaluator = FakeEvaluator(0)
        gate = SafetyGate(policy(limits=ContextLimits(max_content_chars=5)), evaluator)
        result = await gate.evaluate(request())
        self.assertEqual(result.effective_action, Action.REVIEW)
        self.assertEqual(result.reason_code, "CONTEXT_LIMIT_EXCEEDED")
        self.assertEqual(evaluator.calls, 0)

    async def test_oversize_authorization_facts_are_not_sent(self):
        evaluator = FakeEvaluator(0)
        gate = SafetyGate(policy(limits=ContextLimits(max_authorization_facts=1)), evaluator)
        result = await gate.evaluate(request(authorization_facts={"a": True, "b": False}))
        self.assertEqual(result.effective_action, Action.REVIEW)
        self.assertEqual(evaluator.calls, 0)

    async def test_unscanned_attachment_is_held_before_model_call(self):
        evaluator = FakeEvaluator(0)
        gate = SafetyGate(policy(), evaluator)
        result = await gate.evaluate(request(attachment_kinds=("image",),
                                             scanner_status=ScannerStatus.PENDING))
        self.assertEqual(result.effective_action, Action.REVIEW)
        self.assertEqual(result.reason_code, "ATTACHMENT_NOT_SCANNED")
        self.assertEqual(evaluator.calls, 0)

    async def test_shadow_reports_recommendation_without_enforcement(self):
        result = await SafetyGate(policy(Mode.SHADOW), FakeEvaluator(0.9)).evaluate(request())
        self.assertEqual(result.recommended_action, Action.REFUSE)
        self.assertEqual(result.effective_action, Action.ALLOW)
        self.assertFalse(result.tool_eligible)

    async def test_policy_version_mismatch_holds_in_enforce_mode(self):
        evaluator = FakeEvaluator()
        mismatched = SafetyInput("r1", "wrong/1", Stage.INPUT, "chat", "actor", "text")
        result = await SafetyGate(policy(), evaluator).evaluate(mismatched)
        self.assertEqual(result.effective_action, Action.REVIEW)
        self.assertEqual(evaluator.calls, 0)

    def test_threshold_and_policy_validation(self):
        with self.assertRaises(ValueError):
            ThresholdBand(0.9, 0.3)
        with self.assertRaises(ValueError):
            PolicyBundle("test/1", Mode.ENFORCE, {Stage.INPUT: {}},
                         frozenset({CATEGORY}))
        with self.assertRaises(ValueError):
            RecentTurn("system", "untrusted")
        with self.assertRaises(ValueError):
            request(assessment_scopes=(AssessmentScope.SESSION,))
        with self.assertRaises(ValueError):
            Signal(CATEGORY, 0.2, evidence_codes=("raw user text",))

    def test_policy_and_input_snapshot_cannot_be_mutated(self):
        thresholds = {Stage.INPUT: {CATEGORY: ThresholdBand(0.35, 0.85)}}
        bundle = PolicyBundle("test/1", Mode.ENFORCE, thresholds,
                              frozenset({CATEGORY}))
        thresholds[Stage.INPUT][CATEGORY] = ThresholdBand(0.1, 0.2)
        self.assertEqual(bundle.thresholds[Stage.INPUT][CATEGORY].refuse_at, 0.85)
        facts = {"approved": True}
        item = request(authorization_facts=facts)
        facts["approved"] = False
        self.assertTrue(item.authorization_facts["approved"])
