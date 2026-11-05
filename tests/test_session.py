import unittest
import json

from chat_safety import (
    Action, AssessmentScope, Category, Evaluation, EventOutcome, Mode,
    PolicyBundle, RiskTier, SafetyDecision, SafetyGate, ScannerStatus, Signal,
    SessionConflict, SessionEvent, SessionSafetyState, SessionTracker, Stage,
    ThresholdBand, content_digest,
    state_from_record, state_to_record,
)


def decision(sequence, stage, action=Action.ALLOW):
    return SafetyDecision(
        request_id=f"event-{sequence}", policy_version="test-policy/1",
        stage=stage, mode=Mode.ENFORCE,
        recommended_action=action, effective_action=action,
        reason_code="TEST_DECISION",
    )


def event(sequence, stage, outcome, *, action=Action.ALLOW):
    return SessionEvent(
        sequence=sequence,
        content_hash=content_digest(f"private content {sequence}"),
        summary=f"Redacted event {sequence}",
        decision=decision(sequence, stage, action),
        outcome=outcome,
    )


class SessionTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = SessionTracker()
        self.state = SessionSafetyState("opaque-session", "test-policy/1")

    def append(self, state, item, summary="Server safety summary"):
        return self.tracker.append(
            state, item, expected_revision=state.revision,
            next_summary=summary, summary_version="summary-builder/1",
        )

    def test_user_agent_and_tools_feed_next_session_assessment(self):
        first = self.tracker.build_input(
            self.state, request_id="input-1", stage=Stage.INPUT,
            surface="chat", actor_ref="opaque-actor", content="First message",
        )
        self.assertEqual(first.assessment_scopes, (AssessmentScope.MESSAGE,))

        state = self.append(self.state, event(1, Stage.INPUT, EventOutcome.ACCEPTED))
        state = self.append(state, event(2, Stage.TOOL_PROPOSAL, EventOutcome.EXECUTED))
        state = self.append(state, event(3, Stage.TOOL_RESULT, EventOutcome.ACCEPTED))
        state = self.append(state, event(4, Stage.OUTPUT_RELEASE, EventOutcome.RELEASED))
        next_input = self.tracker.build_input(
            state, request_id="input-5", stage=Stage.INPUT,
            surface="chat", actor_ref="opaque-actor", content="What next?",
            focus_categories=(Category.MEDICAL_ADVICE,),
        )

        self.assertEqual(next_input.assessment_scopes,
                         (AssessmentScope.MESSAGE, AssessmentScope.SESSION))
        self.assertEqual([turn.role for turn in next_input.recent_turns],
                         ["user", "tool_proposal", "tool_result", "assistant"])
        self.assertEqual(next_input.session_revision, 4)
        self.assertEqual(next_input.safety_summary, "Server safety summary")
        self.assertEqual(next_input.focus_categories, (Category.MEDICAL_ADVICE,))
        self.assertNotIn("private content", str(state))

    def test_stale_or_duplicate_event_is_rejected(self):
        state = self.append(self.state, event(1, Stage.INPUT, EventOutcome.ACCEPTED))
        with self.assertRaises(SessionConflict):
            self.tracker.append(
                state, event(2, Stage.INPUT, EventOutcome.ACCEPTED),
                expected_revision=0, next_summary="Summary", summary_version="v1",
            )
        with self.assertRaises(SessionConflict):
            self.append(state, event(3, Stage.INPUT, EventOutcome.ACCEPTED))
        duplicate = SessionEvent(
            sequence=2, content_hash=content_digest("other"), summary="Other",
            decision=decision(1, Stage.INPUT), outcome=EventOutcome.ACCEPTED,
        )
        with self.assertRaises(SessionConflict):
            self.append(state, duplicate)

    def test_window_is_bounded_and_summary_preserves_earlier_context(self):
        tracker = SessionTracker(max_recent_events=2)
        state = self.state
        for number in range(1, 4):
            state = tracker.append(
                state, event(number, Stage.INPUT, EventOutcome.ACCEPTED),
                expected_revision=state.revision,
                next_summary="Earlier trajectory retained by the backend",
                summary_version="v1",
            )
        self.assertEqual([item.sequence for item in state.recent_events], [2, 3])
        self.assertEqual(state.last_sequence, 3)
        self.assertIn("Earlier trajectory", state.summary)

    def test_refused_tool_cannot_be_recorded_as_executed(self):
        with self.assertRaises(ValueError):
            event(1, Stage.TOOL_PROPOSAL, EventOutcome.EXECUTED,
                  action=Action.REFUSE)
        with self.assertRaises(ValueError):
            event(1, Stage.TOOL_PROPOSAL, EventOutcome.EXECUTED,
                  action=Action.CONSTRAIN)
        with self.assertRaises(ValueError):
            event(1, Stage.INPUT, EventOutcome.EXECUTED)
        with self.assertRaises(ValueError):
            event(1, Stage.INPUT, EventOutcome.ACCEPTED,
                  action=Action.REVIEW)

    def test_share_export_is_not_a_model_stage(self):
        with self.assertRaises(ValueError):
            Stage("share_export")

    def test_persisted_state_round_trip_uses_only_redacted_event_data(self):
        state = self.append(self.state, event(1, Stage.INPUT, EventOutcome.ACCEPTED))
        encoded = state_to_record(state)
        stored = json.loads(json.dumps(encoded))
        self.assertEqual(state_from_record(stored), state)
        self.assertNotIn("private content 1", json.dumps(stored))
        stored["recent_events"][0]["decision"]["stage"] = "share_export"
        with self.assertRaises(ValueError):
            state_from_record(stored)


class CallerSessionFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_caller_arguments_config_decision_and_recorded_outcome(self):
        class SessionEvaluator:
            async def evaluate(self, request):
                self.asserted_request = request
                return Evaluation((
                    Signal(Category.SELF_HARM, 0.02, scope=AssessmentScope.MESSAGE),
                    Signal(Category.SELF_HARM, 0.7, scope=AssessmentScope.SESSION),
                ), evaluator="synthetic", model_version="test/1")

        bands = {Category.SELF_HARM: ThresholdBand(
            0.4, 0.9, middle_action=Action.CONSTRAIN,
        )}
        config = PolicyBundle(
            version="test-policy/1", mode=Mode.ENFORCE,
            thresholds={Stage.INPUT: bands},
            session_thresholds={Stage.INPUT: bands},
            required_categories=frozenset({Category.SELF_HARM}),
        )
        tracker = SessionTracker()
        initial = SessionSafetyState("opaque-session", config.version)
        prior = tracker.append(
            initial, event(1, Stage.INPUT, EventOutcome.ACCEPTED),
            expected_revision=0, next_summary="Prior trajectory",
            summary_version="summary-builder/1",
        )
        request = tracker.build_input(
            prior, request_id="assess-2", stage=Stage.INPUT, surface="chat",
            actor_ref="opaque-actor", tenant_ref="opaque-tenant",
            content="What next?", risk_tier=RiskTier.HIGH,
            attachment_kinds=("image",), scanner_status=ScannerStatus.CLEAN,
            authorization_facts={"member_verified": True},
        )
        evaluator = SessionEvaluator()
        decision_result = await SafetyGate(config, evaluator).evaluate(request)
        self.assertEqual(evaluator.asserted_request, request)
        self.assertEqual(request.assessment_scopes,
                         (AssessmentScope.MESSAGE, AssessmentScope.SESSION))
        self.assertEqual(request.tenant_ref, "opaque-tenant")
        self.assertEqual(request.authorization_facts["member_verified"], True)
        self.assertEqual(decision_result.effective_action, Action.CONSTRAIN)

        # The caller applies the decision, then records what actually happened.
        updated = tracker.append(
            prior, SessionEvent(
                sequence=2, content_hash=content_digest(request.content),
                summary="User follow-up held for a supportive response",
                decision=decision_result, outcome=EventOutcome.HELD,
            ), expected_revision=prior.revision,
            next_summary="Supportive response needed after prior trajectory",
            summary_version="summary-builder/1",
        )
        self.assertEqual(updated.revision, 2)
        self.assertEqual(updated.recent_events[-1].outcome, EventOutcome.HELD)
