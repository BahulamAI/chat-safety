"""Offline example: caller arguments, policy, decision, action, and session state."""

import asyncio
from pathlib import Path

from chat_safety import (
    Action, AssessmentScope, Category, Evaluation, EventOutcome,
    SafetyGate, SessionEvent, SessionSafetyState, SessionTracker,
    Signal, Stage, content_digest, load_policy_json,
)


class SyntheticEvaluator:
    async def evaluate(self, request):
        # A real adapter asks its configured model questions. These scores are invented.
        signals = [Signal(Category.SELF_HARM, 0.01)]
        if Category.MEDICAL_ADVICE in request.focus_categories:
            signals.append(Signal(Category.MEDICAL_ADVICE, 0.05))
        if AssessmentScope.SESSION in request.assessment_scopes:
            signals.append(Signal(Category.SELF_HARM, 0.01,
                                  scope=AssessmentScope.SESSION))
            if Category.MEDICAL_ADVICE in request.focus_categories:
                signals.append(Signal(Category.MEDICAL_ADVICE, 0.65,
                                      scope=AssessmentScope.SESSION))
        return Evaluation(tuple(signals), "synthetic", "synthetic/1")


async def main():
    # The trusted caller maintains this JSON and supplies a model adapter.
    config = load_policy_json(Path(__file__).with_name("runtime-policy.json"))
    gate = SafetyGate(config.policy, SyntheticEvaluator())
    tracker = SessionTracker()
    state = SessionSafetyState("opaque-session", config.policy.version)

    # First message: the caller supplies the stage, content, actor, and context.
    first = tracker.build_input(
        state, request_id="event-1", stage=Stage.INPUT,
        surface="example-chat", actor_ref="opaque-actor",
        content="Explain how clinical studies compare treatments.",
    )
    first_decision = await gate.evaluate(first)
    assert first_decision.effective_action is Action.ALLOW
    # The caller accepts this input, then records what actually happened.
    state = tracker.append(
        state, SessionEvent(1, content_digest(first.content),
                            "User asked for general clinical research",
                            first_decision, EventOutcome.ACCEPTED),
        expected_revision=state.revision,
        next_summary="Discussion of clinical research", summary_version="example/1",
    )
    # A real caller must persist this revision using SessionStore.compare_and_swap.

    # The next assessment includes message and session scopes automatically.
    follow_up = tracker.build_input(
        state, request_id="event-2", stage=Stage.INPUT,
        surface="example-chat", actor_ref="opaque-actor",
        content="What next?", focus_categories=(Category.MEDICAL_ADVICE,),
    )
    decision = await gate.evaluate(follow_up)
    assert decision.effective_action is Action.REVIEW
    # The caller chooses and enforces its own review path.
    outcome = EventOutcome.HELD
    state = tracker.append(
        state, SessionEvent(2, content_digest(follow_up.content),
                            "Follow-up held for review handling",
                            decision, outcome),
        expected_revision=state.revision,
        next_summary="Follow-up needs review handling",
        summary_version="example/1",
    )
    print(decision.effective_action.value, outcome.value, state.revision)


if __name__ == "__main__":
    asyncio.run(main())
