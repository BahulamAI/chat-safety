"""Offline caller example: rank, explicitly drill down, then build a trusted nudge."""

import asyncio
from pathlib import Path

from chat_safety import (
    AssessmentInput, AssessmentScope, AssessmentStatus, RecentTurn, RiskAssessor,
    RiskEvaluation, RiskScore, Stage, load_risk_registry,
)


class SyntheticEvaluator:
    calls = 0

    async def evaluate(self, request, questions):
        self.calls += 1
        return RiskEvaluation(tuple(
            RiskScore(item.family_id, item.scope,
                      (0.8 if item.subfamily_id else 0.65)
                      if item.family_id == "health" and item.scope is AssessmentScope.SESSION
                      else 0.05, item.subfamily_id)
            for item in questions
        ), "synthetic", "synthetic/1")


async def main():
    registry = load_risk_registry(Path(__file__).with_name("risk-registry.json"))
    evaluator = SyntheticEvaluator()
    assessor = RiskAssessor(registry, evaluator)
    request = AssessmentInput(
        request_id="event-8", registry_version=registry.version,
        stage=Stage.INPUT, content="What next?",
        assessment_scopes=(AssessmentScope.MESSAGE, AssessmentScope.SESSION),
        session_ref="opaque-session", session_revision=7,
        recent_turns=(RecentTurn("assistant", "Earlier research discussion summary"),),
        safety_summary="Trusted summary of the earlier conversation",
    )
    families = await assessor.assess_families(request, top_k_families=1)
    if families.status is not AssessmentStatus.COMPLETE:
        print("Assessment unavailable; caller chooses its recovery path")
        return
    # Application policy, intentionally outside the framework. These thresholds are invented.
    expand_at = {"health": 0.6, "privacy": 0.6}
    selected = tuple(sorted({score.family_id for score in families.scores
                             if score.family_id in expand_at
                             and score.probability >= expand_at[score.family_id]}))
    print("Top families:", [(s.scope.value, s.family_id) for s in families.top_families])
    if selected:
        # This is a second model call on the same immutable content/history snapshot.
        detail = await assessor.assess_subfamilies(request, family_ids=selected)
        if detail.status is not AssessmentStatus.COMPLETE:
            print("Detail unavailable; caller chooses its recovery path")
            return
        assert detail.input_digest == families.input_digest
        assert detail.registry_digest == families.registry_digest
        # Trusted application templates, never a model-generated instruction or raw user text.
        nudges = {("health", "unsafe_personalized_guidance"):
                  "Keep the answer focused on research evidence and its limitations. "
                  "Avoid individualized diagnosis or treatment instructions."}
        selected_nudges = {nudges[(s.family_id, s.subfamily_id)] for s in detail.scores
                           if s.probability >= 0.7 and (s.family_id, s.subfamily_id) in nudges}
        for nudge in sorted(selected_nudges):
            print("Caller nudge:", nudge)
    print("Evaluator calls:", evaluator.calls)
    # The chat app attaches the nudge, generates a draft, and decides whether to reassess it.
    # Persist evidence + actual caller outcome under the session revision; no store runs here.


if __name__ == "__main__":
    asyncio.run(main())
