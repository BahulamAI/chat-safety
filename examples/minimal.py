"""Synthetic, non-production example of a shadow decision."""

import asyncio

from chat_safety import (
    Action, Category, Evaluation, Mode, PolicyBundle, SafetyGate, SafetyInput,
    Signal, Stage, ThresholdBand,
)


class SyntheticEvaluator:
    async def evaluate(self, request: SafetyInput) -> Evaluation:
        return Evaluation(
            signals=(Signal(Category.FRAUD_THEFT, 0.42),),
            evaluator="synthetic",
            model_version="synthetic/1",
        )


async def main() -> None:
    # These numbers illustrate API use; they are not calibrated safety thresholds.
    bundle = PolicyBundle(
        version="example/1",
        mode=Mode.SHADOW,
        thresholds={Stage.INPUT: {
            Category.FRAUD_THEFT: ThresholdBand(0.3, 0.8),
        }},
        required_categories=frozenset({Category.FRAUD_THEFT}),
    )
    decision = await SafetyGate(bundle, SyntheticEvaluator()).evaluate(SafetyInput(
        request_id="example-1",
        policy_version="example/1",
        stage=Stage.INPUT,
        surface="example-chat",
        actor_ref="opaque-example-actor",
        content="A synthetic example request",
    ))
    assert decision.recommended_action is Action.REVIEW
    assert decision.effective_action is Action.ALLOW  # shadow observes only
    print(decision.recommended_action.value, decision.effective_action.value)


if __name__ == "__main__":
    asyncio.run(main())
