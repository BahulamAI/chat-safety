"""Jev transport for dynamic family criteria; no local policy decisions."""

import math

from ..assessment import AssessmentInput, RiskEvaluation, RiskQuestion, RiskScore
from .jev import JevEvaluator, _parse_noul_answers, _post_decisions


class JevRiskEvaluator:
    def __init__(self, *, api_key: str, model: str = "typesafe/jev-1.14",
                 timeout_seconds: float = 8.0, max_response_bytes: int = 262_144,
                 client=None):
        if (not api_key or not model or not math.isfinite(timeout_seconds)
                or timeout_seconds <= 0 or max_response_bytes <= 0):
            raise ValueError("Jev credentials, model, and positive limits are required")
        self._api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max_response_bytes
        self._client = client

    async def evaluate(self, request: AssessmentInput,
                       questions: tuple[RiskQuestion, ...]) -> RiskEvaluation:
        # Opaque question IDs avoid collisions between custom families, children, and scopes.
        named = {f"q{index}": item for index, item in enumerate(questions)}
        body = {"model": self.model, "state": request.model_state(), "questions": {
            name: {"type": "noul", "instructions": item.question.instructions,
                   "criteria": {"true": item.question.true_criteria,
                                "false": item.question.false_criteria}}
            for name, item in named.items()
        }}
        payload = await _post_decisions(
            endpoint=JevEvaluator.ENDPOINT, body=body, api_key=self._api_key,
            timeout_seconds=self.timeout_seconds, max_response_bytes=self.max_response_bytes,
            client=self._client,
        )
        snapshot, answers = _parse_noul_answers(payload, self.model, named)
        return RiskEvaluation(tuple(
            RiskScore(item.family_id, item.scope, answers[name], item.subfamily_id)
            for name, item in named.items()
        ), evaluator="jev", model_version=snapshot)
