"""Jev Noul-question adapter for OpenRouter's alpha Decisions API.

The caller owns question wording and policy thresholds. No API key is logged.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import urlsplit

from ..models import AssessmentScope, Category, Evaluation, SafetyInput, Signal
from ..questions import BinaryQuestion as NoulQuestion


class JevResponseError(ValueError):
    pass


def _validate_endpoint(endpoint: str) -> str:
    """Accept a trusted caller's Decisions endpoint, without URL credentials."""
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("a Decisions endpoint is required")
    endpoint = endpoint.strip()
    parsed = urlsplit(endpoint)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise ValueError("invalid Decisions endpoint")
    return endpoint


async def _post_decisions(*, endpoint, body, api_key, timeout_seconds, max_response_bytes, client):
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("install chat-safety[jev] for the Jev adapter") from exc
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if client is None:
        async with httpx.AsyncClient(timeout=timeout_seconds) as owned_client:
            return await _post_decisions(
                endpoint=endpoint, body=body, api_key=api_key, timeout_seconds=timeout_seconds,
                max_response_bytes=max_response_bytes, client=owned_client,
            )
    response = await client.post(endpoint, json=body, headers=headers, timeout=timeout_seconds)
    response.raise_for_status()
    if len(response.content) > max_response_bytes:
        raise JevResponseError("Jev response exceeds size limit")
    return response.json()


def _parse_noul_answers(payload, model, names):
    if not isinstance(payload, dict):
        raise JevResponseError("Jev response must be an object")
    snapshot, answers = payload.get("model"), payload.get("answers")
    if not isinstance(snapshot, str) or not (snapshot == model or snapshot.startswith(f"{model}-")):
        raise JevResponseError("unexpected Jev model version")
    if not isinstance(answers, dict) or set(answers) != set(names):
        raise JevResponseError("missing or unexpected Jev answers")
    scores = {}
    for name, answer in answers.items():
        if not isinstance(answer, dict) or answer.get("type") != "noul":
            raise JevResponseError("Jev answer has wrong type")
        value = answer.get("noul")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise JevResponseError("Jev answer probability is not numeric")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise JevResponseError("Jev answer probability is out of range")
        scores[name] = float(value)
    return snapshot, scores


class JevEvaluator:
    ENDPOINT = "https://openrouter.ai/api/alpha/decisions"

    def __init__(
        self,
        *,
        api_key: str,
        questions: Mapping[Category, NoulQuestion],
        session_questions: Mapping[Category, NoulQuestion] | None = None,
        mandatory_categories: frozenset[Category] | None = None,
        model: str = "typesafe/jev-1.14",
        endpoint: str = ENDPOINT,
        timeout_seconds: float = 8.0,
        max_response_bytes: int = 65_536,
        client: Any = None,
    ) -> None:
        if not api_key or not questions or not model or timeout_seconds <= 0 or max_response_bytes <= 0:
            raise ValueError("Jev key, questions, model, and positive limits are required")
        self._api_key = api_key
        self.questions = MappingProxyType(dict(questions))
        self.session_questions = MappingProxyType(dict(session_questions or {}))
        self.mandatory_categories = frozenset(
            self.questions if mandatory_categories is None else mandatory_categories
        )
        if not self.mandatory_categories or not self.mandatory_categories.issubset(self.questions):
            raise ValueError("mandatory categories must be a nonempty subset of configured questions")
        self.model = model
        self.endpoint = _validate_endpoint(endpoint)
        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max_response_bytes
        self._client = client

    async def evaluate(self, request: SafetyInput) -> Evaluation:
        question_map = self._question_map(request.assessment_scopes, request.focus_categories)
        body = {
            "model": self.model,
            "state": {
                "stage": request.stage.value,
                "surface": request.surface,
                "latest_content": request.content,
                "assessment_scopes": [scope.value for scope in request.assessment_scopes],
                "focus_categories": [category.value for category in request.focus_categories],
                "session_revision": request.session_revision,
                "recent_turns": [
                    {"role": turn.role, "text": turn.text} for turn in request.recent_turns
                ],
                "safety_summary": request.safety_summary,
                "attachment_kinds": list(request.attachment_kinds),
                "scanner_status": request.scanner_status.value,
                "authorization_facts": dict(request.authorization_facts),
            },
            "questions": {
                name: {
                    "type": "noul",
                    "instructions": question.instructions,
                    "criteria": {
                        "true": question.true_criteria,
                        "false": question.false_criteria,
                    },
                }
                for name, (_scope, _category, question) in question_map.items()
            },
        }
        payload = await _post_decisions(
            endpoint=self.endpoint, body=body, api_key=self._api_key,
            timeout_seconds=self.timeout_seconds, max_response_bytes=self.max_response_bytes,
            client=self._client,
        )
        return self.parse_response(payload, request.assessment_scopes,
                                   request.focus_categories)

    def _question_map(
        self, scopes: tuple[AssessmentScope, ...],
        focus_categories: tuple[Category, ...] = (),
    ) -> dict[str, tuple[AssessmentScope, Category, NoulQuestion]]:
        result: dict[str, tuple[AssessmentScope, Category, NoulQuestion]] = {}
        active_categories = self.mandatory_categories | set(focus_categories)
        if not active_categories.issubset(self.questions):
            raise JevResponseError("focus category has no configured question")
        for scope in scopes:
            questions = self.session_questions if scope is AssessmentScope.SESSION else self.questions
            if not active_categories.issubset(questions):
                raise JevResponseError("session question coverage mismatch")
            for category in self.questions:
                if category not in active_categories:
                    continue
                question = questions[category]
                name = (f"session_{category.value}" if scope is AssessmentScope.SESSION
                        else category.value)
                result[name] = (scope, category, question)
        return result

    def parse_response(
        self, payload: Any,
        assessment_scopes: tuple[AssessmentScope, ...] = (AssessmentScope.MESSAGE,),
        focus_categories: tuple[Category, ...] = (),
    ) -> Evaluation:
        question_map = self._question_map(assessment_scopes, focus_categories)
        model, answers = _parse_noul_answers(payload, self.model, question_map)
        signals: list[Signal] = []
        for name, (scope, category, _question) in question_map.items():
            signals.append(Signal(category=category, probability=answers[name],
                                  evidence_codes=("JEV_CATEGORY_SIGNAL",), scope=scope))
        return Evaluation(tuple(signals), evaluator="jev", model_version=model)
