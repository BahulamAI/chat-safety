import copy
from dataclasses import replace
import json
from pathlib import Path
import unittest

import httpx

from chat_safety import (
    AssessmentInput, AssessmentScope, AssessmentStatus, ContextTooLarge, RecentTurn,
    RiskAssessor, RiskEvaluation, RiskScore, Stage, load_risk_registry, parse_risk_registry,
)
from chat_safety.adapters.jev_risk import JevRiskEvaluator


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/risk-registry.json"


def request(registry, *, session=False):
    return AssessmentInput(
        request_id="r1", registry_version=registry.version, stage=Stage.INPUT,
        content="Synthetic question", **({
            "assessment_scopes": (AssessmentScope.MESSAGE, AssessmentScope.SESSION),
            "session_ref": "private-session", "session_revision": 4,
            "recent_turns": (RecentTurn("assistant", "Earlier assistant content"),
                             RecentTurn("tool_result", "Earlier tool result")),
            "safety_summary": "Trusted trajectory summary",
        } if session else {}),
    )


class Evaluator:
    def __init__(self):
        self.calls = []

    async def evaluate(self, request, questions):
        self.calls.append((request, questions))
        return RiskEvaluation(tuple(RiskScore(
            question.family_id, question.scope,
            {"health": 0.9, "privacy": 0.8, "system_integrity": 0.7, "financial_harm": 0.6}[
                question.family_id], question.subfamily_id,
        ) for question in questions), "synthetic", "synthetic/1")


class RiskAssessmentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.registry = load_risk_registry(EXAMPLE)
        self.evaluator = Evaluator()
        self.assessor = RiskAssessor(self.registry, self.evaluator)

    async def test_default_and_override_rank_without_hiding_other_scores(self):
        item = request(self.registry)
        default = await self.assessor.assess_families(item)
        one = await self.assessor.assess_families(item, top_k_families=1)
        many = await self.assessor.assess_families(item, top_k_families=100)
        self.assertEqual(len(default.top_families), 3)
        self.assertEqual([s.family_id for s in one.top_families], ["health"])
        self.assertEqual(len(many.top_families), 4)
        self.assertEqual(len(one.scores), 4)
        self.assertEqual(next(s.probability for s in one.scores
                              if s.family_id == "financial_harm"), 0.6)
        self.assertEqual(len(self.evaluator.calls), 3)  # no automatic expansion
        encoded = json.loads(json.dumps(one.to_dict()))
        self.assertEqual(encoded["status"], "complete")
        self.assertEqual(encoded["schema"], "chat-safety.assessment/1")
        self.assertNotIn("effective_action", encoded)
        self.assertNotIn("Synthetic question", json.dumps(encoded))

    async def test_subfamily_call_is_explicit_and_keeps_message_session_identity(self):
        item = request(self.registry, session=True)
        broad = await self.assessor.assess_families(item, top_k_families=1)
        self.assertEqual(len(broad.top_families), 2)  # one for each scope
        self.assertEqual(len(self.evaluator.calls), 1)
        detail = await self.assessor.assess_subfamilies(item, family_ids=("health", "privacy"))
        self.assertEqual(len(self.evaluator.calls), 2)
        self.assertEqual(detail.level, "subfamilies")
        self.assertEqual(detail.top_families, ())
        self.assertTrue(all(s.subfamily_id for s in detail.scores))
        self.assertEqual({s.scope for s in detail.scores}, set(AssessmentScope))
        self.assertEqual(detail.input_digest, broad.input_digest)
        self.assertEqual(detail.registry_digest, broad.registry_digest)
        self.assertEqual(detail.session_revision, 4)
        self.assertNotEqual(replace(item, content="Changed draft").fingerprint, broad.input_digest)
        self.assertNotEqual(replace(item, session_revision=5).fingerprint, broad.input_digest)

    async def test_invalid_selection_context_or_registry_never_calls_evaluator(self):
        item = request(self.registry)
        for count in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                await self.assessor.assess_families(item, top_k_families=count)
        for selected in ((), ("missing",), ("health", "health"), ("financial_harm",)):
            with self.assertRaises(ValueError):
                await self.assessor.assess_subfamilies(item, family_ids=selected)
        with self.assertRaises(ValueError):
            await self.assessor.assess_families(replace(item, registry_version="stale"))
        with self.assertRaises(ContextTooLarge):
            await self.assessor.assess_families(replace(item, content="x" * 12_001))
        self.assertEqual(self.evaluator.calls, [])

    async def test_missing_duplicate_or_wrong_scope_scores_are_unavailable(self):
        item = request(self.registry, session=True)
        await self.assessor.assess_families(item)
        questions = self.evaluator.calls[-1][1]
        valid = (await self.evaluator.evaluate(item, questions)).scores
        for scores in (valid[:-1], (*valid, valid[0]),
                       tuple(replace(s, scope=AssessmentScope.MESSAGE) for s in valid)):
            class Broken:
                async def evaluate(self, request, questions):
                    return RiskEvaluation(scores, "synthetic", "synthetic/1")
            result = await RiskAssessor(self.registry, Broken()).assess_families(item)
            self.assertEqual(result.status, AssessmentStatus.UNAVAILABLE)
            self.assertEqual(result.scores, ())
            self.assertEqual(result.top_families, ())

    async def test_timeout_returns_error_not_zero_scores_or_chat_action(self):
        class Down:
            async def evaluate(self, request, questions):
                raise TimeoutError("private provider detail")
        result = await RiskAssessor(self.registry, Down()).assess_families(request(self.registry))
        self.assertEqual(result.status, AssessmentStatus.UNAVAILABLE)
        self.assertEqual(result.scores, ())
        self.assertNotIn("private provider detail", json.dumps(result.to_dict()))


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(EXAMPLE.read_text())

    def test_custom_ids_are_configuration_and_fingerprint_tracks_criteria(self):
        self.raw["families"]["custom_lab_risk"] = self.raw["families"].pop("health")
        self.raw["reporting"]["top_k_families"] = 1
        registry = parse_risk_registry(self.raw)
        self.assertIn("custom_lab_risk", registry.families)
        self.assertEqual(registry.top_k_families, 1)
        self.raw["families"]["custom_lab_risk"]["questions"]["session"]["true_criteria"] += " Changed."
        self.assertNotEqual(registry.fingerprint, parse_risk_registry(self.raw).fingerprint)

    def test_incomplete_questions_unknown_fields_and_bad_count_are_rejected(self):
        for mutate in (
            lambda raw: raw["families"]["health"]["questions"].pop("session"),
            lambda raw: raw["reporting"].update(top_k_families=0),
            lambda raw: raw.update(auto_nudge=True),
            lambda raw: raw["families"]["health"].update(block_at=0.8),
        ):
            raw = copy.deepcopy(self.raw)
            mutate(raw)
            with self.assertRaises(ValueError):
                parse_risk_registry(raw)

    def test_session_assessment_requires_real_context_and_revision(self):
        with self.assertRaises(ValueError):
            AssessmentInput("r", "v1", Stage.INPUT, "Content",
                            assessment_scopes=(AssessmentScope.SESSION,))


class JevRiskTests(unittest.IsolatedAsyncioTestCase):
    async def test_wire_requests_send_only_active_criteria_and_bounded_context(self):
        seen = []
        registry = load_risk_registry(EXAMPLE)

        def respond(http_request):
            body = json.loads(http_request.content)
            seen.append(body)
            return httpx.Response(200, json={
                "model": "typesafe/jev-1.14-test", "answers": {
                    name: {"type": "noul", "noul": 0.8 if index == 0 else 0.1}
                    for index, name in enumerate(body["questions"])
                },
            })

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            assessor = RiskAssessor(registry, JevRiskEvaluator(api_key="test-key", client=client))
            item = request(registry, session=True)
            broad = await assessor.assess_families(item, top_k_families=1)
            detail = await assessor.assess_subfamilies(item, family_ids=("health",))
        self.assertEqual(broad.status, AssessmentStatus.COMPLETE)
        self.assertEqual(detail.status, AssessmentStatus.COMPLETE)
        self.assertEqual([len(body["questions"]) for body in seen], [8, 2])
        self.assertEqual(set(seen[0]), {"model", "state", "questions"})
        self.assertEqual(seen[0]["state"]["recent_turns"][1]["role"], "tool_result")
        self.assertNotIn("private-session", json.dumps(seen))
        self.assertNotIn("top_k_families", json.dumps(seen))
        self.assertTrue(all("medical" in spec["instructions"]
                            for spec in seen[1]["questions"].values()))
        self.assertEqual(detail.model_version, "typesafe/jev-1.14-test")
