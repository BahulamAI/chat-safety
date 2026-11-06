import unittest
import json

import httpx

from chat_safety import AssessmentScope, Category, RecentTurn, SafetyInput, Stage
from chat_safety.adapters.jev import JevEvaluator, JevResponseError, NoulQuestion


class JevResponseTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = JevEvaluator(
            api_key="test-only-key",
            questions={Category.FRAUD_THEFT: NoulQuestion(
                "Does this request seek actionable fraud assistance?",
                "The user seeks steps to deceive or steal.",
                "The user seeks legitimate analysis, reporting, or defense.",
            )},
        )

    def test_valid_noul_answer_preserves_model_snapshot(self):
        result = self.evaluator.parse_response({
            "model": "typesafe/jev-1.14-20261014",
            "answers": {"fraud_theft": {"type": "noul", "noul": 0.92}},
        })
        self.assertEqual(result.model_version, "typesafe/jev-1.14-20261014")
        self.assertEqual(result.signals[0].probability, 0.92)

    def test_malformed_answers_are_rejected(self):
        base = {"model": "typesafe/jev-1.14-20261014", "answers": {}}
        for answer in ({"type": "choice", "choice": "yes"},
                       {"type": "noul", "noul": True},
                       {"type": "noul", "noul": float("nan")},
                       {"type": "noul", "noul": 1.1}):
            with self.subTest(answer=answer), self.assertRaises(JevResponseError):
                self.evaluator.parse_response({**base, "answers": {"fraud_theft": answer}})
        with self.assertRaises(JevResponseError):
            self.evaluator.parse_response(base)
        with self.assertRaises(JevResponseError):
            self.evaluator.parse_response({**base, "model": "typesafe/jev-1.140",
                                           "answers": {"fraud_theft": {"type": "noul", "noul": 0.1}}})


class JevRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_caller_gateway_and_model_are_used_without_direct_provider_fallback(self):
        from chat_safety import AssessmentInput, RiskAssessor, load_risk_registry
        from chat_safety.adapters.jev_risk import JevRiskEvaluator
        from pathlib import Path

        endpoint = "http://gateway.internal/v1/decisions"
        model = "typesafe/jev-configured"
        seen = []

        def respond(request):
            body = json.loads(request.content)
            seen.append((str(request.url), request.headers["Authorization"], body["model"]))
            return httpx.Response(200, json={"model": model, "answers": {
                name: {"type": "noul", "noul": 0.1} for name in body["questions"]
            }})

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            question = NoulQuestion("Evaluate text", "Concern", "No concern")
            evaluator = JevEvaluator(api_key="svc_test", endpoint=endpoint, model=model,
                                     client=client, questions={Category.FRAUD_THEFT: question})
            await evaluator.evaluate(SafetyInput("r", "v1", Stage.INPUT, "chat", "actor", "Text"))
            registry = load_risk_registry(Path(__file__).parents[1] / "examples/risk-registry.json")
            evaluator = JevRiskEvaluator(api_key="svc_test", endpoint=endpoint, model=model, client=client)
            result = await RiskAssessor(registry, evaluator).assess_families(
                AssessmentInput("r", registry.version, Stage.INPUT, "Text"))
            self.assertEqual(result.model_version, model)
        self.assertEqual(seen, [(endpoint, "Bearer svc_test", model)] * 2)

    def test_invalid_gateway_endpoints_are_rejected(self):
        from chat_safety.adapters.jev_risk import JevRiskEvaluator

        for endpoint in ("", "file:///tmp/decisions", "https://user:password@gateway/decisions",
                         "https://gateway/decisions?token=secret", "https://gateway/decisions#part"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                JevRiskEvaluator(api_key="svc_test", endpoint=endpoint)

    async def test_request_is_bounded_state_with_named_noul_question(self):
        seen = {}

        def respond(request):
            seen["url"] = str(request.url)
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={
                "model": "typesafe/jev-1.14-20261014",
                "answers": {"fraud_theft": {"type": "noul", "noul": 0.03}},
            })

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            evaluator = JevEvaluator(
                api_key="test-only-key", client=client,
                questions={Category.FRAUD_THEFT: NoulQuestion(
                    "Does latest_content seek actionable fraud assistance?",
                    "Actionable fraud assistance", "Legitimate analysis or reporting",
                )},
            )
            result = await evaluator.evaluate(SafetyInput(
                "r1", "test/1", Stage.INPUT, "chat", "private-actor-id", "test input"
            ))

        self.assertEqual(seen["url"], JevEvaluator.ENDPOINT)
        self.assertEqual(seen["body"]["model"], "typesafe/jev-1.14")
        self.assertEqual(seen["body"]["state"]["latest_content"], "test input")
        self.assertNotIn("private-actor-id", json.dumps(seen["body"]))
        self.assertEqual(seen["body"]["questions"]["fraud_theft"]["type"], "noul")
        self.assertEqual(result.signals[0].probability, 0.03)

    async def test_optional_focus_adds_question_to_mandatory_set(self):
        seen = []

        def respond(request):
            body = json.loads(request.content)
            names = set(body["questions"])
            seen.append(names)
            return httpx.Response(200, json={
                "model": "typesafe/jev-1.14-20261014",
                "answers": {name: {"type": "noul", "noul": 0.01} for name in names},
            })

        question = NoulQuestion("Assess the text", "Unsafe", "Safe")
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            evaluator = JevEvaluator(
                api_key="test-only-key", client=client,
                questions={Category.FRAUD_THEFT: question,
                           Category.MEDICAL_ADVICE: question},
                mandatory_categories=frozenset({Category.FRAUD_THEFT}),
            )
            ordinary = await evaluator.evaluate(SafetyInput(
                "ordinary", "test/1", Stage.INPUT, "chat", "actor", "Ordinary text"
            ))
            focused = await evaluator.evaluate(SafetyInput(
                "focused", "test/1", Stage.INPUT, "chat", "actor", "Medical text",
                focus_categories=(Category.MEDICAL_ADVICE,),
            ))

        self.assertEqual(seen[0], {"fraud_theft"})
        self.assertEqual(seen[1], {"fraud_theft", "medical_advice"})
        self.assertEqual(len(ordinary.signals), 1)
        self.assertEqual(len(focused.signals), 2)

    async def test_one_request_returns_message_and_session_signals_with_tool_history(self):
        seen = {}

        def respond(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={
                "model": "typesafe/jev-1.14-20261014",
                "answers": {
                    "medical_advice": {"type": "noul", "noul": 0.03},
                    "session_medical_advice": {"type": "noul", "noul": 0.91},
                },
            })

        question = NoulQuestion(
            "Does latest_content ask for unsafe personalized medical advice?",
            "Unsafe individualized advice", "General education or safe support",
        )
        session_question = NoulQuestion(
            "Does the recent-turn trajectory seek unsafe personalized medical advice?",
            "Escalating toward unsafe advice", "Legitimate research or general education",
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            evaluator = JevEvaluator(
                api_key="test-only-key", client=client,
                questions={Category.MEDICAL_ADVICE: question},
                session_questions={Category.MEDICAL_ADVICE: session_question},
            )
            result = await evaluator.evaluate(SafetyInput(
                "r2", "test/1", Stage.INPUT, "chat", "private-actor-id", "What next?",
                assessment_scopes=(AssessmentScope.MESSAGE, AssessmentScope.SESSION),
                focus_categories=(Category.MEDICAL_ADVICE,),
                session_ref="private-session-id", session_revision=5,
                safety_summary="Server-produced summary of earlier interaction",
                recent_turns=(RecentTurn("assistant", "Prior answer summary"),
                              RecentTurn("tool_result", "Retrieved source summary")),
            ))

        self.assertEqual(len(result.signals), 2)
        self.assertEqual(result.signals[1].scope, AssessmentScope.SESSION)
        self.assertEqual(result.signals[1].probability, 0.91)
        self.assertEqual(seen["body"]["state"]["focus_categories"], ["medical_advice"])
        self.assertEqual(seen["body"]["state"]["recent_turns"][1]["role"], "tool_result")
        self.assertNotIn("private-session-id", json.dumps(seen["body"]))
        self.assertNotIn("private-actor-id", json.dumps(seen["body"]))
