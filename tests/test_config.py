import copy
import json
import tempfile
import unittest
from pathlib import Path

from chat_safety import (
    Action, AssessmentScope, Category, Evaluation, SafetyGate, SafetyInput,
    Signal, Stage, load_policy_json, parse_policy_config,
)
from chat_safety.adapters.jev import JevEvaluator
from chat_safety.eval_cli import _load_policy


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/runtime-policy.json"


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    def test_caller_json_defines_categories_questions_and_both_threshold_scopes(self):
        configured = load_policy_json(EXAMPLE)
        self.assertEqual(configured.policy.required_categories,
                         frozenset({Category.SELF_HARM}))
        self.assertEqual(configured.policy.optional_categories,
                         frozenset({Category.MEDICAL_ADVICE}))
        self.assertEqual(set(configured.policy.thresholds), set(Stage))
        self.assertEqual(set(configured.policy.session_thresholds), set(Stage))
        self.assertEqual(_load_policy(EXAMPLE).version, configured.policy.version)
        self.assertIn("trajectory", configured.session_questions[
            Category.MEDICAL_ADVICE].instructions)
        evaluator = JevEvaluator(
            api_key="synthetic-key", questions=configured.message_questions,
            session_questions=configured.session_questions,
            mandatory_categories=configured.policy.required_categories,
        )
        self.assertEqual(set(evaluator._question_map(
            (AssessmentScope.MESSAGE, AssessmentScope.SESSION),
            (Category.MEDICAL_ADVICE,))),
            {"self_harm", "medical_advice", "session_self_harm", "session_medical_advice"})

    def test_unknown_category_or_missing_coverage_fails_before_model_call(self):
        raw = copy.deepcopy(self.raw)
        raw["optional_categories"] = ["invented_category"]
        with self.assertRaises(ValueError):
            parse_policy_config(raw)

        raw = copy.deepcopy(self.raw)
        del raw["categories"]["medical_advice"]["session_question"]
        with self.assertRaisesRegex(ValueError, "session questions"):
            parse_policy_config(raw)

        raw = copy.deepcopy(self.raw)
        del raw["thresholds"]["tool_proposal"]["self_harm"]
        with self.assertRaisesRegex(ValueError, "coverage"):
            parse_policy_config(raw)

        raw = copy.deepcopy(self.raw)
        raw["thresholds"]["input"]["self_harm"]["review_at"] = True
        with self.assertRaisesRegex(ValueError, "numeric"):
            parse_policy_config(raw)

        raw = copy.deepcopy(self.raw)
        raw["thresholds"]["input"]["self_harm"]["middle_action"] = "refuse"
        raw["thresholds"]["input"]["self_harm"]["upper_action"] = "constrain"
        with self.assertRaisesRegex(ValueError, "less restrictive"):
            parse_policy_config(raw)

    def test_duplicate_json_keys_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text('{"schema":"chat-safety.policy/1","schema":"x"}')
            with self.assertRaisesRegex(ValueError, "duplicate JSON policy key"):
                load_policy_json(path)


class ConfiguredGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_caller_focus_adds_optional_category_but_preserves_mandatory_one(self):
        configured = load_policy_json(EXAMPLE)

        class SyntheticEvaluator:
            async def evaluate(self, request):
                return Evaluation((
                    Signal(Category.SELF_HARM, 0.01),
                    Signal(Category.MEDICAL_ADVICE, 0.95),
                ), evaluator="synthetic", model_version="synthetic/1")

        request = SafetyInput(
            request_id="r1", policy_version=configured.policy.version,
            stage=Stage.INPUT, surface="chat", actor_ref="opaque-actor",
            content="Synthetic request", focus_categories=(Category.MEDICAL_ADVICE,),
        )
        decision = await SafetyGate(configured.policy, SyntheticEvaluator()).evaluate(request)
        self.assertEqual(decision.effective_action, Action.REFUSE)
        self.assertEqual({item.category for item in decision.signals},
                         {Category.SELF_HARM, Category.MEDICAL_ADVICE})
