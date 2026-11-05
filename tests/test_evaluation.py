import unittest

from chat_safety import Action, Mode, SafetyDecision, Stage
from chat_safety.evaluation import LabeledCase, evaluate_decisions


def decision(request_id, action):
    return SafetyDecision(request_id, "test/1", Stage.INPUT, Mode.ENFORCE,
                          action, action, "TEST")


class EvaluationTests(unittest.TestCase):
    def test_reports_severe_misses_and_benign_false_refusals(self):
        cases = [
            LabeledCase("bad", Action.REFUSE, severe=True),
            LabeledCase("good", Action.ALLOW),
            LabeledCase("uncertain", Action.REVIEW),
        ]
        report = evaluate_decisions(cases, {
            "bad": decision("bad", Action.ALLOW),
            "good": decision("good", Action.REFUSE),
            "uncertain": decision("uncertain", Action.REVIEW),
        })
        self.assertEqual(report.severe_misses, 1)
        self.assertEqual(report.benign_false_refusals, 1)
        self.assertEqual(report.review_count, 1)
        self.assertAlmostEqual(report.exact_match_rate, 1 / 3)

    def test_mismatched_case_ids_fail(self):
        with self.assertRaises(ValueError):
            evaluate_decisions([LabeledCase("a", Action.ALLOW)], {})
