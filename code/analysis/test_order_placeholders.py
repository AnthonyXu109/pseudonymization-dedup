"""Regression checks for the fixed-release placeholder addendum."""

import unittest

import run_order


class PlaceholderDecisionTests(unittest.TestCase):
    def test_addendum_keeps_release_and_original_protocol_intact(self):
        try:
            from run_order_placeholders import create_placeholder_report
        except ImportError as exc:
            self.fail(f"addendum implementation is missing: {exc}")

        def doc(case, key, copy, name):
            return {"case": case, "key": key, "is_copy": copy,
                    "text": f"{name} alpha beta gamma delta", "spans": [{
                        "start": 0, "end": len(name), "surface": name,
                        "type": "PERSON", "ent": name}]}

        docs = [doc("A", "a", False, "Alice"), doc("B", "b", False, "Bob"),
                doc("A", "a-copy", True, "Alice")]
        before = run_order.STRATEGIES
        report = create_placeholder_report(docs, reps=100)
        self.assertEqual(run_order.STRATEGIES, before)
        self.assertEqual(set(report["rows"]),
                         {"RAW", "SUR-DOC", "SUR-CORPUS", "MASK", "TYPE"})
        self.assertEqual({row["release"] for row in report["rows"].values()}, {"SUR-DOC"})
        self.assertEqual(report["rows"]["RAW"]["pipeline_recall"], 1.0)
        self.assertEqual(report["rows"]["SUR-DOC"]["pipeline_recall"], 0.0)
        self.assertIn("MASK_minus_RAW", report["contrasts"])
        self.assertIn("TYPE_minus_RAW", report["contrasts"])


if __name__ == "__main__":
    unittest.main()
