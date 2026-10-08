"""Hand-checkable regression tests for the bounded scientific follow-up."""

import unittest

import numpy as np

from followup_science import (
    decision_from_matrix,
    full_shingles,
    pair_hash_comparison,
    policy_spans,
)
from followup_thresholds import tab_threshold_report
from followup_collision import count_collisions
from followup_natural_scope import ner_spans_from_entities, validate_frozen_pairs


class FollowupScienceTests(unittest.TestCase):
    def test_threshold_graph_preserves_copy_but_can_lose_distinct_case(self):
        # A is an original, B its copy, C a different original case.
        docs = [
            {"case": 0, "is_copy": False},
            {"case": 0, "is_copy": True},
            {"case": 1, "is_copy": False},
        ]
        jac = np.array([[0, 0.8, 0.6], [0.8, 0, 0.4], [0.6, 0.4, 0]])
        strict = decision_from_matrix(docs, jac, 0.7)
        loose = decision_from_matrix(docs, jac, 0.5)
        self.assertEqual((strict["copies_detected"], strict["lost_cases"]), (1, []))
        self.assertEqual((loose["copies_detected"], loose["lost_cases"]), (1, [1]))
        self.assertEqual((strict["retained_records"], loose["retained_records"]), (2, 1))

    def test_full_shingles_match_short_document_window_convention(self):
        self.assertEqual(full_shingles("Alice met Bob", n=5), {"alice met bob"})

    def test_collision_check_detects_false_pair_match(self):
        # A forced collision demonstrates the observable failure being checked.
        comparison = pair_hash_comparison(
            "alpha bravo charlie delta echo",
            "foxtrot golf hotel india juliet",
            hasher=lambda _: 0,
        )
        self.assertEqual(comparison["full_jaccard"], 0.0)
        self.assertEqual(comparison["hashed_jaccard"], 1.0)

    def test_narrow_marking_excludes_years_and_extra_names(self):
        text = "Alice Smith met Jane Doe in 2020."
        self.assertEqual(policy_spans(text, "Alice Smith", "TITLE_ONLY"), [(0, 11)])
        self.assertEqual(policy_spans(text, "Alice Smith", "TITLE_NAME"), [(0, 11), (16, 24)])
        self.assertEqual(policy_spans(text, "Alice Smith", "DENSE"),
                         [(0, 11), (16, 24), (28, 32)])

    def test_threshold_report_keeps_all_prespecified_cells(self):
        docs = [
            {"case": 0, "is_copy": False},
            {"case": 0, "is_copy": True},
            {"case": 1, "is_copy": False},
        ]
        raw = np.array([[0, 0.8, 0.6], [0.8, 0, 0.4], [0.6, 0.4, 0]])
        doc = np.array([[0, 0.3, 0.2], [0.3, 0, 0.1], [0.2, 0.1, 0]])
        report = tab_threshold_report(docs, {"RAW": raw, "SUR-DOC": doc},
                                      thresholds=(0.5, 0.7), reps=20)
        self.assertEqual(report["thresholds"], [0.5, 0.7])
        self.assertEqual(report["rows"]["0.7"]["RAW"]["copies_detected"], 1)
        self.assertEqual(report["rows"]["0.7"]["SUR-DOC"]["copies_detected"], 0)
        self.assertEqual(report["rows"]["0.5"]["RAW"]["distinct_cases_lost"], 1)

    def test_collision_inventory_counts_unique_texts_not_occurrences(self):
        sets = [{"one two three", "four five six"}, {"one two three"}]
        result = count_collisions(sets, hasher=lambda _: 0)
        self.assertEqual(result["distinct_text_shingles"], 2)
        self.assertEqual(result["distinct_crc_ids"], 1)
        self.assertEqual(result["true_collisions"], 1)

    def test_ner_scope_filters_unrelated_labels(self):
        entities = [(0, 11, "PERSON"), (16, 24, "PRODUCT"), (28, 32, "DATE")]
        self.assertEqual(ner_spans_from_entities(entities), [(0, 11), (28, 32)])

    def test_natural_followup_rejects_changed_cohort(self):
        frozen = [{"pageid": 7, "revids": [10, 9]}]
        same = [{"pageid": 7, "revids": [10, 9]}]
        wrong = [{"pageid": 7, "revids": [10, 8]}]
        validate_frozen_pairs(same, frozen)
        with self.assertRaises(ValueError):
            validate_frozen_pairs(wrong, frozen)


if __name__ == "__main__":
    unittest.main()
