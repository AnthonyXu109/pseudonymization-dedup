"""Hand-checked behavioral fixtures for the follow-up analysis."""

import unittest

from analysis import (
    cluster_count_interval,
    cluster_ratio_interval,
    evaluate_decision,
    evaluate_membership,
    released_linkage,
    union_groups,
)
from run_order import create_report
from sensitivity import perturb_docs
from run_sensitivity import make_seed_rows
from build_audit import collect_items


class MembershipTests(unittest.TestCase):
    def test_transitive_detection_is_not_direct_pair_detection(self):
        # A original and its copy do not form an edge, but both meet B.
        got = evaluate_membership(
            case_ids=["A", "B", "A"],
            is_copy=[False, False, True],
            components=[0, 0, 0],
            pair_hits=[False],
        )
        self.assertEqual(got["pipeline_recall"], 1.0)
        self.assertEqual(got["pair_recall"], 0.0)
        self.assertEqual(got["lost_cases"], ["B"])
        self.assertEqual(got["kept_indices"], [0])

    def test_surviving_copy_preserves_its_case(self):
        got = evaluate_membership(
            case_ids=["A", "A"],
            is_copy=[False, True],
            components=[1, 1],
            pair_hits=[True],
        )
        self.assertEqual(got["lost_cases"], [])
        self.assertEqual(got["kept_indices"], [1])

    def test_union_groups_keep_whole_family_together(self):
        # The two original graphs split different pairs; their union joins all.
        self.assertEqual(union_groups([0, 0, 2], [0, 1, 1]), [0, 0, 0])


class ReleaseTests(unittest.TestCase):
    def test_linkage_uses_only_records_released_after_deduplication(self):
        docs = [
            {"case": "A", "key": "doc-a", "is_copy": False, "text": "Alice", "spans": [
                {"start": 0, "end": 5, "surface": "Alice", "type": "PERSON", "ent": "alice"}]},
            {"case": "B", "key": "doc-b", "is_copy": False, "text": "Alice", "spans": [
                {"start": 0, "end": 5, "surface": "Alice", "type": "PERSON", "ent": "alice"}]},
        ]
        before = released_linkage(docs, [0, 1], "SUR-CORPUS")
        after = released_linkage(docs, [0], "SUR-CORPUS")
        self.assertEqual((before["true"], before["predicted"], before["correct"]), (1, 1, 1))
        self.assertEqual((after["true"], after["predicted"], after["correct"]), (0, 0, 0))


class DecisionTests(unittest.TestCase):
    def test_document_scope_breaks_an_exact_copy_with_marked_name(self):
        def doc(case, key, copy, name):
            text = f"{name} alpha beta gamma delta"
            return {"case": case, "key": key, "is_copy": copy, "text": text, "spans": [
                {"start": 0, "end": len(name), "surface": name, "type": "PERSON", "ent": name}]}

        docs = [doc("A", "a", False, "Alice"), doc("B", "b", False, "Bob"),
                doc("A", "a-copy", True, "Alice")]
        raw = evaluate_decision(docs, "RAW", 0.7)
        scoped = evaluate_decision(docs, "SUR-DOC", 0.7)
        self.assertEqual(raw["pipeline_recall"], 1.0)
        self.assertEqual(scoped["pipeline_recall"], 0.0)
        self.assertEqual(raw["lost_cases"], [])
        self.assertEqual(scoped["lost_cases"], [])

    def test_report_holds_release_fixed_across_decision_strategies(self):
        def doc(case, key, copy, name):
            return {"case": case, "key": key, "is_copy": copy,
                    "text": f"{name} alpha beta gamma delta", "spans": [{
                        "start": 0, "end": len(name), "surface": name,
                        "type": "PERSON", "ent": name}]}

        docs = [doc("A", "a", False, "Alice"), doc("B", "b", False, "Bob"),
                doc("A", "a-copy", True, "Alice")]
        report = create_report(docs, reps=100)
        self.assertEqual(report["rows"]["RAW"]["pipeline_recall"], 1.0)
        self.assertEqual(report["rows"]["SUR-DOC"]["pipeline_recall"], 0.0)
        self.assertEqual(report["contrasts"]["SUR-DOC_minus_RAW"]["pipeline_recall_delta"]["estimate"], -1.0)
        self.assertEqual({row["release"] for row in report["rows"].values()}, {"SUR-DOC"})


class BootstrapTests(unittest.TestCase):
    def test_resampling_keeps_two_related_units_together(self):
        delta, lo, hi = cluster_ratio_interval([1, 1, -1], [0, 0, 1], reps=1000, seed=7)
        self.assertAlmostEqual(delta, 1 / 3)
        self.assertEqual((lo, hi), (-1.0, 1.0))
        count, clo, chi = cluster_count_interval([1, 1, -1], [0, 0, 1], reps=1000, seed=7)
        self.assertEqual((count, clo, chi), (1, -2, 4))


class SensitivityTests(unittest.TestCase):
    def test_entity_dropout_is_consistent_within_document(self):
        spans = []
        for n in range(10):
            for mention in range(2):
                spans.append({"start": n * 20 + mention * 5, "end": n * 20 + mention * 5 + 3,
                              "surface": "Ann", "type": "PERSON", "ent": ("gold", n)})
        docs = [{"key": "a", "text": "x" * 200, "spans": spans}]
        perturbed, qa = perturb_docs(docs, "entity", 20, 0)
        remaining = [m["ent"] for m in perturbed[0]["spans"]]
        self.assertEqual(len(remaining), 16)
        self.assertEqual(all(remaining.count(ent) == 2 for ent in set(remaining)), True)
        self.assertEqual(qa["inconsistent_within_document_pair_fraction"], 0.0)
        self.assertEqual(qa["dropped_entity_units"], 2)

    def test_mention_dropout_can_split_repeated_entity(self):
        docs = [{"key": "a", "text": "x" * 10, "spans": [
            {"start": 0, "end": 3, "surface": "Ann", "type": "PERSON", "ent": ("gold", 1)}
            for _ in range(10)]}]
        perturbed, qa = perturb_docs(docs, "mention", 20, 0)
        self.assertEqual(len(perturbed[0]["spans"]), 8)
        self.assertAlmostEqual(qa["inconsistent_within_document_pair_fraction"], 16 / 45)
        self.assertEqual(qa["dropped_mention_units"], 2)

    def test_report_has_both_scopes_on_the_same_perturbed_docs(self):
        def doc(case, key, copy, name):
            return {"case": case, "key": key, "is_copy": copy,
                    "text": f"{name} alpha beta gamma delta", "spans": [{
                        "start": 0, "end": len(name), "surface": name,
                        "type": "PERSON", "ent": ("gold", name)}]}
        docs = [doc("A", "a", False, "Alice"), doc("B", "b", False, "Bob"),
                doc("A", "a-copy", True, "Alice")]
        rows = make_seed_rows(docs, "entity", 20, 0)
        self.assertEqual({r["decision"] for r in rows}, {"SUR-DOC", "SUR-CORPUS"})
        self.assertEqual({r["qa"]["units_dropped"] for r in rows}, {1})
        self.assertEqual({r["seed"] for r in rows}, {0})


class AuditPackTests(unittest.TestCase):
    def test_deduplicates_pairs_without_exposing_row_or_prior_labels(self):
        tables = {"TAB": {"MASK_added": {"items": [
            {"record": "r", "survivor": "s", "text": "first text", "survivor_text": "second text",
             "survivor_a_only": "first", "survivor_b_only": "second"}]},
            "SURDOC_restored": {"items": [
                {"record": "r", "survivor": "s", "text": "first text", "survivor_text": "second text",
                 "survivor_a_only": "first", "survivor_b_only": "second"}]}}}
        items, key = collect_items(tables)
        self.assertEqual(len(items), 1)
        self.assertEqual(len(key[items[0]["id"]]["memberships"]), 2)
        self.assertNotIn("MASK", str(items[0]))
        self.assertNotIn("label", str(items[0]))


if __name__ == "__main__":
    unittest.main()
