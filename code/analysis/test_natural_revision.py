import unittest

import natural_revision as nr


class NaturalRevisionTests(unittest.TestCase):
    def test_same_page_revision_pair_is_selected_before_cutoff(self):
        page = {"pageid": 42, "title": "Ada Writer", "revisions": [
            {"revid": 3, "timestamp": "2026-01-02T00:00:00Z", "slots": {"main": {"contentmodel": "wikitext", "content": "later"}}},
            {"revid": 2, "timestamp": "2025-12-30T00:00:00Z", "slots": {"main": {"contentmodel": "wikitext", "content": "second"}}},
            {"revid": 1, "timestamp": "2025-12-20T00:00:00Z", "slots": {"main": {"contentmodel": "wikitext", "content": "first"}}},
        ]}
        self.assertEqual(nr.select_pair(page)[0:2], (2, 1))

    def test_title_and_year_are_marked_without_overlapping_name(self):
        text = "Ada Writer met Grace Hopper in 2025. Ada Writer wrote books."
        spans = nr.mark_spans(text, "Ada Writer")
        self.assertEqual([text[a:b] for a, b in spans],
                         ["Ada Writer", "Grace Hopper", "2025", "Ada Writer"])

    def test_scope_changes_natural_copy_similarity(self):
        text = "Ada Writer wrote this article and Ada Writer revised this article yesterday"
        spans = nr.mark_spans(text, "Ada Writer")
        original = nr.release(text, spans, "RAW", "r1")
        corpus_a = nr.release(text, spans, "SUR-CORPUS", "r1")
        corpus_b = nr.release(text, spans, "SUR-CORPUS", "r2")
        doc_a = nr.release(text, spans, "SUR-DOC", "r1")
        doc_b = nr.release(text, spans, "SUR-DOC", "r2")
        self.assertEqual(nr.jaccard(original, original), 1.0)
        self.assertEqual(nr.jaccard(corpus_a, corpus_b), 1.0)
        self.assertLess(nr.jaccard(doc_a, doc_b), 1.0)

    def test_minimum_document_and_raw_similarity_filter(self):
        self.assertFalse(nr.eligible("short text", "short text"))
        a = " ".join("word" + str(i) for i in range(500))
        self.assertFalse(nr.eligible(a, a))  # exact copies are not near-duplicate stratum

    def test_marked_token_and_shingle_shares_use_word_positions(self):
        text = "Ada Writer went home yesterday and now"
        self.assertEqual(nr.marking_density(text, [(0, 10)]), (2 / 7, 2 / 3))

    def test_retry_merges_only_original_category_page_ids(self):
        members = [{"pageid": 1}, {"pageid": 2}, {"pageid": 3}]
        def row(pageid):
            return {"pageid": pageid, "response": {"query": {"pages": [{"pageid": pageid}]}}}
        combined, missing = nr.merge_rows(members, [row(1)], [row(3)])
        self.assertEqual([r["pageid"] for r in combined], [1, 3])
        self.assertEqual(missing, [2])
        with self.assertRaises(ValueError):
            nr.merge_rows(members, [row(1)], [row(1)])
        with self.assertRaises(ValueError):
            nr.merge_rows(members, [row(1)], [row(4)])


if __name__ == "__main__":
    unittest.main()
