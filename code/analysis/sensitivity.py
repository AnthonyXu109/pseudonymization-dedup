"""Controlled omissions of TAB's marked spans, without changing the documents."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any


def perturb_docs(docs: list[dict[str, Any]], mode: str, percent: int, seed: int):
    """Drop an exact percentage of corpus-wide entity or mention units.

    Entity units are (document, gold entity ID), so all repeated mentions of
    an ID in that document share a marking decision. Mention units are
    independent. No randomness affects text, copy construction, or the
    pseudonymization key. The returned documents do not mutate the input.
    """
    if mode not in {"entity", "mention"} or not 0 <= percent <= 100:
        raise ValueError("mode must be entity/mention and percent within 0..100")
    units = []
    memberships = []
    for i, doc in enumerate(docs):
        seen = {}
        row = []
        for j, span in enumerate(doc["spans"]):
            if mode == "mention":
                unit = (i, j)
                units.append(unit)
            else:
                ent = span["ent"]
                if ent not in seen:
                    seen[ent] = (i, len(seen))
                    units.append(seen[ent])
                unit = seen[ent]
            row.append(unit)
        memberships.append(row)

    n_drop = round(percent / 100 * len(units))
    chosen = set(random.Random(seed).sample(units, n_drop))
    result = []
    original_pairs = inconsistent_pairs = 0
    dropped_mentions = 0
    for doc, units_for_spans in zip(docs, memberships):
        retained = []
        entity_total = Counter()
        entity_kept = Counter()
        for span, unit in zip(doc["spans"], units_for_spans):
            ent = span["ent"]
            entity_total[ent] += 1
            if unit in chosen:
                dropped_mentions += 1
            else:
                retained.append(span)
                entity_kept[ent] += 1
        for ent, total in entity_total.items():
            kept = entity_kept[ent]
            original_pairs += total * (total - 1) // 2
            inconsistent_pairs += kept * (total - kept)
        result.append({**doc, "spans": retained})
    return result, {
        "mode": mode,
        "percent_requested": percent,
        "seed": seed,
        "units_total": len(units),
        "units_dropped": n_drop,
        "dropped_entity_units": n_drop if mode == "entity" else None,
        "dropped_mention_units": n_drop if mode == "mention" else None,
        "mentions_total": sum(len(doc["spans"]) for doc in docs),
        "mentions_dropped": dropped_mentions,
        "within_document_same_entity_mention_pairs": original_pairs,
        "inconsistent_within_document_pairs": inconsistent_pairs,
        "inconsistent_within_document_pair_fraction": inconsistent_pairs / original_pairs if original_pairs else 0.0,
    }
