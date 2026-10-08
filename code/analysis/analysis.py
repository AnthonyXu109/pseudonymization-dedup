"""Small, independent follow-up calculations over the v7 fixed corpus.

The v7 modules are imported as read-only dependencies. All new output paths
are supplied by the caller and must be on the external project volume.
"""

from __future__ import annotations

import sys
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


LEGACY_XCUR = Path(__file__).resolve().parent.parent / "v7" / "pseudonymization-dedup" / "xcur"
if str(LEGACY_XCUR) not in sys.path:
    sys.path.insert(0, str(LEGACY_XCUR))

from evalcore import linkage  # noqa: E402: pinned, read-only v7 implementation
from common import components, jaccard_matrix, release, shingles  # noqa: E402


def evaluate_membership(
    case_ids: list[Any],
    is_copy: list[bool],
    components: list[int],
    pair_hits: list[bool],
) -> dict[str, Any]:
    """Score direct and transitive detection and list lost original cases.

    ``components`` contains canonical representative indices, matching v7's
    ``common.components`` convention. ``pair_hits`` follows copy order.
    """
    n = len(case_ids)
    if len(is_copy) != n or len(components) != n:
        raise ValueError("document vectors have different lengths")
    copies = [i for i, flag in enumerate(is_copy) if flag]
    if len(pair_hits) != len(copies):
        raise ValueError("pair_hits must have one value per copy")
    originals = {case_ids[i]: i for i, flag in enumerate(is_copy) if not flag}
    if len(originals) != n - len(copies):
        raise ValueError("original case IDs must be unique")
    if any(case_ids[i] not in originals for i in copies):
        raise ValueError("every copy needs its original")
    if any(not 0 <= root < n or components[root] != root for root in components):
        raise ValueError("components must use canonical representative indices")

    kept = [i for i, root in enumerate(components) if i == root]
    surviving_cases = {case_ids[i] for i in kept}
    detected = [components[originals[case_ids[i]]] == components[i] for i in copies]
    return {
        "pipeline_recall": sum(detected) / len(copies) if copies else None,
        "pair_recall": sum(pair_hits) / len(copies) if copies else None,
        "detected": detected,
        "kept_indices": kept,
        "lost_cases": sorted(set(originals) - surviving_cases),
    }


def union_groups(first: list[int], second: list[int]) -> list[int]:
    """Group documents that meet in either decision graph for resampling."""
    if len(first) != len(second):
        raise ValueError("component vectors have different lengths")
    parent = list(range(len(first)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def unite(a: int, b: int) -> None:
        a, b = root(a), root(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for groups in (first, second):
        seen: dict[int, int] = {}
        for i, label in enumerate(groups):
            if label in seen:
                unite(i, seen[label])
            else:
                seen[label] = i
    return [root(i) for i in range(len(first))]


def released_linkage(docs: list[dict[str, Any]], kept_indices: list[int], strategy: str) -> dict[str, Any]:
    """Recalculate v7's cross-case person-string measure on retained texts.

    V7's ``linkage`` excludes rows marked ``is_copy``. A retained copy is a
    published document, so the temporary view marks every survivor as active.
    The source documents are never modified.
    """
    if len(set(kept_indices)) != len(kept_indices):
        raise ValueError("kept_indices contains duplicates")
    if any(not 0 <= i < len(docs) for i in kept_indices):
        raise ValueError("kept index out of range")
    retained = [{**docs[i], "is_copy": False} for i in kept_indices]
    return linkage(retained, strategy, "PERSON")


def evaluate_decision(docs: list[dict[str, Any]], strategy: str, theta: float) -> dict[str, Any]:
    """Run v7's exact shingle/Jaccard graph, then score its components."""
    if not 0 < theta <= 1:
        raise ValueError("theta must be in (0, 1]")
    representations = [shingles(release(d["text"], d["spans"], strategy, d["key"], d["case"]))
                       for d in docs]
    jac = jaccard_matrix(representations)
    index_by_case = {d["case"]: i for i, d in enumerate(docs) if not d["is_copy"]}
    pair_hits = [bool(jac[index_by_case[d["case"]], i] >= theta)
                 for i, d in enumerate(docs) if d["is_copy"]]
    edges = np.argwhere(np.triu(jac >= theta, 1)).tolist()
    memberships = components(len(docs), edges)
    result = evaluate_membership(
        [d["case"] for d in docs], [bool(d["is_copy"]) for d in docs], memberships, pair_hits
    )
    result["components"] = memberships
    result["edge_count"] = len(edges)
    return result


def _resampled_values(values: list[float], groups: list[int], reps: int, seed: int, ratio: bool) -> list[float]:
    if len(values) != len(groups) or not values or reps <= 0:
        raise ValueError("nonempty equal-length values/groups and positive reps required")
    by_group: dict[int, list[float]] = defaultdict(list)
    for value, group in zip(values, groups):
        by_group[group].append(value)
    group_arrays = list(by_group.values())
    rng = random.Random(seed)
    out = []
    for _ in range(reps):
        draw = [group_arrays[rng.randrange(len(group_arrays))] for _ in group_arrays]
        total = sum(sum(g) for g in draw)
        out.append(total / sum(len(g) for g in draw) if ratio else total)
    return sorted(out)


def cluster_ratio_interval(
    values: list[float], groups: list[int], reps: int = 2000, seed: int = 20261001
) -> tuple[float, float, float]:
    """Paired mean difference with whole union-component bootstrap clusters."""
    draws = _resampled_values(values, groups, reps, seed, True)
    return sum(values) / len(values), draws[int(0.025 * (reps - 1))], draws[int(0.975 * (reps - 1))]


def cluster_count_interval(
    values: list[float], groups: list[int], reps: int = 2000, seed: int = 20261001
) -> tuple[float, float, float]:
    """Paired total-count difference with whole union-component resampling."""
    draws = _resampled_values(values, groups, reps, seed, False)
    return sum(values), draws[int(0.025 * (reps - 1))], draws[int(0.975 * (reps - 1))]
