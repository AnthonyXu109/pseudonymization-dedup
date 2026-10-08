"""Small shared measurements for the 2026-10-04 scientific follow-up."""

from __future__ import annotations

import re
import zlib

import numpy as np

import analysis
import natural_revision
from common import words


def decision_from_matrix(docs: list[dict], jac: np.ndarray, theta: float) -> dict:
    """Score a precomputed Jaccard matrix at one exact decision threshold."""
    if not 0 < theta <= 1 or jac.shape != (len(docs), len(docs)):
        raise ValueError("invalid threshold or matrix shape")
    originals = {d["case"]: i for i, d in enumerate(docs) if not d["is_copy"]}
    copies = [(i, d) for i, d in enumerate(docs) if d["is_copy"]]
    pair_hits = [bool(jac[originals[d["case"]], i] >= theta) for i, d in copies]
    edges = np.argwhere(np.triu(jac >= theta, 1)).tolist()
    memberships = analysis.components(len(docs), edges)
    scored = analysis.evaluate_membership(
        [d["case"] for d in docs],
        [bool(d["is_copy"]) for d in docs],
        memberships,
        pair_hits,
    )
    scored["copies_detected"] = sum(scored["detected"])
    scored["retained_records"] = len(scored["kept_indices"])
    scored["edge_count"] = len(edges)
    scored["components"] = memberships
    return scored


def full_shingles(text: str, n: int = 5) -> set[str]:
    """Full strings with the pinned implementation's short-document rule."""
    if n <= 0:
        raise ValueError("n must be positive")
    tokens = words(text)
    return {" ".join(tokens[i:i + n]) for i in range(max(1, len(tokens) - n + 1))}


def _crc(shingle: str) -> int:
    return zlib.crc32(shingle.encode())


def pair_hash_comparison(text_a: str, text_b: str, hasher=_crc) -> dict:
    """Compare collision-free and hashed Jaccard on an exact text pair."""
    first, second = full_shingles(text_a), full_shingles(text_b)
    hashed_first = {hasher(s) for s in first}
    hashed_second = {hasher(s) for s in second}
    full_union = len(first | second)
    hash_union = len(hashed_first | hashed_second)
    return {
        "full_jaccard": len(first & second) / full_union if full_union else 0.0,
        "hashed_jaccard": len(hashed_first & hashed_second) / hash_union if hash_union else 0.0,
    }


def policy_spans(text: str, title: str, policy: str) -> list[tuple[int, int]]:
    """Return reproducible, nested Wikipedia marking scopes."""
    if policy == "DENSE":
        return natural_revision.mark_spans(text, title)
    if policy not in ("TITLE_ONLY", "TITLE_NAME"):
        raise ValueError(policy)
    title = re.sub(r"\s*\([^)]*\)$", "", title).strip()
    proposed = []
    if title:
        pattern = re.compile(r"(?<!\w)" + re.escape(title).replace(r"\ ", r"\s+") + r"(?!\w)", re.I)
        proposed.extend(match.span() for match in pattern.finditer(text))
    if policy == "TITLE_NAME":
        proposed.extend(match.span() for match in natural_revision.NAME.finditer(text))
    chosen = []
    for a, b in sorted(proposed, key=lambda x: (-(x[1] - x[0]), x[0])):
        if not any(a < d and c < b for c, d in chosen):
            chosen.append((a, b))
    return sorted(chosen)
