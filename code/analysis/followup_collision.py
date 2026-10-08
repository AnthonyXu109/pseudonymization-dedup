"""Audit 32-bit shingle collisions against full textual shingles on TAB."""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import analysis
import run_order
from followup_science import decision_from_matrix, full_shingles


STRATEGIES = ("RAW", "MASK", "TYPE", "SUR-CORPUS", "SUR-DOC")
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)
OUTPUT = run_order.PROJECT / "runs" / "20261004" / "shingle-collision-v1.json"
PROTOCOL = Path(__file__).with_name("FOLLOWUP_SCIENCE_PROTOCOL_20261004.md")


def count_collisions(sets: list[set[str]], hasher=None) -> dict:
    """Count distinct textual shingles merged under a finite hash ID."""
    if hasher is None:
        hasher = lambda text: zlib.crc32(text.encode())
    distinct = set().union(*sets) if sets else set()
    seen_ids = {hasher(text) for text in distinct}
    return {
        "distinct_text_shingles": len(distinct),
        "distinct_crc_ids": len(seen_ids),
        "true_collisions": len(distinct) - len(seen_ids),
    }


def audit_strategy(docs: list[dict], strategy: str) -> dict:
    """Measure pair decisions and, for pivotal rows, the full graph."""
    full_sets = [full_shingles(analysis.release(
        d["text"], d["spans"], strategy, d["key"], d["case"]
    )) for d in docs]
    crc_sets = [{zlib.crc32(s.encode()) for s in item} for item in full_sets]
    collisions = count_collisions(full_sets)
    originals = {d["case"]: i for i, d in enumerate(docs) if not d["is_copy"]}
    pair_results = {str(theta): {"status_changes": 0, "offending_case_ids": []}
                    for theta in THRESHOLDS}
    max_abs = 0.0
    for i, d in enumerate(docs):
        if not d["is_copy"]:
            continue
        j = originals[d["case"]]
        full_a, full_b = full_sets[i], full_sets[j]
        crc_a, crc_b = crc_sets[i], crc_sets[j]
        full_score = len(full_a & full_b) / len(full_a | full_b)
        crc_score = len(crc_a & crc_b) / len(crc_a | crc_b)
        max_abs = max(max_abs, abs(full_score - crc_score))
        for theta in THRESHOLDS:
            cell = pair_results[str(theta)]
            if (full_score >= theta) != (crc_score >= theta):
                cell["status_changes"] += 1
                cell["offending_case_ids"].append(d["case"])
    result = {"collisions": collisions, "original_copy_pair_count": 400,
              "max_pair_jaccard_absolute_difference": max_abs,
              "threshold_pair_decisions": pair_results}
    if strategy in ("RAW", "SUR-DOC"):
        full_matrix = analysis.jaccard_matrix(full_sets)
        crc_matrix = analysis.jaccard_matrix(crc_sets)
        full = decision_from_matrix(docs, full_matrix, 0.70)
        hashed = decision_from_matrix(docs, crc_matrix, 0.70)
        edge_differences = int(np.count_nonzero(np.triu(
            (full_matrix >= 0.70) != (crc_matrix >= 0.70), 1
        )))
        result["full_graph_at_0_70"] = {
            "edge_status_changes": edge_differences,
            "component_assignment_changes": sum(a != b for a, b in zip(
                full["components"], hashed["components"])),
            "full_copies_detected": full["copies_detected"],
            "crc_copies_detected": hashed["copies_detected"],
            "full_distinct_cases_lost": len(full["lost_cases"]),
            "crc_distinct_cases_lost": len(hashed["lost_cases"]),
            "full_minus_crc_lost_case_ids": sorted(set(full["lost_cases"]) - set(hashed["lost_cases"])),
            "crc_minus_full_lost_case_ids": sorted(set(hashed["lost_cases"]) - set(full["lost_cases"])),
        }
    del full_sets, crc_sets
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = run_order._external_path(args.output)
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError("refusing to overwrite an existing result")
    checked = run_order.verify_inputs(run_order.DEFAULT_TAB)
    import common
    import build_corpus
    common.TAB = str(run_order.DEFAULT_TAB)
    common.NORM["mode"] = "plain"
    common.NGRAM["n"] = 5
    build_corpus.TAB = str(run_order.DEFAULT_TAB)
    docs = build_corpus.build(mode="first")
    if (len(docs), sum(d["is_copy"] for d in docs)) != (1668, 400):
        raise ValueError("frozen TAB corpus shape changed")
    rows = {}
    for strategy in STRATEGIES:
        rows[strategy] = audit_strategy(docs, strategy)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if rss > run_order.MAX_RSS_BYTES:
            raise MemoryError(f"peak RSS {rss} exceeds 6 GiB")
        print(f"Audited {strategy}; true collisions={rows[strategy]['collisions']['true_collisions']}; "
              f"peak RSS={rss}", flush=True)
    report = {
        "analysis": "TAB_full_shingle_vs_32bit_CRC_check",
        "status": "exploratory_follow_up_on_seen_TAB_data",
        "thresholds": list(THRESHOLDS),
        "rows": rows,
        "provenance": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "tab_inputs": checked,
            "code": [{"path": str(p), "sha256": run_order._sha256(p)} for p in
                     [Path(__file__), Path(__file__).with_name("followup_science.py"),
                      Path(__file__).with_name("analysis.py"), PROTOCOL,
                      run_order.LEGACY_XCUR / "common.py", run_order.LEGACY_XCUR / "build_corpus.py"]],
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("x") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(partial, output)
    print(f"Wrote {output}; sha256={run_order._sha256(output)}", flush=True)


if __name__ == "__main__":
    main()
