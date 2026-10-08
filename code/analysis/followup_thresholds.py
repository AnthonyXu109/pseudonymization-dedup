"""Fixed-release TAB threshold sweep, separated from all historical runs."""

from __future__ import annotations

import argparse
import json
import os
import resource
from datetime import datetime, timezone
from pathlib import Path

import analysis
import run_order
from followup_science import decision_from_matrix


STRATEGIES = ("RAW", "MASK", "TYPE", "SUR-CORPUS", "SUR-DOC")
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)
OUTPUT = run_order.PROJECT / "runs" / "20261004" / "fixed-release-thresholds-v1.json"
PROTOCOL = Path(__file__).with_name("FOLLOWUP_SCIENCE_PROTOCOL_20261004.md")


def tab_threshold_report(docs: list[dict], matrices: dict, thresholds=THRESHOLDS,
                         reps: int = 2000) -> dict:
    """Evaluate fixed strategy matrices without recomputing or tuning them."""
    if not matrices or "RAW" not in matrices or reps <= 0:
        raise ValueError("RAW matrix and positive resampling count required")
    if not thresholds or any(not 0 < x <= 1 for x in thresholds):
        raise ValueError("invalid thresholds")
    copies = [(i, d["case"]) for i, d in enumerate(docs) if d["is_copy"]]
    originals = [(i, d["case"]) for i, d in enumerate(docs) if not d["is_copy"]]
    rows, contrasts = {}, {}
    for theta in thresholds:
        key = str(theta)
        evaluated = {name: decision_from_matrix(docs, jac, theta)
                     for name, jac in matrices.items()}
        rows[key] = {
            name: {
                "copies_detected": scored["copies_detected"],
                "copies_total": len(copies),
                "pair_recall": scored["pair_recall"],
                "pipeline_recall": scored["pipeline_recall"],
                "distinct_cases_lost": len(scored["lost_cases"]),
                "lost_case_ids": scored["lost_cases"],
                "retained_records": scored["retained_records"],
                "candidate_edges": scored["edge_count"],
            }
            for name, scored in evaluated.items()
        }
        baseline = evaluated["RAW"]
        contrasts[key] = {}
        for name, scored in evaluated.items():
            if name == "RAW":
                continue
            groups = analysis.union_groups(baseline["components"], scored["components"])
            copy_values = [int(a) - int(b) for a, b in
                           zip(scored["detected"], baseline["detected"])]
            lost_values = [int(case in scored["lost_cases"]) -
                           int(case in baseline["lost_cases"]) for _, case in originals]
            contrast = {}
            for field, values, indices, interval in (
                ("pipeline_recall_delta", copy_values, copies, analysis.cluster_ratio_interval),
                ("lost_case_count_delta", lost_values, originals, analysis.cluster_count_interval),
            ):
                estimate, lower, upper = interval(
                    values, [groups[i] for i, _ in indices], reps=reps, seed=run_order.SEED
                )
                contrast[field] = {"estimate": estimate, "ci95": [lower, upper]}
            contrast["bootstrap_clusters"] = len(set(groups))
            contrasts[key][f"{name}_minus_RAW"] = contrast
    return {
        "analysis": "fixed_release_tab_threshold_sensitivity",
        "status": "exploratory_follow_up_on_seen_TAB_data",
        "thresholds": list(thresholds),
        "decision_representations": list(matrices),
        "fixed_final_release_form": "SUR-DOC",
        "shingle_words": 5,
        "shingle_id_bits": 32,
        "bootstrap_replicates": reps,
        "bootstrap_seed": run_order.SEED,
        "rows": rows,
        "contrasts": contrasts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = run_order._external_path(args.output)
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError("refusing to overwrite a prior result or partial")
    checked = run_order.verify_inputs(run_order.DEFAULT_TAB)
    print("TAB input hashes verified", flush=True)

    import common
    import build_corpus

    common.TAB = str(run_order.DEFAULT_TAB)
    common.NORM["mode"] = "plain"
    common.NGRAM["n"] = 5
    build_corpus.TAB = str(run_order.DEFAULT_TAB)
    docs = build_corpus.build(mode="first")
    if (len(docs), sum(d["is_copy"] for d in docs)) != (1668, 400):
        raise ValueError("frozen TAB corpus shape changed")
    matrices = {}
    for strategy in STRATEGIES:
        sets = [analysis.shingles(analysis.release(
            d["text"], d["spans"], strategy, d["key"], d["case"]
        )) for d in docs]
        matrices[strategy] = analysis.jaccard_matrix(sets)
        print(f"Computed {strategy} matrix", flush=True)
    report = tab_threshold_report(docs, matrices)
    old = json.loads((run_order.PROJECT / "runs" / "20261004" /
                      "order-placeholders-v1.json").read_text())
    for name in STRATEGIES:
        row = report["rows"]["0.7"][name]
        reference = old["rows"][name]
        if (row["copies_detected"], row["distinct_cases_lost"], row["retained_records"]) != (
            reference["copies_detected_in_component"],
            reference["distinct_cases_lost"], reference["records_retained"]
        ):
            raise AssertionError(f"0.70 does not reproduce Table IV: {name}")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if peak > run_order.MAX_RSS_BYTES:
        raise MemoryError(f"peak RSS {peak} exceeds 6 GiB")
    code = [Path(__file__), Path(__file__).with_name("followup_science.py"),
            Path(__file__).with_name("analysis.py"), PROTOCOL,
            run_order.LEGACY_XCUR / "common.py", run_order.LEGACY_XCUR / "build_corpus.py"]
    report["provenance"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tab_inputs": checked,
        "code": [{"path": str(p), "sha256": run_order._sha256(p)} for p in code],
        "reference_result_sha256": run_order._sha256(
            run_order.PROJECT / "runs" / "20261004" / "order-placeholders-v1.json"
        ),
        "peak_rss_bytes": peak,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("x") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(partial, output)
    print(f"Wrote {output}; sha256={run_order._sha256(output)}; peak RSS={peak}", flush=True)


if __name__ == "__main__":
    main()
