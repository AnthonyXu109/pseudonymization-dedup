"""Fixed-release TAB curation-order follow-up; outputs only to the external disk."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
from datetime import datetime, timezone
from pathlib import Path

from analysis import (
    LEGACY_XCUR,
    cluster_count_interval,
    cluster_ratio_interval,
    evaluate_decision,
    released_linkage,
    union_groups,
)


PROJECT = Path(os.environ.get("XCUR_PROJECT", "."))
TASK = PROJECT / "workspace" / "niw-extreme-curation-2026"
DEFAULT_TAB = PROJECT / "data" / "20261001" / "tab"
DEFAULT_OUTPUT = PROJECT / "runs" / "20261001" / "order-tab-gold-v1.json"
STRATEGIES = ("RAW", "SUR-DOC", "SUR-CORPUS")
FINAL_RELEASE = "SUR-DOC"
THRESHOLD = 0.7
SEED = 20261001
MAX_RSS_BYTES = 6 * 1024**3


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _interval(values, groups, count, reps):
    fn = cluster_count_interval if count else cluster_ratio_interval
    estimate, lower, upper = fn(values, groups, reps=reps, seed=SEED)
    return {"estimate": estimate, "ci95": [lower, upper], "bootstrap_clusters": len(set(groups)),
            "bootstrap_replicates": reps}


def create_report(docs: list[dict], reps: int = 2000) -> dict:
    """Compare internal decisions while holding the published representation fixed.

    This is a follow-up on a previously explored corpus, not an independent test.
    The union graph in each pairwise contrast defines resampling clusters.
    """
    if reps <= 0:
        raise ValueError("reps must be positive")
    decisions = {}
    rows = {}
    originals = [(i, d["case"]) for i, d in enumerate(docs) if not d["is_copy"]]
    copies = [(i, d["case"]) for i, d in enumerate(docs) if d["is_copy"]]
    if len({case for _, case in originals}) != len(originals):
        raise ValueError("original case IDs must be unique")

    for strategy in STRATEGIES:
        result = evaluate_decision(docs, strategy, THRESHOLD)
        decisions[strategy] = result
        link = released_linkage(docs, result["kept_indices"], FINAL_RELEASE)
        rows[strategy] = {
            "decision": strategy,
            "release": FINAL_RELEASE,
            "direct_pair_recall": result["pair_recall"],
            "pipeline_recall": result["pipeline_recall"],
            "copies_detected_in_component": sum(result["detected"]),
            "copies_total": len(copies),
            "distinct_cases_lost": len(result["lost_cases"]),
            "lost_case_ids": result["lost_cases"],
            "records_retained": len(result["kept_indices"]),
            "candidate_edges": result["edge_count"],
            "released_cross_case_person_string": {k: _finite(v) for k, v in link.items()},
        }

    contrasts = {}
    baseline = decisions["RAW"]
    original_cases = [case for _, case in originals]
    for strategy in STRATEGIES[1:]:
        current = decisions[strategy]
        graph_groups = union_groups(baseline["components"], current["components"])
        copy_values = [int(a) - int(b) for a, b in zip(current["detected"], baseline["detected"])]
        original_values = [int(case in current["lost_cases"]) -
                           int(case in baseline["lost_cases"]) for case in original_cases]
        contrasts[f"{strategy}_minus_RAW"] = {
            "pipeline_recall_delta": _interval(copy_values, [graph_groups[i] for i, _ in copies], False, reps),
            "lost_case_count_delta": _interval(original_values, [graph_groups[i] for i, _ in originals], True, reps),
            "copy_unit_count": len(copies),
            "original_case_count": len(originals),
        }

    return {
        "analysis": "controlled_internal_decision_order_fixed_document_release",
        "status": "follow_up_on_seen_data",
        "threshold": THRESHOLD,
        "shingle_words": 5,
        "normalization": "plain",
        "gold_span_mode": "first_annotator",
        "fixed_release": FINAL_RELEASE,
        "retention_policy": "retain_each_distinct_TAB_case; merge_copies_of_same_case",
        "rows": rows,
        "contrasts": contrasts,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs(tab_dir: Path) -> list[dict]:
    manifest = json.loads((TASK / "v7" / "pseudonymization-dedup" / "inputs_manifest.json").read_text())
    expected = {entry["file"]: entry for entry in manifest["files"] if entry["file"].startswith("echr_")}
    if set(expected) != {"echr_train.json", "echr_dev.json", "echr_test.json"}:
        raise ValueError("v7 TAB manifest is incomplete")
    checked = []
    for name in sorted(expected):
        path = tab_dir / name
        size, digest = path.stat().st_size, _sha256(path)
        if (size, digest) != (expected[name]["bytes"], expected[name]["sha256"]):
            raise ValueError(f"TAB input mismatch: {path}")
        checked.append({"path": str(path), "bytes": size, "sha256": digest})
    return checked


def _external_path(path: Path) -> Path:
    target = path.resolve()
    if not target.is_relative_to(PROJECT):
        raise ValueError(f"path must remain under the external project: {target}")
    if not PROJECT.is_dir() or not os.path.ismount(PROJECT.parents[2]):
        raise RuntimeError("external Documents volume is unavailable")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tab-dir", type=Path, default=DEFAULT_TAB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    tab_dir, output = _external_path(args.tab_dir), _external_path(args.output)
    if output.exists() or output.with_suffix(output.suffix + ".partial").exists():
        raise FileExistsError("refusing to overwrite an existing run or partial file")
    checked = verify_inputs(tab_dir)
    print("TAB inputs verified; building the fixed corpus", flush=True)

    import common  # imported from the pinned v7 source via analysis
    import build_corpus

    common.TAB = str(tab_dir)
    common.NORM["mode"] = "plain"
    common.NGRAM["n"] = 5
    build_corpus.TAB = str(tab_dir)
    docs = build_corpus.build(mode="first")
    if (len(docs), sum(bool(d["is_copy"]) for d in docs)) != (1668, 400):
        raise ValueError("corpus shape differs from the fixed protocol")
    print("Running three exact-Jaccard decision graphs, single process", flush=True)
    report = create_report(docs)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss > MAX_RSS_BYTES:
        raise MemoryError(f"peak resident memory {rss} exceeds 6 GiB")
    report["provenance"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tab_inputs": checked,
        "code": [{"path": str(path), "sha256": _sha256(path)} for path in
                 [Path(__file__), Path(__file__).with_name("analysis.py"),
                  Path(__file__).with_name("PROTOCOL.md"), LEGACY_XCUR / "common.py",
                  LEGACY_XCUR / "build_corpus.py", LEGACY_XCUR / "evalcore.py"]],
        "source_candidate_pdf_sha256": _sha256(TASK / "v7" / "pseudonymization-dedup-v7-draft-2.pdf"),
        "peak_rss_bytes": rss,
        "random_seed": SEED,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    with partial.open("x") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(partial, output)
    print(f"Wrote {output}; sha256={_sha256(output)}; peak RSS={rss}", flush=True)


if __name__ == "__main__":
    main()
