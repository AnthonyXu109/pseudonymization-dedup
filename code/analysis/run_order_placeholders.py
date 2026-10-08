"""Add MASK and TYPE internal decisions to the frozen TAB fixed-release follow-up."""

from __future__ import annotations

import argparse
import json
import os
import resource
from datetime import datetime, timezone
from pathlib import Path

import run_order


DEFAULT_OUTPUT = run_order.PROJECT / "runs" / "20261004" / "order-placeholders-v1.json"
DECISIONS = ("RAW", "SUR-DOC", "SUR-CORPUS", "MASK", "TYPE")
PROTOCOL = Path(__file__).with_name("PLACEHOLDER_DECISION_PROTOCOL_20261004.md")


def create_placeholder_report(docs: list[dict], reps: int = 2000) -> dict:
    """Reuse the original analysis unchanged, then restore its original strategy set."""
    original = run_order.STRATEGIES
    if original != DECISIONS[:3]:
        raise RuntimeError("original order-decision protocol has changed")
    try:
        run_order.STRATEGIES = DECISIONS
        report = run_order.create_report(docs, reps=reps)
    finally:
        run_order.STRATEGIES = original
    report["addendum"] = "frozen_fixed_release_placeholder_decisions_20261004"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tab-dir", type=Path, default=run_order.DEFAULT_TAB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    tab_dir = run_order._external_path(args.tab_dir)
    output = run_order._external_path(args.output)
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError("refusing to overwrite an existing run or partial file")
    checked = run_order.verify_inputs(tab_dir)
    print("TAB input hashes verified; building frozen corpus", flush=True)

    import common
    import build_corpus

    common.TAB = str(tab_dir)
    common.NORM["mode"] = "plain"
    common.NGRAM["n"] = 5
    build_corpus.TAB = str(tab_dir)
    docs = build_corpus.build(mode="first")
    if (len(docs), sum(bool(d["is_copy"]) for d in docs)) != (1668, 400):
        raise ValueError("corpus shape differs from frozen protocol")
    print("Evaluating five exact-Jaccard decision graphs, single process", flush=True)
    report = create_placeholder_report(docs)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss > run_order.MAX_RSS_BYTES:
        raise MemoryError(f"peak RSS {rss} exceeds 6 GiB")
    code_files = [Path(__file__), Path(__file__).with_name("run_order.py"),
                  Path(__file__).with_name("analysis.py"), PROTOCOL,
                  run_order.LEGACY_XCUR / "common.py",
                  run_order.LEGACY_XCUR / "build_corpus.py",
                  run_order.LEGACY_XCUR / "evalcore.py"]
    report["provenance"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tab_inputs": checked,
        "code": [{"path": str(path), "sha256": run_order._sha256(path)} for path in code_files],
        "source_candidate_pdf_sha256": run_order._sha256(
            run_order.TASK / "v7" / "pseudonymization-dedup-v7-draft-2.pdf"),
        "peak_rss_bytes": rss,
        "random_seed": run_order.SEED,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("x") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(partial, output)
    print(f"Wrote {output}; sha256={run_order._sha256(output)}; peak RSS={rss}", flush=True)


if __name__ == "__main__":
    main()
