"""Prespecified gold-span omission sensitivity on the fixed TAB copies."""

from __future__ import annotations

import argparse
import json
import os
import resource
from datetime import datetime, timezone
from pathlib import Path

from analysis import evaluate_decision
from run_order import (
    DEFAULT_TAB,
    LEGACY_XCUR,
    MAX_RSS_BYTES,
    PROJECT,
    TASK,
    _external_path,
    _sha256,
    verify_inputs,
)
from sensitivity import perturb_docs


DEFAULT_OUTPUT = PROJECT / "runs" / "20261001" / "sensitivity-tab-gold-v1.json"
BASELINE = PROJECT / "runs" / "20261001" / "order-tab-gold-v1.json"
BASELINE_SHA256 = "448d4f5142776526d194d7a049b7b64c0d74021370e145d960b8638821780f08"


def make_seed_rows(docs: list[dict], mode: str, percent: int, seed: int) -> list[dict]:
    changed, qa = perturb_docs(docs, mode, percent, seed)
    rows = []
    for strategy in ("SUR-DOC", "SUR-CORPUS"):
        result = evaluate_decision(changed, strategy, 0.7)
        rows.append({
            "omission_mode": mode,
            "percent_requested": percent,
            "seed": seed,
            "decision": strategy,
            "direct_pair_recall": result["pair_recall"],
            "pipeline_recall": result["pipeline_recall"],
            "copies_detected_in_component": sum(result["detected"]),
            "distinct_cases_lost": len(result["lost_cases"]),
            "lost_case_ids": result["lost_cases"],
            "qa": qa,
        })
    return rows


def _describe(values: list[float]) -> dict:
    return {"mean": sum(values) / len(values), "min": min(values), "max": max(values),
            "n_seeds": len(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tab-dir", type=Path, default=DEFAULT_TAB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    tab_dir, output = _external_path(args.tab_dir), _external_path(args.output)
    if output.exists() or output.with_suffix(output.suffix + ".partial").exists():
        raise FileExistsError("refusing to overwrite an existing run or partial file")
    if _sha256(BASELINE) != BASELINE_SHA256:
        raise ValueError("fixed order-run baseline has changed")
    checked = verify_inputs(tab_dir)

    import common
    import build_corpus

    common.TAB = str(tab_dir)
    common.NORM["mode"] = "plain"
    common.NGRAM["n"] = 5
    build_corpus.TAB = str(tab_dir)
    docs = build_corpus.build(mode="first")
    if (len(docs), sum(bool(d["is_copy"]) for d in docs)) != (1668, 400):
        raise ValueError("corpus shape differs from the fixed protocol")

    rows = []
    for mode in ("entity", "mention"):
        for seed in range(5):
            rows.extend(make_seed_rows(docs, mode, 20, seed))
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if rss > MAX_RSS_BYTES:
                raise MemoryError(f"peak resident memory {rss} exceeds 6 GiB")
            print(f"Completed {mode} omission seed {seed}; peak RSS {rss}", flush=True)

    base = json.loads(BASELINE.read_text())["rows"]
    summary = {}
    for mode in ("entity", "mention"):
        selection = [row for row in rows if row["omission_mode"] == mode]
        name = f"{mode}_20_percent"
        summary[name] = {}
        for strategy in ("SUR-DOC", "SUR-CORPUS"):
            group = [row for row in selection if row["decision"] == strategy]
            summary[name][strategy] = {
                "pipeline_recall": _describe([r["pipeline_recall"] for r in group]),
                "direct_pair_recall": _describe([r["direct_pair_recall"] for r in group]),
                "distinct_cases_lost": _describe([r["distinct_cases_lost"] for r in group]),
                "pipeline_recall_delta_from_unperturbed": _describe([
                    r["pipeline_recall"] - base[strategy]["pipeline_recall"] for r in group]),
            }
        by_seed = {seed: {r["decision"]: r for r in selection if r["seed"] == seed} for seed in range(5)}
        summary[name]["SUR-CORPUS_minus_SUR-DOC"] = {
            "pipeline_recall_delta": _describe([
                pair["SUR-CORPUS"]["pipeline_recall"] - pair["SUR-DOC"]["pipeline_recall"]
                for pair in by_seed.values()]),
            "distinct_cases_lost_delta": _describe([
                pair["SUR-CORPUS"]["distinct_cases_lost"] - pair["SUR-DOC"]["distinct_cases_lost"]
                for pair in by_seed.values()]),
        }

    report = {
        "analysis": "controlled_marked_span_omission_sensitivity",
        "status": "follow_up_on_seen_data_not_a_detector_error_rate_estimate",
        "percent_requested": 20,
        "seeds": list(range(5)),
        "decision_strategies": ["SUR-DOC", "SUR-CORPUS"],
        "baseline": {s: {k: base[s][k] for k in
                          ("pipeline_recall", "direct_pair_recall", "distinct_cases_lost")}
                     for s in ("SUR-DOC", "SUR-CORPUS")},
        "rows": rows,
        "summary": summary,
        "provenance": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "tab_inputs": checked,
            "baseline_path": str(BASELINE),
            "baseline_sha256": BASELINE_SHA256,
            "code": [{"path": str(path), "sha256": _sha256(path)} for path in
                     [Path(__file__), Path(__file__).with_name("sensitivity.py"),
                      Path(__file__).with_name("analysis.py"), Path(__file__).with_name("PROTOCOL.md"),
                      LEGACY_XCUR / "common.py", LEGACY_XCUR / "build_corpus.py"]],
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    with partial.open("x") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(partial, output)
    print(f"Wrote {output}; sha256={_sha256(output)}", flush=True)


if __name__ == "__main__":
    main()
