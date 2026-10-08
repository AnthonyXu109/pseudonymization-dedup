"""Re-evaluate the frozen Wikipedia revision pairs at several marking scopes."""

from __future__ import annotations

import argparse
import json
import os
import random
import resource
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import analysis
import natural_revision
import run_order
from followup_science import policy_spans


DATA = run_order.PROJECT / "data" / "20261004" / "natural-revisions"
FROZEN_RESULT = run_order.PROJECT / "runs" / "20261004" / "natural-revision-v1.json"
OUTPUT = run_order.PROJECT / "runs" / "20261004" / "natural-marking-scope-v1.json"
PROTOCOL = Path(__file__).with_name("FOLLOWUP_SCIENCE_PROTOCOL_20261004.md")
NER_LABELS = frozenset(("PERSON", "ORG", "GPE", "LOC", "NORP", "DATE", "TIME", "QUANTITY"))
STRATEGIES = ("RAW", "MASK", "SUR-CORPUS", "SUR-DOC")


def ner_spans_from_entities(entities: list[tuple[int, int, str]]) -> list[tuple[int, int]]:
    """Keep the predeclared identifier-like labels, not every NER label."""
    spans = sorted((int(a), int(b)) for a, b, label in entities
                   if label in NER_LABELS and 0 <= a < b)
    if any(spans[i][0] < spans[i - 1][1] for i in range(1, len(spans))):
        raise ValueError("NER spans unexpectedly overlap")
    return spans


def validate_frozen_pairs(rows: list[dict], frozen: list[dict]) -> None:
    if [(r["pageid"], r["revids"]) for r in rows] != [
        (r["pageid"], r["revids"]) for r in frozen
    ]:
        raise ValueError("natural revision cohort or revision IDs changed")


def load_frozen_rows() -> tuple[list[dict], dict]:
    frozen = json.loads(FROZEN_RESULT.read_text())
    if frozen["status"] != "COMPLETE" or frozen["eligible_pairs"] != 105:
        raise ValueError("prior natural-revision result is not complete")
    members = json.loads((DATA / "category.json").read_text())["query"]["categorymembers"]
    source_rows, missing = natural_revision._all_rows(DATA, members)
    if missing or len(source_rows) != 250:
        raise ValueError("frozen page download is incomplete")
    by_id = {entry["pageid"]: entry for entry in source_rows}
    rows = []
    for old in frozen["pairs"]:
        page = by_id[old["pageid"]]["response"]["query"]["pages"][0]
        pair = natural_revision.select_pair(page)
        if pair is None or list(pair[:2]) != old["revids"]:
            raise ValueError("revision pair differs from frozen result")
        raw_j = natural_revision.jaccard(pair[2], pair[3])
        if abs(raw_j - old["raw_j"]) > 1e-12:
            raise ValueError("raw shingle score differs from frozen result")
        rows.append({"pageid": old["pageid"], "title": by_id[old["pageid"]]["title"],
                     "revids": old["revids"], "texts": [pair[2], pair[3]]})
    validate_frozen_pairs(rows, frozen["pairs"])
    return rows, frozen


def bootstrap_difference(pair_hits: list[bool], baseline_hits: list[bool]) -> dict:
    delta = [int(a) - int(b) for a, b in zip(pair_hits, baseline_hits)]
    if len(delta) != 105:
        raise ValueError("expected 105 frozen page pairs")
    rng = random.Random(20261004)
    draws = sorted(sum(delta[rng.randrange(len(delta))] for _ in delta) / len(delta)
                   for _ in range(2000))
    return {"estimate": sum(delta) / len(delta), "ci95": [draws[49], draws[1949]],
            "resampling_unit": "page_id", "replicates": 2000}


def evaluate_policy(rows: list[dict], scope: str, spans: list[list[list[tuple[int, int]]]]) -> dict:
    if len(rows) != len(spans) or any(len(pair) != 2 for pair in spans):
        raise ValueError("one two-revision span list is required per page")
    densities = [[natural_revision.marking_density(text, marks)
                  for text, marks in zip(row["texts"], pair)]
                 for row, pair in zip(rows, spans)]
    metrics = {}
    raw_hits = None
    for strategy in STRATEGIES:
        sets = []
        for row, pair in zip(rows, spans):
            for text, marks, revision in zip(row["texts"], pair, row["revids"]):
                released = natural_revision.release(text, marks, strategy, revision)
                sets.append(natural_revision.shingles(released))
        jac = analysis.jaccard_matrix(sets)
        edges = np.argwhere(np.triu(jac >= 0.70, 1)).tolist()
        components = analysis.components(len(sets), edges)
        hits = [bool(jac[2 * i, 2 * i + 1] >= 0.70) for i in range(len(rows))]
        comp_hits = [components[2 * i] == components[2 * i + 1] for i in range(len(rows))]
        retained = {rows[i // 2]["pageid"] for i, root in enumerate(components) if i == root}
        scores = [float(jac[2 * i, 2 * i + 1]) for i in range(len(rows))]
        metrics[strategy] = {
            "pair_hits": sum(hits), "pair_recall": sum(hits) / len(rows),
            "component_hits": sum(comp_hits), "component_recall": sum(comp_hits) / len(rows),
            "distinct_page_ids_lost": len(rows) - len(retained),
            "median_pair_jaccard": statistics.median(scores),
            "paired_difference_vs_RAW": None if strategy == "RAW" else
                bootstrap_difference(hits, raw_hits),
        }
        if strategy == "RAW":
            raw_hits = hits
    token_densities = [sum(pair[i][0] for i in range(2)) / 2 for pair in densities]
    shingle_densities = [sum(pair[i][1] for i in range(2)) / 2 for pair in densities]
    return {
        "scope": scope,
        "eligible_pairs": len(rows),
        "median_marked_token_share": statistics.median(token_densities),
        "median_touched_shingle_share": statistics.median(shingle_densities),
        "touched_shingle_share_range": [min(shingle_densities), max(shingle_densities)],
        "pages_with_any_marking": sum(any(pair[i][0] > 0 for i in range(2)) for pair in densities),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--use-spacy-sm", action="store_true")
    args = parser.parse_args()
    output = run_order._external_path(args.output)
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError("refusing to overwrite prior result or partial")
    rows, old = load_frozen_rows()
    print("Verified frozen 105 Wikipedia revision pairs", flush=True)
    arms = {}
    for scope in ("TITLE_ONLY", "TITLE_NAME", "DENSE"):
        spans = [[policy_spans(text, row["title"], scope) for text in row["texts"]]
                 for row in rows]
        arms[scope] = evaluate_policy(rows, scope, spans)
        print(f"Evaluated {scope}", flush=True)
    model = None
    if args.use_spacy_sm:
        import spacy
        nlp = spacy.load("en_core_web_sm", exclude=["parser", "tagger", "lemmatizer", "attribute_ruler"])
        texts = [text for row in rows for text in row["texts"]]
        docs = list(nlp.pipe(texts, batch_size=4, n_process=1))
        spans = [[ner_spans_from_entities([(ent.start_char, ent.end_char, ent.label_)
                                           for ent in docs[2 * i + j].ents]) for j in range(2)]
                 for i in range(len(rows))]
        arms["SPACY_SM"] = evaluate_policy(rows, "SPACY_SM", spans)
        meta = Path(nlp.path) / "meta.json"
        wheel = run_order.PROJECT / "models" / "en_core_web_sm-3.8.0-py3-none-any.whl"
        wheel_sha256 = run_order._sha256(wheel)
        if wheel_sha256 != "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85":
            raise ValueError("small NER model wheel differs from publisher metadata")
        model = {"name": "en_core_web_sm", "version": "3.8.0", "license": "MIT",
                 "spacy_version": spacy.__version__, "model_metadata_sha256": run_order._sha256(meta),
                 "model_metadata_path": str(meta), "model_wheel_sha256": wheel_sha256,
                 "model_wheel_path": str(wheel)}
        print("Evaluated SPACY_SM", flush=True)
    dense = arms["DENSE"]["metrics"]
    for strategy in STRATEGIES:
        earlier = old["metrics"][strategy]
        current = dense[strategy]
        if (current["pair_recall"], current["component_recall"],
            current["distinct_page_ids_lost"]) != (
            earlier["pair_recall"], earlier["component_recall"],
            earlier["distinct_page_ids_lost"]
        ):
            raise AssertionError(f"dense arm fails frozen-result reproduction: {strategy}")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if peak > run_order.MAX_RSS_BYTES:
        raise MemoryError(f"peak RSS {peak} exceeds 6 GiB")
    report = {
        "analysis": "natural_revision_marking_scope_sensitivity",
        "status": "exploratory_follow_up_on_seen_revision_cohort",
        "cohort_page_ids": [row["pageid"] for row in rows],
        "arms": arms,
        "provenance": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_result_sha256": run_order._sha256(FROZEN_RESULT),
            "input_files": [{"path": str(p), "sha256": run_order._sha256(p)} for p in
                            [DATA / "category.json", DATA / "revisions.jsonl",
                             DATA / "revisions-retry-1.jsonl", DATA / "revisions-retry-2.jsonl"]],
            "code": [{"path": str(p), "sha256": run_order._sha256(p)} for p in
                     [Path(__file__), Path(__file__).with_name("followup_science.py"),
                      Path(__file__).with_name("natural_revision.py"), PROTOCOL]],
            "model": model,
            "peak_rss_bytes": peak,
        },
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
