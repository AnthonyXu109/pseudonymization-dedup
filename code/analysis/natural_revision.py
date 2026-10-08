"""Natural Wikipedia revision check; protocol: NATURAL_REVISION_PROTOCOL_20261004.md.

Only public revision source text is fetched. The output report contains no text.
"""

import argparse
import hashlib
import hmac
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://en.wikipedia.org/w/api.php"
CUTOFF = "2026-01-01T00:00:00Z"
WORD = re.compile(r"\w+", re.UNICODE)
NAME = re.compile(r"(?<!\w)[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}(?!\w)")
YEAR = re.compile(r"(?<!\d)(?:19\d\d|20(?:0\d|1\d|2[0-6]))(?!\d)")
KEY = b"xcur-natural-revision-public-experimental-key-20261004"
MAX_RESPONSE = 2 * 1024 * 1024
MAX_TOTAL = 100 * 1024 * 1024


def words(text):
    return [m.group(0).lower() for m in WORD.finditer(text)]


def shingles(text):
    w = words(text)
    return {tuple(w[i:i + 5]) for i in range(max(0, len(w) - 4))}


def score_sets(a, b):
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def jaccard(a, b):
    return score_sets(shingles(a), shingles(b))


def eligible(a, b):
    if not (500 <= len(words(a)) <= 30000 and 500 <= len(words(b)) <= 30000):
        return False
    return 0.70 <= jaccard(a, b) < 0.99


def mark_spans(text, title):
    title = re.sub(r"\s*\([^)]*\)$", "", title).strip()
    proposed = []
    if title:
        pat = re.compile(r"(?<!\w)" + re.escape(title).replace(r"\ ", r"\s+") + r"(?!\w)", re.I)
        proposed.extend(m.span() for m in pat.finditer(text))
    proposed.extend(m.span() for m in NAME.finditer(text))
    proposed.extend(m.span() for m in YEAR.finditer(text))
    chosen = []
    for a, b in sorted(proposed, key=lambda x: (-(x[1] - x[0]), x[0])):
        if not any(a < d and c < b for c, d in chosen):
            chosen.append((a, b))
    return sorted(chosen)


def marking_density(text, spans):
    tokens = list(WORD.finditer(text))
    flags = [any(a < m.end() and m.start() < b for a, b in spans) for m in tokens]
    marked_token_share = sum(flags) / len(flags) if flags else 0.0
    windows = [any(flags[i:i + 5]) for i in range(max(0, len(flags) - 4))]
    touching_shingle_share = sum(windows) / len(windows) if windows else 0.0
    return marked_token_share, touching_shingle_share


def _token(token, scope):
    dig = hmac.new(KEY + b"|" + str(scope).encode(), token.lower().encode(), hashlib.sha256).digest()
    n = int.from_bytes(dig[:8], "big")
    out = []
    for _ in range(10):
        n, r = divmod(n, 26)
        out.append(chr(97 + r))
    return "zq" + "".join(out)


def release(text, spans, strategy, revision_id):
    if strategy == "RAW":
        return text
    chunks, end = [], 0
    for a, b in spans:
        chunks.append(text[end:a])
        if strategy == "MASK":
            chunks.append("[PII]")
        elif strategy in ("SUR-CORPUS", "SUR-DOC"):
            scope = "CORPUS" if strategy == "SUR-CORPUS" else revision_id
            chunks.append(WORD.sub(lambda m: _token(m.group(0), scope), text[a:b]))
        else:
            raise ValueError(strategy)
        end = b
    chunks.append(text[end:])
    return "".join(chunks)


def select_pair(page):
    valid = []
    for r in page.get("revisions", []):
        main = r.get("slots", {}).get("main", {})
        if r.get("timestamp", "") >= CUTOFF or main.get("contentmodel") != "wikitext":
            continue
        content = main.get("content")
        if isinstance(content, str):
            valid.append((r["timestamp"], int(r["revid"]), content))
    valid.sort(reverse=True)
    if len(valid) < 2:
        return None
    return valid[0][1], valid[1][1], valid[0][2], valid[1][2]


def _api(params):
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "XCurResearch/1.0 (local academic reproducibility audit)"})
    with urllib.request.urlopen(req, timeout=30) as response:
        raw = response.read(MAX_RESPONSE + 1)
    if len(raw) > MAX_RESPONSE:
        raise ValueError("API response exceeds 2 MiB bound")
    return raw, json.loads(raw)


def fetch(data_dir):
    data_dir = Path(data_dir)
    if data_dir.exists() and any(data_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite nonempty data directory")
    data_dir.mkdir(parents=True, exist_ok=True)
    raw, obj = _api({"action": "query", "list": "categorymembers",
                     "cmtitle": "Category:21st-century American women writers",
                     "cmtype": "page", "cmlimit": "250", "format": "json", "formatversion": "2"})
    members = obj.get("query", {}).get("categorymembers", [])
    if len(members) != 250 or any(m.get("ns") != 0 for m in members):
        raise ValueError("Category response does not contain 250 article pages")
    (data_dir / "category.json").write_bytes(raw)
    total = len(raw)
    failures = []
    with (data_dir / "revisions.jsonl").open("xb") as out:
        for i, member in enumerate(members):
            time.sleep(1.0)
            page_id = member["pageid"]
            try:
                body, result = _api({"action": "query", "prop": "revisions", "pageids": str(page_id),
                                     "rvprop": "ids|timestamp|contentmodel|content", "rvslots": "main",
                                     "rvstart": CUTOFF, "rvdir": "older", "rvlimit": "2",
                                     "format": "json", "formatversion": "2"})
                total += len(body)
                if total > MAX_TOTAL:
                    raise ValueError("100 MiB raw-download cap reached")
                row = {"pageid": page_id, "title": member["title"],
                       "response_sha256": hashlib.sha256(body).hexdigest(), "response": result}
                out.write((json.dumps(row, ensure_ascii=False) + "\n").encode())
            except ValueError:
                raise
            except Exception as exc:
                failures.append({"pageid": page_id, "error": type(exc).__name__ + ": " + str(exc)})
            if (i + 1) % 25 == 0:
                print(f"fetched {i+1}/250, failures {len(failures)}, bytes {total}", flush=True)
    (data_dir / "fetch_manifest.json").write_text(json.dumps({
        "category_sha256": hashlib.sha256(raw).hexdigest(),
        "revisions_sha256": hashlib.sha256((data_dir / "revisions.jsonl").read_bytes()).hexdigest(),
        "pages_requested": 250, "failures": failures, "total_response_bytes": total,
        "source": API, "license": "CC BY-SA 4.0 / GFDL; see Wikimedia Terms",
    }, indent=2))
    print(f"FETCH_DONE failures={len(failures)} total_bytes={total}")


def _read_jsonl(path):
    if not path.exists():
        return []
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def merge_rows(members, original, retries):
    """Merge transport retries without changing the frozen category cohort."""
    member_ids = {m["pageid"] for m in members}
    by_id = {}
    for row in [*original, *retries]:
        page_id = row["pageid"]
        if page_id not in member_ids or page_id in by_id:
            raise ValueError(f"Unexpected or duplicate page ID: {page_id}")
        pages = row.get("response", {}).get("query", {}).get("pages", [])
        if len(pages) != 1 or pages[0].get("pageid") != page_id:
            raise ValueError(f"Response/page ID mismatch: {page_id}")
        by_id[page_id] = row
    return ([by_id[m["pageid"]] for m in members if m["pageid"] in by_id],
            [m["pageid"] for m in members if m["pageid"] not in by_id])


def _all_rows(data_dir, members):
    original = _read_jsonl(data_dir / "revisions.jsonl")
    retry_files = sorted(data_dir.glob("revisions-retry-*.jsonl"))
    retries = [row for path in retry_files for row in _read_jsonl(path)]
    return merge_rows(members, original, retries)


def retry_failed(data_dir):
    """Retry only 429-failed IDs after cooldown; preserve every attempt."""
    data_dir = Path(data_dir)
    members = json.loads((data_dir / "category.json").read_bytes())["query"]["categorymembers"]
    original_manifest = json.loads((data_dir / "fetch_manifest.json").read_text())
    if original_manifest["pages_requested"] != 250:
        raise ValueError("Original frozen cohort size changed")
    rows, pending = _all_rows(data_dir, members)
    allowed = {f["pageid"] for f in original_manifest["failures"]}
    if not set(pending) <= allowed:
        raise ValueError("Retry would request outside original failed ID set")
    if not pending:
        print("RETRY_NOT_NEEDED")
        return
    attempts = len(list(data_dir.glob("revisions-retry-*.jsonl"))) + 1
    out_path = data_dir / f"revisions-retry-{attempts}.jsonl"
    manifest_path = data_dir / f"retry-manifest-{attempts}.json"
    if out_path.exists() or manifest_path.exists():
        raise FileExistsError("Retry attempt path already exists")
    prior_bytes = original_manifest["total_response_bytes"]
    for path in data_dir.glob("retry-manifest-*.json"):
        prior_bytes += json.loads(path.read_text())["response_bytes"]
    titles = {m["pageid"]: m["title"] for m in members}
    failures, requested, response_bytes, back_to_back_429 = [], [], 0, 0
    with out_path.open("xb") as out:
        for page_id in pending:
            time.sleep(5.0)
            requested.append(page_id)
            try:
                body, result = _api({"action": "query", "prop": "revisions", "pageids": str(page_id),
                                     "rvprop": "ids|timestamp|contentmodel|content", "rvslots": "main",
                                     "rvstart": CUTOFF, "rvdir": "older", "rvlimit": "2",
                                     "format": "json", "formatversion": "2"})
                if prior_bytes + response_bytes + len(body) > MAX_TOTAL:
                    raise ValueError("100 MiB raw-download cap reached")
                pages = result.get("query", {}).get("pages", [])
                if len(pages) != 1 or pages[0].get("pageid") != page_id:
                    raise ValueError("API response page ID mismatch")
                response_bytes += len(body)
                row = {"pageid": page_id, "title": titles[page_id],
                       "response_sha256": hashlib.sha256(body).hexdigest(), "response": result}
                out.write((json.dumps(row, ensure_ascii=False) + "\n").encode())
                back_to_back_429 = 0
            except urllib.error.HTTPError as exc:
                failures.append({"pageid": page_id, "error": f"HTTP {exc.code}"})
                if exc.code == 429:
                    back_to_back_429 += 1
                    if back_to_back_429 >= 3:
                        print("RETRY_STOPPED after three consecutive HTTP 429 responses", flush=True)
                        break
                    delay = min(60, 15 * (2 ** (back_to_back_429 - 1)))
                    print(f"HTTP 429; cooling down {delay}s", flush=True)
                    time.sleep(delay)
            except ValueError:
                raise
            except Exception as exc:
                failures.append({"pageid": page_id, "error": type(exc).__name__ + ": " + str(exc)})
            if len(requested) % 25 == 0:
                print(f"retry {len(requested)}/{len(pending)}, failures {len(failures)}", flush=True)
    manifest_path.write_text(json.dumps({
        "attempt": attempts, "original_manifest_sha256": hashlib.sha256(
            (data_dir / "fetch_manifest.json").read_bytes()).hexdigest(),
        "original_revisions_sha256": original_manifest["revisions_sha256"],
        "pending_at_start": pending, "requested": requested, "failures": failures,
        "not_attempted": pending[len(requested):], "response_bytes": response_bytes,
        "revisions_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }, indent=2))
    print(f"RETRY_DONE attempted={len(requested)} failures={len(failures)}")


def _components(sets):
    parent = list(range(len(sets)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(sets)):
        for j in range(i):
            if score_sets(sets[i], sets[j]) >= 0.70:
                a, b = root(i), root(j)
                parent[max(a, b)] = min(a, b)
    return [root(i) for i in range(len(sets))]


def evaluate(data_dir, output):
    data_dir, output = Path(data_dir), Path(output)
    if output.exists():
        raise FileExistsError(output)
    categories = json.loads((data_dir / "category.json").read_bytes())["query"]["categorymembers"]
    by_id = {m["pageid"]: m for m in categories}
    raw_rows, missing = _all_rows(data_dir, categories)
    rows = []
    reasons = {}
    for entry in raw_rows:
        pages = entry["response"].get("query", {}).get("pages", [])
        page = pages[0] if pages else {}
        pair = select_pair(page)
        reason = None
        if pair is None:
            reason = "fewer_than_two_public_wikitext_revisions"
        elif not (500 <= len(words(pair[2])) <= 30000 and 500 <= len(words(pair[3])) <= 30000):
            reason = "length_outside_range"
        else:
            raw_j = jaccard(pair[2], pair[3])
            if not 0.70 <= raw_j < 0.99:
                reason = "raw_similarity_outside_range"
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        assert entry["pageid"] in by_id
        r1, r2, t1, t2 = pair
        s1, s2 = mark_spans(t1, entry["title"]), mark_spans(t2, entry["title"])
        rows.append({"pageid": entry["pageid"], "title": entry["title"],
                     "revids": [r1, r2], "texts": [t1, t2], "spans": [s1, s2],
                     "raw_j": raw_j})
    if missing or len(rows) < 30:
        status = "INCOMPLETE_FETCH" if missing else "BELOW_MINIMUM"
        result = {"status": status, "eligible_pairs": len(rows), "exclusions": reasons,
                  "pages_fetched": len(raw_rows), "missing_page_ids": missing}
    else:
        strategies = ["RAW", "MASK", "SUR-CORPUS", "SUR-DOC"]
        metrics = {}
        for strategy in strategies:
            sets = []
            hits = []
            for row in rows:
                a = release(row["texts"][0], row["spans"][0], strategy, row["revids"][0])
                b = release(row["texts"][1], row["spans"][1], strategy, row["revids"][1])
                sa, sb = shingles(a), shingles(b)
                sets.extend((sa, sb))
                hits.append(score_sets(sa, sb) >= 0.70)
            comp = _components(sets)
            comp_hits = [comp[2*i] == comp[2*i+1] for i in range(len(rows))]
            # A distinct page ID is lost if no representative from its page
            # survives the component; representatives are lowest doc indexes.
            reps = {c: min(i for i, cc in enumerate(comp) if cc == c) for c in set(comp)}
            retained_ids = {rows[i//2]["pageid"] for i in reps.values()}
            lost = len(rows) - len(retained_ids)
            metrics[strategy] = {"pair_recall": sum(hits)/len(rows),
                                 "component_recall": sum(comp_hits)/len(rows),
                                 "distinct_page_ids_lost": lost,
                                 "pair_hits": hits}
        rng = random.Random(20261004)
        diffs = {}
        for strategy in strategies[1:]:
            delta = [int(metrics[strategy]["pair_hits"][i])-int(metrics["RAW"]["pair_hits"][i])
                     for i in range(len(rows))]
            draws = [sum(delta[rng.randrange(len(delta))] for _ in delta)/len(delta)
                     for _ in range(2000)]
            draws.sort()
            diffs[strategy] = {"points": sum(delta)/len(delta),
                               "ci95": [draws[49], draws[1949]]}
        result = {"status": "COMPLETE", "eligible_pairs": len(rows),
                  "exclusions": reasons, "metrics": metrics, "paired_differences": diffs,
                  "pages_fetched": len(raw_rows), "missing_page_ids": [],
                  "raw_jaccard": {"min": min(r["raw_j"] for r in rows),
                                   "median": sorted(r["raw_j"] for r in rows)[len(rows)//2],
                                   "max": max(r["raw_j"] for r in rows)},
                  "pairs": [{"pageid": r["pageid"], "revids": r["revids"], "raw_j": r["raw_j"],
                             "marking_density": [marking_density(t, sp)
                                                 for t, sp in zip(r["texts"], r["spans"])]}
                            for r in rows]}
        for item in metrics.values():
            del item["pair_hits"]
    result["category_sha256"] = hashlib.sha256((data_dir / "category.json").read_bytes()).hexdigest()
    result["revisions_sha256"] = hashlib.sha256((data_dir / "revisions.jsonl").read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as out:
        json.dump(result, out, indent=2)
    print(json.dumps({k: result[k] for k in ("status", "eligible_pairs", "exclusions")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["fetch", "retry", "evaluate"])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.mode == "fetch":
        fetch(args.data_dir)
    elif args.mode == "retry":
        retry_failed(args.data_dir)
    else:
        if not args.output:
            parser.error("evaluate requires --output")
        evaluate(args.data_dir, args.output)
