# Natural-revision validation protocol (frozen before content access)

Date: 2026-10-04. This is a new, independent source cohort for a bounded
external-validity check. The current eight-page PDF (SHA-256
`ccbd5c2f3d9dfadbf197d6676025fd1951e7d8c62287b70ef98c002e93f18c6a`)
is unchanged. Earlier TAB/Enron/ECtHR/C4 outcomes are exploratory context,
not selection data for this cohort.

## Input and rights

- Source: English Wikipedia, category
  `Category:21st-century American women writers`, via the official MediaWiki
  categorymembers and revisions APIs. Wikimedia's terms identify text
  contributions as CC BY-SA 4.0 / GFDL (with exceptions for imported text);
  no article text or revision diff will be redistributed from this study.
  Save only local raw API responses under external `data/20261004/`;
  eventual public artifacts, if approved, are revision IDs, source URLs,
  code, hashes and aggregates.
- Take the first 250 namespace-0 category members in API sort order as
  observed on 2026-10-04 and save the raw category response/continuation
  chain before fetching text. Do not replace failed page requests with
  handpicked pages; record failures.
- For each page, fetch the latest two publicly visible revisions whose
  timestamps precede 2026-01-01 00:00 UTC. A positive pair consists of those
  consecutive available revisions of the *same page ID*. Distinct page IDs
  are negative for the retention analysis. Exclude pages with fewer than two
  revisions, missing revision content, non-wikitext content, or either text
  outside 500–30,000 word tokens. Do not choose pages based on treatment
  results.
- The primary near-duplicate stratum is pairs with *raw* word-5-gram Jaccard
  in [0.70, 0.99). This is a predeclared eligibility condition for a natural
  near-duplicate question, not outcome tuning. Report the full eligible
  cohort count and the excluded-pair flow. Do not increase the sample or
  change this interval after seeing results.

## Fixed representation and outcomes

- Operate on revision wikitext, documenting this as source-text rather than
  rendered-article evaluation. Use the existing paper's lowercase word
  tokenizer and exact sets of 5-grams at threshold 0.70, not LSH. Use full
  strings for set identity, avoiding the old implementation's CRC32 shortcut.
- Mark as a deterministic *stress* identifier surface every literal match
  of the full article title after removing a terminal parenthetical
  disambiguator (case-insensitive, whole-token phrase), any
  2–3-token capitalized personal-name-shaped phrase, and standalone
  four-digit years in [1900, 2026]. Resolve overlaps by longest span then
  earliest start. This rule is intentionally not claimed to be a validated
  PII detector or complete anonymizer. Apply independently to each revision.
- Compare RAW, one shared `[PII]` token per marked span (MASK), tokenwise
  corpus-scoped keyed pseudonyms (SUR-CORPUS), and tokenwise
  revision-scoped keyed pseudonyms (SUR-DOC). The fixed public experimental
  HMAC key is a reproducibility seed, not a secrecy claim. Use a separate
  revision ID as the SUR-DOC scope. Marked spans may differ between
  revisions; do not force-align them.
- Primary outcome: pair recall at Jaccard >= 0.70 among eligible natural
  positive pairs for each strategy, and paired differences versus RAW with
  95% page-cluster bootstrap intervals (2,000 resamples, seed 20261004).
  Because one pair is sampled per page, the page is the resampling unit.
- Secondary: exact-Jaccard connected-component recall on the 2-revision
  documents plus distinct page IDs lost under deterministic representative
  selection; this is *record loss*, not policy harm. Also report marked-token
  and identifier-touching-shingle fractions and raw similarity distribution.
  Pair-level table and aggregate only. No privacy-risk claim is tested here.
- Minimum interpretable sample is 30 eligible page pairs. If fewer, report
  the failed feasibility result without relaxing protocol. If API, license,
  storage or RAM constraints fail, stop and preserve the manifest.

## Resource and provenance guardrails

Single process, no GPU, no training, no paid API, RAM <= 6 GiB. Rate-limit
requests to <= 1/s; bound each API response to 2 MiB and the raw download
set to 100 MiB. Use external project `data`, `cache`, `runs`, and `artifacts`
directories only. Store SHA-256 for each response and final code, with
failures and exclusions. Do not modify the current PDF or prior result files.
