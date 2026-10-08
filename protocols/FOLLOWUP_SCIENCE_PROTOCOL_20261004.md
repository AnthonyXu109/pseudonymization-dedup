# Bounded scientific follow-up, fixed before new results

Date: 2026-10-04. This is a follow-up on already inspected corpora, not an
independent held-out validation. The existing 8-page candidate and all prior
run files remain immutable. New code uses the analysis workspace; new run,
cache, model, and QA outputs stay in the matching external-project folders.
One process at a time, no GPU or training, peak RSS below 6 GiB.

## A. Fixed-release decision sensitivity

Rebuild the existing TAB first-annotator corpus of 1,268 judgments and its
same 400 paragraph copies, after verifying all three pinned TAB input hashes.
For internal RAW, MASK, TYPE, SUR-CORPUS, and SUR-DOC decisions, keep the
same lowercased word five-grams, 32-bit shingle IDs, connected components,
and fixed final SUR-DOC release form. Sweep exactly 0.50, 0.60, 0.70, 0.80,
and 0.90 Jaccard thresholds; do not choose a new headline threshold from
outcomes. Report per cell direct pair and component recall, distinct cases
with no retained representative, and retained-record count. Report paired
SUR-CORPUS/RAW, MASK/RAW, TYPE/RAW, and SUR-DOC/RAW differences at each
threshold with 2,000 union-component bootstrap resamples, seed 20261001.
The 0.70 cells must reproduce the previously published fixed-release table.
The public release *form* is fixed, but retained document sets may differ.

## B. Natural-revision marking scope

Use only the already downloaded and frozen 250 Wikipedia page IDs and the
same 105 eligible pre-2026 revision pairs, with no new page selection or
source download. Keep full-string word-five-gram Jaccard and threshold 0.70.
Compare the existing dense TITLE+NAME+YEAR rule to TITLE+NAME (no years) and
TITLE-only (the article's own person name). Additionally test spaCy
en_core_web_sm 3.8.0 entity labels PERSON, ORG, GPE, LOC, NORP, DATE, TIME,
and QUANTITY if its official MIT-licensed 12 MB model and CPU dependencies
can be installed under the external project without breaching the resource
limits. spaCy is a separate detector from the main paper's large model;
never label it the same policy. Do not alter the 105-pair cohort. Report
marked-token and touched-shingle shares, pair and component recall, and
distinct page IDs lost for each arm. Paired page bootstrap: 2,000 draws,
seed 20261004. Record the model/package versions and hashes. If model
installation fails, preserve the failure and complete the deterministic arms.
None of these arms establishes actual PII-recognition accuracy.

## C. Collision check

On the same TAB decision corpus and all five decision representations,
compare full textual five-gram shingle sets against the production 32-bit
CRC representation. First count distinct textual shingles and true CRC
collisions in this finite corpus. Then, for each of the 400 original-copy
pairs at thresholds 0.50, 0.60, 0.70, 0.80, and 0.90, count status changes
and maximum absolute Jaccard difference. If feasible within 6 GiB, compare
the complete TAB decision graph/component membership at 0.70 for RAW and
SUR-DOC. Do not describe 32-bit results as collision-free. Save only
aggregates and offending pair IDs (if any), not raw text.

## Interpretation and manuscript rule

All outcomes, including null or contrary ones, enter the internal ledger.
Only scientifically material findings enter the 8-page manuscript. Changes
to the candidate require a new PDF hash and fresh independent evidence,
content, readability, and release review; prior advisory Accept judgments
do not transfer to changed bytes. No upload, submission, push, or authorship
change is authorized.
