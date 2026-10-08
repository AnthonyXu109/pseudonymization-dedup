# Fixed-release placeholder decision addendum

Frozen before running the additional comparator on 2026-10-04. This is an
exploratory follow-up on the previously studied TAB corpus, not a new
independent test. It adds two key-free internal decision baselines to the
already published fixed-release comparison, without changing its original
result file or protocol.

## Question and fixed design

Can a curator decide duplicates on a simple placeholder form rather than
raw or corpus-keyed text while still publishing the same document-scoped
pseudonyms? Use the existing 1,268 TAB cases, first annotator's gold spans,
and exactly 400 paragraph copies from the pinned v7 build with seed
20261001. Verify the three TAB source hashes in `inputs_manifest.json`.
No new sampling or copy generation.

Keep the original lowercased word-5-gram/32-bit-shingle-ID representation,
exact Jaccard decision graph, threshold 0.70, deterministic representative
rule, retention policy (one distinct case retained), and fixed final
`SUR-DOC` release. Add only internal `MASK` and `TYPE` decisions to the
original `RAW`, `SUR-CORPUS`, `SUR-DOC` rows. `MASK` replaces each gold span
with `[PII]`; `TYPE` replaces it with the gold entity-type token. Both use
the same input spans as the other rows.

## Outcomes and interpretation

For each new row, report copy-to-original component recall, exact pair
recall, distinct cases with no surviving representative, retained-record
count, and released cross-case person-string equality using the existing
implementation. For each contrast to `RAW`, report paired differences in
copy recall and lost-case count with 95% union-component bootstrap intervals,
2,000 draws and seed 20261001. Do not tune threshold or choose a headline
representation after seeing outcomes. A row is not called better solely
because it has higher recall; assess case retention as well. The comparison
is on explored data, and the published text is fixed in form, not identical
as a set of retained records across rows.

Run in one process, no model/training/download, RAM below 6 GiB. Write the
new JSON once to the external project's `runs/20261004/`; keep source
code and this addendum in an independent analysis workspace. Record file
hashes and any failed attempt.
