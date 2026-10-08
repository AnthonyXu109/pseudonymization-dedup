# Supplementary tables

Numbers come from the files in `results/`. TAB settings use 1,268 original judgments and 400 paragraph copies; exact Jaccard on word 5-grams; connected components.

## S1. Fixed-release decisions at five cutoffs (Section VI)

Copies detected / distinct cases lost, out of 400 copies (gold spans; survivors released as SUR-DOC). Source: `results/main/fixed-release-thresholds-v1.json`.

| Jaccard cutoff | RAW | SUR-CORPUS | MASK | TYPE | SUR-DOC |
|---|---:|---:|---:|---:|---:|
| 0.50 | 397/41 | 397/33 | 397/48 | 397/48 | 185/2 |
| 0.60 | 390/24 | 390/18 | 391/37 | 391/37 | 87/0 |
| 0.70 | 364/11 | 364/5 | 363/25 | 363/25 | 17/0 |
| 0.80 | 303/5 | 303/1 | 305/9 | 306/9 | 4/0 |
| 0.90 | 176/0 | 176/0 | 177/2 | 177/2 | 0/0 |

SUR-CORPUS minus RAW in lost cases [95% cluster-bootstrap interval], by cutoff: -8 [-14,-3], -6 [-12,-1], -6 [-14,-1], -4 [-11,0], 0 [0,0]. MASK and TYPE lose more cases than RAW at every cutoff; their intervals do not all exclude zero.

## S2. Natural Wikipedia revisions under four marking scopes

105 revision pairs. Source: `results/main/natural-marking-scope-v1.json`.

| Marking scope | Median touched-shingle share | SUR-DOC detected | RAW, MASK, SUR-CORPUS detected |
|---|---:|---:|---:|
| Article title only | 3.26% | 104/105 | 105/105 |
| Title plus name-shaped phrases | 35.68% | 0/105 | 105/105 |
| Title, names and years | 49.26% | 0/105 | 105/105 |
| Small spaCy NER | 50.65% | 0/105 | 105/105 |

## S3. Full strings versus 32-bit shingle IDs

Across the 400 TAB original-copy pairs, the full-string and 32-bit Jaccard values agree exactly and no decision changes at any of the five cutoffs (`results/main/shingle-collision-v1.json`).

## S4. Screening rule m_doc > 0.176 per setting

Pairs that reach 0.7 without scrubbing; drops = pairs that SUR-DOC moves below 0.7. Source: `results/followups/e10_screen.json`.

| Setting | Pairs >= 0.7 raw | Drops | Flagged | Missed | False alarms | Median m_doc |
|---|---:|---:|---:|---:|---:|---:|
| ECtHR second-source | 974 | 968 | 959 | 9 | 0 | 0.37 |
| Enron Dolma-like | 619 | 21 | 8 | 13 | 1 | 0.00 |
| Enron detector | 619 | 561 | 522 | 39 | 1 | 0.35 |
| Enron natural | 1339 | 1159 | 1159 | 0 | 0 | 0.37 |
| Enron second-source | 510 | 495 | 434 | 61 | 0 | 0.35 |
| TAB detector | 364 | 362 | 361 | 1 | 0 | 0.42 |
| TAB gold | 364 | 347 | 323 | 24 | 0 | 0.29 |
| TAB second-source | 394 | 394 | 391 | 3 | 0 | 0.41 |
| Web second-source | 1761 | 1416 | 1177 | 239 | 0 | 0.26 |

The Dolma-like scope uses e-mail, phone and IP patterns; on ECtHR the phone pattern mostly matches year and page ranges.

## S5. Mitigations on TAB (Table V setting)

Copies detected / distinct cases lost / residual copies; difference in lost cases against RAW with 5-grams [95% interval]. Sources: `e7_tab.json`, `e9_mismatch.json`, `e12_e13_tab.json`.

| Decision text | Detected | Lost | Residual | Delta lost |
|---|---:|---:|---:|---|
| RAW n5 | 364/400 | 11 | 36 | reference |
| RAW n3 | 373/400 | 20 | 27 | +9 [3, 17] |
| RAW n2 | 379/400 | 34 | 21 | +23 [10, 39] |
| SUR-DOC n5 | 17/400 | 0 | 383 | -11 [-27, -1] |
| SUR-DOC n3 | 57/400 | 0 | 343 | -11 [-27, -1] |
| SUR-DOC n2 | 95/400 | 1 | 305 | -10 [-27, 0] |
| FAKER-DOC n5 | 19/400 | 0 | 381 | -11 [-27, -1] |
| FAKER-DOC n3 | 67/400 | 0 | 333 | -11 [-27, -1] |
| FAKER-DOC n2 | 123/400 | 1 | 277 | -10 [-27, 0] |
| SUR-DOC-SKIP | 361/400 | 24 | 39 | +13 [5, 23] |
| HASH-SKIP | 361/400 | 24 | 39 | +13 [5, 23] |
| YEAR | 165/400 | 0 | 235 | -11 [-27, -1] |
| YEAR-FAKER | 188/400 | 0 | 212 | -11 [-27, -1] |
| ALIAS-DOC | 237/400 | 13 | 163 | +2 [-4, 9] |
| DROP | 362/400 | 24 | 38 | +13 [5, 23] |
| KEYED-RAW | 364/400 | 11 | 36 | +0 [0, 0] |
| FAKER-DOC redetect-collapse | 290/400 | 8 | 109 | -3 [-11, 3] |
| SUR-DOC redetect-collapse | 29/400 | 0 | 371 | -11 [-27, -1] |
| ALIAS-DOC collapse | 363/400 | 25 | 37 | +14 [5, 25] |
| detector lg/lg SUR-DOC-COLLAPSED | 368/400 | 25 | 31 | +14 [6, 25] |
| detector lg/lg SUR-DOC-SKIP | 359/400 | 21 | 41 | +10 [3, 18] |
| detector lg/lg SUR-DOC | 2/400 | 0 | 398 | -11 [-27, -1] |
| detector lg/sm SUR-DOC-COLLAPSED | 297/400 | 23 | 100 | +12 [4, 21] |
| detector lg/sm SUR-DOC-SKIP | 278/400 | 19 | 120 | +8 [2, 15] |
| detector lg/sm SUR-DOC | 1/400 | 0 | 399 | -11 [-27, -1] |

## S6. Pair recall of mitigations on second-source copies

Exact Jaccard >= 0.7; ECtHR 1,000 pairs, web 2,000 pairs; detector spans. Sources: `e8_pairs.json`, `e12_pairs.json`.

| Variant | ECtHR | Web |
|---|---:|---:|
| RAW n5 | 97.4% | 88.0% |
| RAW n3 | 98.3% | 90.8% |
| RAW n2 | 98.9% | 91.3% |
| SUR-DOC n5 | 0.6% | 17.2% |
| SUR-DOC n3 | 4.8% | 29.0% |
| SUR-DOC n2 | 13.0% | 38.6% |
| HASH-ENTITY n5 | 2.7% | 22.7% |
| HASH-ENTITY n3 | 12.8% | 37.4% |
| HASH-ENTITY n2 | 23.6% | 51.2% |
| FAKER-DOC n5 | 0.5% | 16.0% |
| FAKER-DOC n3 | 5.2% | 26.3% |
| FAKER-DOC n2 | 14.4% | 34.1% |
| SUR-DOC-SKIP n5 | 97.1% | 92.2% |
| HASH-SKIP n5 | 97.0% | 92.2% |
| YEAR n5 | 13.3% | 24.9% |
| YEAR-FAKER n5 | 15.2% | 23.2% |
| FAKER-DOC plain | 0.5% | 16.0% |
| FAKER-DOC redetect-collapse | 52.4% | 45.7% |
| ALIAS-DOC plain | 24.6% | 54.2% |
| ALIAS-DOC collapse | 97.2% | 89.8% |

Random pairs of different originals with Jaccard >= 0.7: 0 of 20,000 at n = 5, 3 and 2 on both corpora.

## S7. Embedding similarity

Threshold: 99.9th percentile of cosine over pairs of different originals, set per strategy. RoBERTa = mean last layer of en_core_web_trf over the first 300 words; static = en_core_web_lg vectors. Sources: `e5_embed.json`, `e5b_embed.json`.

| Sample | Strategy | RoBERTa recall | Static recall |
|---|---|---:|---:|
| ecthr (n=300) | MASK | 96.7% | 95.3% |
| ecthr (n=300) | RAW | 98.7% | 98.3% |
| ecthr (n=300) | SUR-DOC | 22.3% | 98.7% |
| c4 (n=500) | MASK | 70.6% | 79.6% |
| c4 (n=500) | RAW | 93.8% | 88.8% |
| c4 (n=500) | SUR-DOC | 49.6% | 95.0% |
| natural (n=300) | MASK | 0.0% | 54.7% |
| natural (n=300) | RAW | 0.0% | 64.3% |
| natural (n=300) | SUR-DOC | 0.0% | 58.3% |
| ecthr (n=300) | FAKER-DOC | 96.0% | 88.3% |
| ecthr (n=300) | HASH-ENTITY | 71.3% | 98.7% |

## S8. Surrogate exposure

TAB, FAKER-DOC over detector spans, known original-copy pairs. Candidates: capitalized words in at most five released documents. A missed name is a distinct capitalized string inside a gold PERSON, ORG or LOC span that no detector span overlaps (846 such strings that also occur in the copy). Sources: `e6_diff.json`, `e14_fakerdict.json`.

| Rule | Flagged | Missed names | Surrogates | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| One copy | 2188 | 200 | 1488 | 9.1% | 23.6% |
| One copy, Faker vocabulary removed | 757 | 198 | 61 | 26.2% | 23.4% |
| Two copies | 628 | 197 | 7 | 31.4% | 23.3% |
| Two copies, Faker vocabulary removed | 624 | 195 | 6 | 31.2% | 23.0% |

## S9. Retention audit agreement

Two reviewers labeled the same 185 changed-record and survivor pairs independently, with the strategy hidden. First review: 112 distinct, 68 duplicate, 5 unclear. Second review: 111 distinct, 68 duplicate, 6 unclear. Agreement 182/185 (98.4%), Cohen's kappa 0.968 over the three labels (0.976 on the 179 pairs both reviewers decided). By corpus: e-mail 56/58, judgments (TAB) 24/24, judgments (ECtHR) 45/46, web 57/57. No row of the paper's audit table changes by more than one record under the second review. Labels and texts are not released.
