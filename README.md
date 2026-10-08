# When Pseudonymization Breaks Text Deduplication: code and results

Code, protocols and numeric results for the paper *When Pseudonymization Breaks Text Deduplication*
(Extreme Data Curation workshop, IEEE BigData 2026). `SUPPLEMENT.md` holds the tables that the paper
refers to but does not print, including the full cutoff grid of Section VI.

## Layout

| Path | Contents |
|---|---|
| `SUPPLEMENT.md` | Supplementary tables (cutoff grid, natural-revision scopes, screening per setting, mitigations, embeddings, surrogate exposure, audit agreement) |
| `code/base/` | Corpus construction, identifier detection, release strategies, exact and MinHash deduplication, Fig. 1 and Fig. 2 |
| `code/analysis/` | Fixed-release decision comparison (Table V first blocks), marking omission, natural revisions, cutoff grid, shingle-collision check, unit tests |
| `code/followups/` | Follow-up experiments v8 to v12 (shared keys and collapsed pseudonyms, scope ablation, TAB-LexGLUE pairs, embeddings, surrogate exposure, mitigations, re-detection, keyed raw shingles) |
| `protocols/` | Pre-run protocols for each experiment, written before the runs |
| `results/main/` | Numeric results behind Figs. 1-2 and Tables II, III and V |
| `results/followups/` | Numeric results of the follow-up experiments |

## Mapping to the paper

| Paper item | Result files | Code |
|---|---|---|
| Fig. 1, prediction and screening rule | `results/main/dose_*.json`, `results/followups/e10_screen.json` | `code/base/dose*.py`, `code/followups/fig_dose3.py`, `code/followups/exp_v11.py` (e10) |
| Table II, pair recall | `results/main/e1_gold_first_plain.json`, `f2_real.json`, `e4_enron.json`, `extra_cells.json`, `scale_*.json` | `code/base/e1v2.py`, `f2_real.py`, `e4v2_enron.py`, `extra_cells.py`, `scale_eval.py` |
| Fig. 2, pipeline recall vs. records removed | `results/main/curves_exact.json`, `pipeline_*_plain.json` | `code/base/curves.py`, `pipeline.py`, `code/followups/fig_curves2.py` |
| Scrubbing scope (Dolma-like, entity types) | `results/main/pipeline_*_narrow_v5.json`, `results/followups/e3_scope.json` | `code/base/pipeline.py`, `code/followups/exp_v8.py` |
| Natural duplicates | `results/main/natural-*.json`, `results/followups/e4_natural.json`, `e11_containment.json` | `code/analysis/natural_revision.py`, `followup_natural_scope.py`, `code/followups/exp_v9.py`, `exp_v11.py` |
| Embedding similarity | `results/followups/e5_embed.json`, `e5b_embed.json` | `code/followups/exp_v9.py` |
| Table III, MASK changes | `results/main/cboot_exact.json`, `pipeline_*` | `code/base/cboot_exact.py`, `pipeline.py` |
| Table V and Section VI | `results/main/order-*.json`, `sensitivity-tab-gold-v1.json`, `fixed-release-thresholds-v1.json`, `results/followups/e1_e2_tab.json`, `e7_tab.json`, `e8_pairs.json`, `e9_mismatch.json`, `e12_*.json` | `code/analysis/run_order*.py`, `sensitivity.py`, `followup_thresholds.py`, `code/followups/exp_v8.py`, `exp_v11.py`, `exp_v12.py` |
| Surrogate exposure | `results/followups/e6_diff.json`, `e14_fakerdict.json` | `code/followups/exp_v10.py`, `exp_v12.py` |

## Data

The source corpora are not redistributed. Obtain them from their publishers under their own terms
(see `DATA_LICENSES.md`). The retention-audit labels, the review interface and any text excerpts are
not included; `SUPPLEMENT.md` reports the label counts and the agreement between the two reviewers.
Pseudonyms in the experiments are keyed with a fixed experimental secret (`SALT` in `code/base/common.py`).
It protects nothing and is published so that the results can be reproduced.

## Running

Python 3.11 or later with NumPy, SciPy, pandas, spaCy 3.8 (`en_core_web_lg`, `en_core_web_sm`, and
`en_core_web_trf` for the embedding test), and Faker. MinHash and LSH are implemented in `code/base` (`fastlsh.py`, `lsh2v2.py`).
Scripts expect the corpora under a project directory given by `XCUR_PROJECT` (default: the current
directory) and write to `results/`. Some older scripts still contain defaults from the original
machine; adjust the paths at the top of each script. `requirements-tab.txt` pins the dependencies of
the lightweight TAB rerun (`code/analysis/run_order.py`).

Unit tests:

```sh
PYTHONPATH=code/base:code/analysis python3 -m unittest discover -s code/analysis -p 'test_*.py'
PYTHONPATH=code/base python3 code/base/tests_prop/test_proposition.py
```

## License

Code: MIT (see `LICENSE`). Result files: CC BY 4.0. Source corpora keep their own licenses.
