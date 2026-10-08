"""E2: pipeline orderings = (strategy used for dedup decisions, strategy used for release)."""
import json, sys
PIPES = [
    ('sanitize(MASK) -> dedup', 'MASK', 'MASK'),
    ('sanitize(TYPE) -> dedup', 'TYPE', 'TYPE'),
    ('sanitize(ALIAS-DOC) -> dedup', 'ALIAS-DOC', 'ALIAS-DOC'),
    ('sanitize(SUR-DOC) -> dedup', 'SUR-DOC', 'SUR-DOC'),
    ('sanitize(SUR-CORPUS) -> dedup', 'SUR-CORPUS', 'SUR-CORPUS'),
    ('dedup(RAW) -> release SUR-DOC', 'RAW', 'SUR-DOC'),
    ('dedup(keyed SUR-CORPUS, internal) -> release SUR-DOC', 'SUR-CORPUS', 'SUR-DOC'),
]
def table(res, th='0.7'):
    rows = []
    for name, ded, rel in PIPES:
        r, l = res[ded][th], res[rel]['link']
        rows.append(dict(pipeline=name, cases_lost=r['cases_lost'], residual_dups=r['residual_dups'],
                         dup_recall=r['dup_recall'], link_precision=l['precision'], link_recall=l['recall']))
    return rows
out = {}
for tag, f, key in [('gold', 'results/e1_first.json', 'strategies'), ('detector', 'results/e3_tab.json', 'results')]:
    res = json.load(open(f))[key]
    out[tag] = table(res)
    print('==', tag)
    for r in out[tag]:
        print(f"{r['pipeline']:55s} lost={r['cases_lost']:3d} resid={r['residual_dups']:3d} rec={r['dup_recall']:.3f} linkP={r['link_precision']:.3f} linkR={r['link_recall']:.4f}")
json.dump(out, open('results/e2.json', 'w'), indent=1)
