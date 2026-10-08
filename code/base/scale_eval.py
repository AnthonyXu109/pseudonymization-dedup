"""Evaluate scale corpora from cached detector spans. ECtHR: exact sparse Jaccard. C4: MinHash LSH (14x8) candidates
verified by exact Jaccard for removals; copy recall from exact Jaccard of each original/copy pair."""
import sys, json, pickle, collections, numpy as np
from common import *
from evalsparse import evaluate_sparse
from evalcore import linkage
from fastlsh import signatures, lsh_edges
from stats import compare
name = sys.argv[1]
import glob
from build_scale import build_scale
docs = build_scale(name)
spans = [x for f in sorted(glob.glob(f'results/det/{name}_*.pkl')) for x in pickle.load(open(f, 'rb'))]
assert len(spans) == len(docs), (len(spans), len(docs))
for d, sp in zip(docs, spans): d['spans'] = sp
ALL = STRATS + EXTRA_STRATS
def jac(a, b):
    if not isinstance(a, np.ndarray): return len(a & b) / len(a | b) if a or b else 0.0
    i = len(np.intersect1d(a, b, assume_unique=True)); u = len(a) + len(b) - i
    return i / u if u else 0.0

def eval_lsh(docs, strats, ths):
    """Removals: datatrove-style MinHash LSH (5-grams, 14x8), bucket members linked without verification.
    Recall: exact Jaccard of each original/copy pair at each threshold."""
    from fastlsh import lsh_star_edges
    case = [d['case'] for d in docs]; ncase = len(set(case))
    orig = {d['case']: i for i, d in enumerate(docs) if not d['is_copy']}
    copies = [i for i, d in enumerate(docs) if d['is_copy']]
    res, vec = {}, {}
    for s in strats:
        sets = [np.fromiter(x, dtype=np.uint32, count=len(x)) for x in (shingles(release(d['text'], d['spans'], s, d['key'], d['case'])) for d in docs)]
        comp = components(len(docs), lsh_star_edges(signatures(sets, seed=0)))
        kc = collections.Counter(case[i] for i, c in enumerate(comp) if c == i)
        lost = np.array([kc.get(c, 0) == 0 for c in sorted(set(case))])
        same = np.array([comp[orig[case[i]]] == comp[i] for i in copies])
        r = {}
        for th in ths:
            detJ = np.array([jac(sets[orig[case[i]]], sets[i]) >= th for i in copies])
            r[str(th)] = dict(cases_lost=int(lost.sum()), dup_recall=float(detJ.mean()), dup_recall_lsh=float(same.mean()))
            vec[(s, th)] = dict(lost=lost, det=detJ)
        r['link_person'] = linkage(docs, s, 'PERSON')
        res[s] = r
        del sets
        print(s, {k: v for k, v in r.items() if k != 'link_person'}, round(r['link_person']['precision'], 3), flush=True)
    return res, vec

out = {}
PLAN = [('real', 'plain', [x for x in ALL if x not in ('DROP', 'SUR-SUBJECT', 'TYPE')]),
        ('real', 'datatrove', ['RAW', 'MASK', 'SUR-DOC']),
        ('para', 'plain', ['RAW', 'MASK', 'SUR-DOC'])]
if name == 'c4':
    part = sys.argv[2] if len(sys.argv) > 2 else 'a'
    PLAN = [('real', 'plain', ['RAW', 'MASK', 'SUR-DOC', 'SUR-CORPUS'])] if part == 'a' else \
           [('real', 'datatrove', ['RAW', 'MASK']), ('real', 'plain', ['HASH-ENTITY', 'FAKER-DOC', 'ALIAS-DOC'])]
    tag = part
else:
    tag = ''
for kind, mode, strats in PLAN:
    sub = [d for d in docs if not d['is_copy'] or d.get('copy_kind') == kind]
    NORM['mode'] = mode
    print('==', name, kind, mode, flush=True)
    r, v = eval_lsh(sub, strats, (0.7, 0.75))
    out[f'{kind}|{mode}'] = dict(results=r, compare={f'{s}@{t}': compare(v, s, th=t) for s in r for t in (0.7, 0.75) if s != 'RAW' and ('RAW', t) in v})
    json.dump(out, open(f'results/v2/scale_{name}{tag}.json', 'w'), indent=1, default=float)
# dose-response on second-source copies
if name == 'c4' and tag != 'a': sys.exit(0)
NORM['mode'] = 'plain'
import dose as D
rows = D.pairs_for([d for d in docs if not d['is_copy'] or d.get('copy_kind') == 'real'], name + ' second-source')
m = np.array([r['m'] for r in rows]); js = np.array([r['J_surdoc'] for r in rows]); jp = np.array([r['J_pred'] for r in rows]); jr = np.array([r['J_raw'] for r in rows])
out['dose'] = dict(n=len(rows), median_m=float(np.median(m)), rec_raw=float(np.mean(jr >= 0.7)), rec_sur=float(np.mean(js >= 0.7)), rec_pred=float(np.mean(jp >= 0.7)), mae=float(np.mean(np.abs(js - jp))))
print('dose', out['dose'], flush=True)
json.dump(out, open(f'results/v2/scale_{name}{tag}.json', 'w'), indent=1, default=float)
