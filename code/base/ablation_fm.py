"""Ablations: which identifier categories drive false merges; country skew; MinHash-LSH realization."""
import json, collections, numpy as np
from common import *
from build_corpus import build
from datasketch import MinHash, MinHashLSH

D = load_tab()
TH = 0.7
iu = np.triu_indices(len(D), 1)
def fm_stats(spans_fn, strat='MASK'):
    rel = [release(d['text'], spans_fn(d), strat, d['doc_id']) for d in D]
    J = jaccard_matrix([shingles(r) for r in rel])
    ei = np.argwhere(np.triu(J >= TH, 1))
    comp = components(len(D), ei.tolist())
    kept = {c for c in comp}
    return int((J[iu] >= TH).sum()), len(D) - len(kept), J
out = {}
base_pairs, base_lost, Jraw = fm_stats(lambda d: [], 'RAW')
out['RAW'] = dict(pairs=base_pairs, lost=base_lost)
out['MASK DIRECT only'] = dict(zip(('pairs', 'lost'), fm_stats(lambda d: gold_spans(d, idtypes=('DIRECT',)))[:2]))
out['MASK QUASI only'] = dict(zip(('pairs', 'lost'), fm_stats(lambda d: gold_spans(d, idtypes=('QUASI',)))[:2]))
p, l, Jmask = fm_stats(lambda d: gold_spans(d))
out['MASK DIRECT+QUASI'] = dict(pairs=p, lost=l)
for t in ['PERSON', 'CODE', 'DATETIME', 'ORG', 'LOC', 'DEM', 'QUANTITY', 'MISC']:
    out[f'MASK only {t}'] = dict(zip(('pairs', 'lost'), fm_stats(lambda d, t=t: gold_spans(d, types=(t,)))[:2]))
    out[f'MASK all but {t}'] = dict(zip(('pairs', 'lost'), fm_stats(lambda d, t=t: gold_spans(d, types=tuple(x for x in ['PERSON','CODE','DATETIME','ORG','LOC','DEM','QUANTITY','MISC'] if x != t)))[:2]))
for k, v in out.items(): print(f'{k:28s}', v)

# country skew: cases removed by dedup (not first in component) RAW vs MASK
def removed(J):
    ei = np.argwhere(np.triu(J >= TH, 1)); comp = components(len(D), ei.tolist())
    return [i for i, c in enumerate(comp) if c != i]
cty = [d['meta'].get('countries') for d in D]
allc = collections.Counter(cty)
rr, rm = collections.Counter(cty[i] for i in removed(Jraw)), collections.Counter(cty[i] for i in removed(Jmask))
print('removed RAW', dict(rr), 'MASK', dict(rm))
print('corpus share TUR %.3f' % (allc['TUR'] / len(D)))
out['country_removed'] = dict(RAW=dict(rr), MASK=dict(rm), corpus=dict(allc))

json.dump(out, open('results/ablation.json', 'w'), indent=1)
