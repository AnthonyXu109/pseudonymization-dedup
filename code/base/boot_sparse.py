import sys, pickle, json, collections, numpy as np
from common import *
from evalsparse import jaccard_pairs
name = sys.argv[1]; th = 0.7
docs = pickle.load(open(f'results/{name}_detected.pkl', 'rb'))
case = np.array([d['case'] for d in docs]); ncase = case.max() + 1
def lostvec(s):
    I, J, V = jaccard_pairs([shingles(release(d['text'], d['spans'], s, d['key'])) for d in docs], th)
    comp = components(len(docs), list(zip(I.tolist(), J.tolist())))
    kept = [i for i, c in enumerate(comp) if c == i]
    kc = set(case[kept].tolist())
    return np.array([c not in kc for c in range(ncase)])
base = lostvec('RAW'); out = {}
rng = np.random.default_rng(20261001)
idx = rng.integers(0, ncase, (2000, ncase))
for s in ['MASK', 'TYPE', 'DROP']:
    v = lostvec(s); d = v[idx].mean(1) - base[idx].mean(1)
    out[s] = dict(diff=float(v.mean() - base.mean()), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)), n_lost=int(v.sum()), n_raw=int(base.sum()))
    print(s, out[s], flush=True)
json.dump(out, open(f'results/boot_{name}.json', 'w'), indent=1)
