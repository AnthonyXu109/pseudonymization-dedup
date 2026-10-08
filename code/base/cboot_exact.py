"""Cluster bootstrap for Table II rows (exact Jaccard): clusters = components of the union of the RAW and MASK graphs."""
import json, pickle, numpy as np
from common import *
from evalsparse import jaccard_pairs
def comps(docs, s, th):
    I, J, V = jaccard_pairs([shingles(release(d['text'], d['spans'], s, d['key'], d['case'])) for d in docs], th)
    return components(len(docs), list(zip(I.tolist(), J.tolist()))), list(zip(I.tolist(), J.tolist()))
def boot(docs, th, s='MASK', B=4000, seed=20261001):
    n = len(docs); case = np.array([d['case'] for d in docs]); origs = sorted(set(case.tolist()))
    cr, er = comps(docs, 'RAW', th); cs, es = comps(docs, s, th)
    U = np.array(components(n, er + es))
    kept = lambda c: set(case[[i for i in range(n) if c[i] == i]].tolist())
    kr, ks = kept(cr), kept(cs)
    lr = np.array([c not in kr for c in origs]); ls = np.array([c not in ks for c in origs])
    oi = {d['case']: i for i, d in enumerate(docs) if not d['is_copy']}
    lab = U[[oi[c] for c in origs]]; u, g = np.unique(lab, return_inverse=True)
    d = np.bincount(g, weights=ls.astype(float) - lr.astype(float), minlength=len(u))
    rng = np.random.default_rng(seed)
    bs = [(np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)) * d).sum() for _ in range(B)]
    nz = int((d != 0).sum())
    return dict(raw=int(lr.sum()), strat=int(ls.sum()), diff=float(d.sum()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                clusters_changed=nz, added=int((ls & ~lr).sum()), restored=int((~ls & lr).sum()))
out = {}
for name, f in (('TAB', 'results/tab_detected.pkl'), ('Enron', 'results/aeslc_detected.pkl')):
    docs = pickle.load(open(f, 'rb'))
    for mode in ('plain', 'datatrove'):
        NORM['mode'] = mode
        for th in (0.7, 0.75):
            out[f'{name}|{mode}|{th}'] = boot(docs, th); print(name, mode, th, out[f'{name}|{mode}|{th}'], flush=True)
json.dump(out, open('results/v2/cboot_exact.json', 'w'), indent=1)
