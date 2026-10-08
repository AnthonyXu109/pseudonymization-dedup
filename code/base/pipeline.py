"""G2: production-style MinHash pipeline on ECtHR / web (second-source copies).
For each strategy: shingles once; for each seed: 112 MinHash signatures; for each banding (b x r = 112):
datatrove-style linking (documents sharing any band bucket are merged, no verification).
Outputs per (strategy, seed, banding): distinct originals removed, copy recall (original and copy in same component),
plus RAW-vs-strategy cluster bootstrap (clusters = components of the union graph)."""
import sys, glob, pickle, json, collections, numpy as np
from common import *
from build_scale import build_scale
from fastlsh import signatures
name = sys.argv[1]; mode = sys.argv[2] if len(sys.argv) > 2 else 'plain'
scope = sys.argv[3] if len(sys.argv) > 3 else 'broad'
NORM['mode'] = mode
STR = ['RAW', 'MASK', 'SUR-DOC', 'SUR-CORPUS', 'HASH-ENTITY'] if mode == 'plain' else ['RAW', 'MASK', 'SUR-DOC']
if scope == 'narrow': STR = ['RAW', 'TYPE', 'SUR-DOC', 'HASH-ENTITY']
BANDS = [(28, 4), (16, 7), (14, 8), (8, 14), (7, 16)]
SEEDS = [0, 1, 2]
docs = build_scale(name)
spans = [x for f in sorted(glob.glob(f'results/det/{name}_*.pkl')) for x in pickle.load(open(f, 'rb'))]
for d, sp in zip(docs, spans): d['spans'] = sp
if scope == 'narrow':
    import re as _re
    EMAIL = _re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"); PHONE = _re.compile(r"\+?\d[\d ()-]{7,}\d"); IP = _re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    for d in docs:
        ms = []
        for rx, t in ((EMAIL, 'EMAIL_ADDRESS'), (PHONE, 'PHONE_NUMBER'), (IP, 'IP_ADDRESS')):
            ms += [dict(start=m.start(), end=m.end(), type=t, surface=m.group(0), ent=(t, m.group(0).lower())) for m in rx.finditer(d['text'])]
        d['spans'] = resolve(ms)
    print('narrow scope: docs with any span', sum(1 for d in docs if d['spans']), 'of', len(docs), flush=True)
docs = [d for d in docs if not d['is_copy'] or d.get('copy_kind') == 'real']
n = len(docs); case = np.array([d['case'] for d in docs])
origs = sorted(set(case.tolist())); orig_idx = {d['case']: i for i, d in enumerate(docs) if not d['is_copy']}
copies = [i for i, d in enumerate(docs) if d['is_copy']]
has_id = np.array([bool(docs[i]['spans']) or bool(docs[orig_idx[case[i]]]['spans']) for i in copies])
print('copies whose original or copy has a span:', int(has_id.sum()), 'of', len(copies), flush=True)
touched = np.array([bool(docs[i]['spans']) or bool(docs[orig_idx[case[i]]]['spans']) for i in copies])

def uf_components(n, edges):
    p = np.arange(n)
    def f(x):
        r = x
        while p[r] != r: r = p[r]
        while p[x] != r: p[x], x = r, p[x]
        return r
    for a, b in edges:
        ra, rb = f(a), f(b)
        if ra != rb: p[max(ra, rb)] = min(ra, rb)
    return np.array([f(i) for i in range(n)])

def star_edges(sig, b, r):
    E = []
    for bi in range(b):
        first = {}
        for i, row in enumerate(map(bytes, sig[:, bi * r:(bi + 1) * r])):
            j = first.setdefault(row, i)
            if j != i: E.append((j, i))
    return E

res = {}; comps_store = {}
for s in STR:
    sets = [np.fromiter(x, dtype=np.uint32, count=len(x)) for x in (shingles(release(d['text'], d['spans'], s, d['key'], d['case'])) for d in docs)]
    for seed in SEEDS:
        sig = signatures(sets, num_perm=112, seed=seed)
        for b, r in BANDS:
            comp = uf_components(n, star_edges(sig, b, r))
            kept_cases = set(case[np.where(comp == np.arange(n))[0]].tolist())
            lost = np.array([c not in kept_cases for c in origs])
            same = np.array([comp[orig_idx[case[i]]] == comp[i] for i in copies])
            res[f'{s}|{seed}|{b}x{r}'] = dict(removed=int(lost.sum()), recall=float(same.mean()), recall_touched=float(same[touched].mean()) if touched.any() else None)
            if seed == 0 and (b, r) == (14, 8): comps_store[s] = (comp, lost, same)
        print(name, mode, s, seed, {k.split('|')[2]: (v['removed'], round(v['recall'], 3)) for k, v in res.items() if k.startswith(f'{s}|{seed}|')}, flush=True)
    del sets
# cluster bootstrap RAW vs each strategy at seed 0, 14x8: clusters = components of union graph of both runs
rng = np.random.default_rng(20261001); boot = {}
c_raw, lost_raw, same_raw = comps_store['RAW']
orig_pos = np.array([orig_idx[c] for c in origs]); copy_pos = np.array(copies)
for s in STR:
    if s == 'RAW' or s not in comps_store: continue
    c_s, lost_s, same_s = comps_store[s]
    # union of the two partitions: merge labels
    pairs = list(zip(range(n), c_raw.tolist())) + list(zip(range(n), c_s.tolist()))
    U = uf_components(n, pairs)
    cl_orig = U[orig_pos]; cl_copy = U[copy_pos]
    labels = np.unique(np.concatenate([cl_orig, cl_copy])); lab_index = {l: k for k, l in enumerate(labels)}
    go = np.array([lab_index[x] for x in cl_orig]); gc = np.array([lab_index[x] for x in cl_copy])
    dl = np.bincount(go, weights=lost_s.astype(float) - lost_raw.astype(float), minlength=len(labels))
    dr = np.bincount(gc, weights=same_s.astype(float) - same_raw.astype(float), minlength=len(labels))
    nc = np.bincount(gc, minlength=len(labels)).astype(float)
    B = 2000; dL, dR = [], []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(labels), len(labels)), minlength=len(labels))
        dL.append((w * dl).sum()); dR.append((w * dr).sum() / max(1, (w * nc).sum()))
    boot[s] = dict(d_removed=float(dl.sum()), d_removed_ci=[float(np.percentile(dL, 2.5)), float(np.percentile(dL, 97.5))],
                   d_recall=float(dr.sum() / nc.sum()), d_recall_ci=[float(np.percentile(dR, 2.5)), float(np.percentile(dR, 97.5))],
                   n_clusters=int(len(labels)), added=int((lost_s & ~lost_raw).sum()), restored=int((~lost_s & lost_raw).sum()))
    print('boot', s, boot[s], flush=True)
json.dump(dict(results=res, boot=boot, n_orig=len(origs), n_copies=len(copies), n_copies_touched=int(touched.sum())), open(f'results/v2/pipeline_{name}_{mode}' + ('_narrow' if scope == 'narrow' else '') + '.json', 'w'), indent=1)
