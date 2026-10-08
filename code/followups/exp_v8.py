"""v8 follow-ups (see PROTOCOL_V8_20261006.md). E1/E2 on TAB with the Table IV setting; E3 scope ablation."""
import sys, re, json, os, glob, pickle, collections, numpy as np
from common import *
from evalsparse import jaccard_pairs
os.makedirs('results/v8', exist_ok=True)
TH = 0.7
ZQ = re.compile(r'\bzq[a-z]{10}\b'); HEX = re.compile(r'\b[0-9a-f]{64}\b')

def uf(n, edges):
    p = list(range(n))
    def f(x):
        while p[x] != x:
            p[x] = p[p[x]]; x = p[x]
        return x
    for a, b in edges:
        ra, rb = f(a), f(b)
        if ra != rb: p[max(ra, rb)] = min(ra, rb)
    return np.array([f(i) for i in range(n)])

def decide(docs, rep):
    texts = []
    for d in docs:
        if rep == 'SUR-DOC-COLLAPSED':
            t = ZQ.sub('[PII]', release(d['text'], d['spans'], 'SUR-DOC', d['key'], d['case']))
        elif rep == 'HASH-COLLAPSED':
            t = HEX.sub('[PII]', release(d['text'], d['spans'], 'HASH-ENTITY', d['key'], d['case']))
        else:
            t = release(d['text'], d['spans'], rep, d['key'], d['case'])
        texts.append(t)
    sets = [shingles(t) for t in texts]
    I, J, V = jaccard_pairs(sets, TH)
    n = len(docs); comp = uf(n, list(zip(I.tolist(), J.tolist())))
    case = np.array([d['case'] for d in docs]); cases = sorted(set(case.tolist()))
    kept = np.where(comp == np.arange(n))[0]; kc = collections.Counter(case[kept].tolist())
    lost = np.array([kc.get(c, 0) == 0 for c in cases])
    oi = {d['case']: i for i, d in enumerate(docs) if not d['is_copy']}
    copies = [i for i, d in enumerate(docs) if d['is_copy']]
    det = np.array([comp[oi[case[i]]] == comp[i] for i in copies])
    resid = int(sum(1 for i in copies if kc.get(case[i], 0) >= 2))
    return dict(comp=comp, lost=lost, det=det, recall=int(det.sum()), n_copies=len(copies), lost_n=int(lost.sum()), residual=resid,
                oi=oi, cases=cases)

def boot_dlost(ref, alt, docs, B=2000, seed=20261001):
    n = len(docs)
    U = uf(n, list(zip(range(n), ref['comp'].tolist())) + list(zip(range(n), alt['comp'].tolist())))
    cl = np.array([U[ref['oi'][c]] for c in ref['cases']])
    labs, inv = np.unique(cl, return_inverse=True)
    d = np.bincount(inv, weights=alt['lost'].astype(float) - ref['lost'].astype(float), minlength=len(labs))
    rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(labs), len(labs)), minlength=len(labs)); bs.append((w * d).sum())
    return dict(d_lost=float(d.sum()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                clusters_changed=int((d != 0).sum()))

def row(name, r, ref):
    out = dict(recall=f"{r['recall']}/{r['n_copies']}", lost=r['lost_n'], residual_dups=r['residual'])
    if ref is not None: out['vs_RAW'] = boot_dlost(ref, r, DOCS)
    print(name, out, flush=True); return out

if __name__ == '__main__':
    part = sys.argv[1]
    if part == 'tab':
        from build_corpus import build
        gold = build(); det = pickle.load(open('results/tab_detected.pkl', 'rb'))
        DOCS = gold
        res = {}
        ref = decide(gold, 'RAW'); res['RAW (gold)'] = row('RAW', ref, None)
        for rep in ('SUR-CORPUS', 'MASK', 'TYPE', 'SUR-DOC', 'SUR-DOC-COLLAPSED', 'HASH-COLLAPSED'):
            res[f'{rep} (gold)'] = row(rep, decide(gold, rep), ref)
        # E2: shared key, different markers
        res['SUR-CORPUS detector/detector'] = row('SC det/det', decide(det, 'SUR-CORPUS'), ref)
        mixed = [dict(g, spans=(dd['spans'] if g['is_copy'] else g['spans'])) for g, dd in zip(gold, det)]
        res['SUR-CORPUS gold/lg-detector'] = row('SC gold/lg', decide(mixed, 'SUR-CORPUS'), ref)
        from detector import detect_many
        cidx = [i for i, d in enumerate(det) if d['is_copy']]
        sm = detect_many([det[i]['text'] for i in cidx], model='en_core_web_sm')
        lgsm = [dict(d) for d in det]
        for i, s_ in zip(cidx, sm): lgsm[i]['spans'] = s_
        res['SUR-CORPUS lg/sm'] = row('SC lg/sm', decide(lgsm, 'SUR-CORPUS'), ref)
        # RAW is unaffected by markers, so ref stays the same for all rows
        json.dump(res, open('results/v8/e1_e2_tab.json', 'w'), indent=1)
    if part == 'scope':
        from build_scale import build_scale
        from dose import marked_shingles
        SC = {'PERSON': {'PERSON'}, 'PERSON+LOC': {'PERSON', 'LOC'}, 'PERSON+LOC+DATETIME': {'PERSON', 'LOC', 'DATETIME'}, 'full': None}
        out = {}
        for nm in ('ecthr', 'c4'):
            docs = build_scale(nm)
            sp = [x for f in sorted(glob.glob(f'results/det/{nm}_*.pkl')) for x in pickle.load(open(f, 'rb'))]
            for d, x in zip(docs, sp): d['spans'] = x
            orig = {d['case']: d for d in docs if not d['is_copy']}
            pairs = [(orig[d['case']], d) for d in docs if d['is_copy'] and d.get('copy_kind') == 'real']
            del docs, sp
            res = {}
            for scn, types in SC.items():
                f = (lambda s: s) if types is None else (lambda s, T=types: [m for m in s if m['type'] in T])
                js, ms = [], []
                for o, c in pairs:
                    so, sc = f(o['spans']), f(c['spans'])
                    A = shingles(release(o['text'], so, 'SUR-DOC', o['key'], o['case'])); B = shingles(release(c['text'], sc, 'SUR-DOC', c['key'], c['case']))
                    js.append(len(A & B) / len(A | B) if A | B else 0.0)
                    allh, unm = marked_shingles(o['text'], so); ms.append(1 - len(unm) / max(1, len(allh)))
                res[scn] = dict(surdoc_pair_recall=float(np.mean(np.array(js) >= TH)), median_m_doc=float(np.median(ms)), n=len(pairs))
                print(nm, scn, res[scn], flush=True)
            A0 = [shingles(o['text']) for o, _ in pairs]; B0 = [shingles(c['text']) for _, c in pairs]
            res['RAW'] = dict(raw_pair_recall=float(np.mean([len(a & b) / len(a | b) >= TH for a, b in zip(A0, B0)])))
            print(nm, 'RAW', res['RAW'], flush=True)
            out[nm] = res
        json.dump(out, open('results/v8/e3_scope.json', 'w'), indent=1)
