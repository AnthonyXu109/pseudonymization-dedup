"""Check the proposition's conditions pair by pair: (C1) within each document every occurrence of a marked token
string is marked; (C2) both documents mark the same strings among those they share. Residuals by condition."""
import json, pickle, re, os, glob, collections, zlib, numpy as np
from common import *
from dose import jac
from build_corpus import build
def info(text, spans, n=5):
    toks = [(mo.start(), mo.end(), mo.group(0).lower()) for mo in TOK.finditer(text)]
    mk = np.zeros(len(toks), bool); si = 0; ss = sorted(spans, key=lambda m: m['start'])
    for i, (a, b, _) in enumerate(toks):
        while si < len(ss) and ss[si]['end'] <= a: si += 1
        mk[i] = si < len(ss) and ss[si]['start'] < b
    marked = {t[2] for t, k in zip(toks, mk) if k}; unmarked = {t[2] for t, k in zip(toks, mk) if not k}
    allh, unm = set(), set()
    for i in range(max(1, len(toks) - n + 1)):
        h = zlib.crc32(' '.join(t[2] for t in toks[i:i + n]).encode()); allh.add(h)
        if not mk[i:i + n].any(): unm.add(h)
    return allh, unm, marked, unmarked, {t[2] for t in toks}
def check(docs, name, cap=None):
    orig = {d['case']: d for d in docs if not d['is_copy']}; rows = []
    for d in docs:
        if not d['is_copy']: continue
        o = orig[d['case']]
        A, UA, MA, NA, TA = info(o['text'], o['spans']); B, UB, MB, NB, TB = info(d['text'], d['spans'])
        c1 = not (MA & NA) and not (MB & NB)
        common = TA & TB
        c2 = (MA & common) == (MB & common)
        S = A & B; J = jac(A, B); m = 1 - len(S & UA & UB) / max(1, len(S))
        pred = (1 - m) * J / (1 + m * J)
        obs = jac(shingles(release(o['text'], o['spans'], 'SUR-DOC', o['key'], o['case'])), shingles(release(d['text'], d['spans'], 'SUR-DOC', d['key'], d['case'])))
        rows.append(dict(c1=c1, c2=c2, err=abs(obs - pred)))
        if cap and len(rows) >= cap: break
    out = {}
    for lab, sel in (('C1 holds', lambda r: r['c1']), ('C1 violated', lambda r: not r['c1'])):
        e = [r['err'] for r in rows if sel(r)]
        out[lab] = dict(n=len(e), mae=float(np.mean(e)) if e else None, max=float(np.max(e)) if e else None)
    print(name, out, flush=True); return out
res = {}
res['TAB gold'] = check(build(), 'TAB gold')
res['TAB detector'] = check(pickle.load(open('results/tab_detected.pkl', 'rb')), 'TAB detector')
res['Enron detector'] = check(pickle.load(open('results/aeslc_detected.pkl', 'rb')), 'Enron detector')
res['TAB second-source'] = check(pickle.load(open('results/tab_real_detected.pkl', 'rb')), 'TAB second-source')
res['Enron second-source'] = check(pickle.load(open('results/enron_real_detected.pkl', 'rb')), 'Enron second-source')
json.dump(res, open('results/v2/dose_conditions.json', 'w'), indent=1)
