"""Condition check for all nine settings of Fig. 1: per pair, whether marking is consistent within both documents (C1),
with the residual of Eq. (1). Output results/v2/dose_c1_pairs.json (same settings and order as dose_pairs.json)."""
import json, pickle, re, os, glob, collections, numpy as np
from common import *
from dose import jac
import zlib
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
from build_corpus import build

def rows_for(docs, name):
    orig = {d['case']: d for d in docs if not d['is_copy']}; out = []
    for d in docs:
        if not d['is_copy']: continue
        o = orig[d['case']]
        A, UA, MA, NA, TA = info(o['text'], o['spans']); B, UB, MB, NB, TB = info(d['text'], d['spans'])
        c1 = not (MA & NA) and not (MB & NB)
        S = A & B; J = jac(A, B); m = 1 - len(S & UA & UB) / max(1, len(S))
        pred = (1 - m) * J / (1 + m * J)
        obs = jac(shingles(release(o['text'], o['spans'], 'SUR-DOC', o['key'], o['case'])), shingles(release(d['text'], d['spans'], 'SUR-DOC', d['key'], d['case'])))
        out.append(dict(setting=name, c1=bool(c1), m_shared=m, pred=pred, obs=obs))
    print(name, len(out), sum(r['c1'] for r in out), flush=True); return out

if __name__ == '__main__':
    rows = []
    rows += rows_for(build(), 'TAB gold')
    rows += rows_for(pickle.load(open('results/tab_detected.pkl', 'rb')), 'TAB detector')
    en = pickle.load(open('results/aeslc_detected.pkl', 'rb'))
    rows += rows_for(en, 'Enron detector')
    src = './enron_subject_line'
    texts = [open(os.path.join(src, sp, f), encoding='utf-8', errors='replace').read().split('@subject')[0].strip()
             for sp in ('train', 'dev', 'test') for f in sorted(os.listdir(os.path.join(src, sp)))]
    orig = {d['case']: d for d in en if not d['is_copy']}
    k2c = {' '.join(d['text'].split()): c for c, d in orig.items()}
    seen = collections.Counter(); nat = []
    for i, t in enumerate(texts):
        k = ' '.join(t.split())
        if k in k2c:
            seen[k] += 1
            if seen[k] >= 2: nat.append({**orig[k2c[k]], 'key': f'aeslc-nat-{i}', 'is_copy': True, 'natural': True})
    rows += rows_for([d for d in en if not d['is_copy']] + nat, 'Enron natural')
    EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"); PHONE = re.compile(r"\+?\d[\d ()-]{7,}\d"); IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    def dolma(d):
        ms = []
        for rx, t in ((EMAIL, 'EMAIL_ADDRESS'), (PHONE, 'PHONE_NUMBER'), (IP, 'IP_ADDRESS')):
            ms += [dict(start=m.start(), end=m.end(), type=t, surface=m.group(0), ent=(t, m.group(0).lower())) for m in rx.finditer(d['text'])]
        return dict(d, spans=resolve(ms))
    rows += rows_for([dolma(d) for d in en], 'Enron Dolma-like')
    rows += rows_for(pickle.load(open('results/tab_real_detected.pkl', 'rb')), 'TAB second-source')
    rows += rows_for(pickle.load(open('results/enron_real_detected.pkl', 'rb')), 'Enron second-source')
    from build_scale import build_scale
    for nm in ('ecthr', 'c4'):
        docs = build_scale(nm)
        sp = [x for f in sorted(glob.glob(f'results/det/{nm}_*.pkl')) for x in pickle.load(open(f, 'rb'))]
        for d, x in zip(docs, sp): d['spans'] = x
        need = {d['case'] for d in docs if d['is_copy'] and d.get('copy_kind') == 'real'}
        sub = [d for d in docs if (not d['is_copy'] and d['case'] in need) or (d['is_copy'] and d.get('copy_kind') == 'real')]
        rows += rows_for(sub, {'ecthr': 'ECtHR second-source', 'c4': 'Web second-source'}[nm])
        del docs, sub
    json.dump(rows, open('results/v2/dose_c1_pairs.json', 'w'))
    summ = {}
    for s in dict.fromkeys(r['setting'] for r in rows):
        for lab, v in (('holds', True), ('violated', False)):
            e = [abs(r['obs'] - r['pred']) for r in rows if r['setting'] == s and r['c1'] == v]
            summ[f'{s}|{lab}'] = dict(n=len(e), mae=float(np.mean(e)) if e else None, max=float(np.max(e)) if e else None)
    for lab, v in (('holds', True), ('violated', False)):
        e = [abs(r['obs'] - r['pred']) for r in rows if r['c1'] == v]
        summ[f'ALL|{lab}'] = dict(n=len(e), mae=float(np.mean(e)), max=float(np.max(e)))
    json.dump(summ, open('results/v2/dose_c1_summary.json', 'w'), indent=1)
    print(json.dumps(summ, indent=0))
