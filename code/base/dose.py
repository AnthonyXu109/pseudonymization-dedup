"""F1: dose-response. For each original/copy pair: m = share of the original's shingles that touch a marked token;
observed Jaccard after SUR-DOC vs prediction J' = (1-m)J/(1+mJ) from the RAW Jaccard J."""
import json, pickle, re, os, sys, zlib, collections, numpy as np
from common import *
from build_corpus import build

def marked_shingles(text, spans, n=5):
    toks = [(mo.start(), mo.end(), mo.group(0).lower()) for mo in TOK.finditer(text)]
    if NORM['mode'] == 'datatrove':
        raise NotImplementedError
    mk = np.zeros(len(toks), bool)
    si = 0; ss = sorted(spans, key=lambda m: m['start'])
    for i, (a, b, _) in enumerate(toks):
        while si < len(ss) and ss[si]['end'] <= a: si += 1
        mk[i] = si < len(ss) and ss[si]['start'] < b
    allh, unm = set(), set()
    for i in range(max(1, len(toks) - n + 1)):
        h = zlib.crc32(' '.join(t[2] for t in toks[i:i + n]).encode())
        allh.add(h)
        if not mk[i:i + n].any(): unm.add(h)
    return allh, unm

def jac(a, b): return len(a & b) / len(a | b) if a or b else 0.0

def pairs_for(docs, name, extra=('HASH-ENTITY', 'FAKER-DOC')):
    orig = {d['case']: d for d in docs if not d['is_copy']}
    rows = []
    for d in docs:
        if not d['is_copy']: continue
        o = orig[d['case']]
        A, UA = marked_shingles(o['text'], o['spans']); B, UB = marked_shingles(d['text'], d['spans'])
        S = A & B
        J = jac(A, B)
        m_doc = 1 - len(UA) / max(1, len(A))
        m_sh = 1 - len(S & UA & UB) / max(1, len(S))
        row = dict(setting=name, m=m_doc, m_doc=m_doc, m_shared=m_sh, J_raw=J, natural=d.get('natural', False),
                   J_pred=(1 - m_doc) * J / (1 + m_doc * J), J_pred_shared=(1 - m_sh) * J / (1 + m_sh * J))
        for st in ('SUR-DOC',) + tuple(extra):
            row['J_' + st] = jac(shingles(release(o['text'], o['spans'], st, o['key'], o['case'])), shingles(release(d['text'], d['spans'], st, d['key'], d['case'])))
        row['J_surdoc'] = row['J_SUR-DOC']
        rows.append(row)
    return rows

if __name__ == '__main__':
    rows = []
    rows += pairs_for(build(), 'TAB gold')
    rows += pairs_for(pickle.load(open('results/tab_detected.pkl', 'rb')), 'TAB detector')
    en = pickle.load(open('results/aeslc_detected.pkl', 'rb'))
    rows += pairs_for(en, 'Enron detector')
    # natural exact repeats and Dolma-like spans (rebuild as in e4v2)
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
    rows += pairs_for([d for d in en if not d['is_copy']] + nat, 'Enron natural')
    EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"); PHONE = re.compile(r"\+?\d[\d ()-]{7,}\d"); IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    def dolma(d):
        ms = []
        for rx, t in ((EMAIL, 'EMAIL_ADDRESS'), (PHONE, 'PHONE_NUMBER'), (IP, 'IP_ADDRESS')):
            ms += [dict(start=m.start(), end=m.end(), type=t, surface=m.group(0), ent=(t, m.group(0).lower())) for m in rx.finditer(d['text'])]
        return dict(d, spans=resolve(ms))
    rows += pairs_for([dolma(d) for d in en], 'Enron Dolma-like')
    rows += pairs_for(pickle.load(open('results/tab_real_detected.pkl', 'rb')), 'TAB second-source')
    rows += pairs_for(pickle.load(open('results/enron_real_detected.pkl', 'rb')), 'Enron second-source')
    from build_scale import build_scale
    import glob
    for nm in ('ecthr', 'c4'):
        docs = build_scale(nm)
        sp = [x for f in sorted(glob.glob(f'results/det/{nm}_*.pkl')) for x in pickle.load(open(f, 'rb'))]
        for d, x in zip(docs, sp): d['spans'] = x
        need = {d['case'] for d in docs if d['is_copy'] and d.get('copy_kind') == 'real'}
        sub = [d for d in docs if (not d['is_copy'] and d['case'] in need) or (d['is_copy'] and d.get('copy_kind') == 'real')]
        rows += pairs_for(sub, {'ecthr': 'ECtHR second-source', 'c4': 'Web second-source'}[nm])
        del docs, sub
    json.dump(rows, open('results/v2/dose_pairs.json', 'w'))
    TH = 0.7
    print(f"{'setting':18s} {'n':>5s} {'median m':>9s} {'rec RAW':>8s} {'rec SUR':>8s} {'rec pred':>8s} {'MAE J':>6s}")
    for s in dict.fromkeys(r['setting'] for r in rows):
        R = [r for r in rows if r['setting'] == s]
        m = np.array([r['m'] for r in R]); jr = np.array([r['J_raw'] for r in R]); js = np.array([r['J_surdoc'] for r in R]); jp = np.array([r['J_pred'] for r in R])
        print(f"{s:18s} {len(R):5d} {np.median(m):9.3f} {np.mean(jr>=TH):8.3f} {np.mean(js>=TH):8.3f} {np.mean(jp>=TH):8.3f} {np.mean(np.abs(js-jp)):6.3f}")
    print('rule: exact copies survive SUR-DOC at theta iff m <= (1-theta)/(1+theta):', {t: round((1 - t) / (1 + t), 3) for t in (0.5, 0.6, 0.7, 0.75, 0.8, 0.9)})
