"""v11 (PROTOCOL_V11_20261006.md): E7 mitigations on TAB, E8 pair recall on ECtHR/web, E9 keyless collapse under
detector mismatch, E10 m_doc screening per setting, E11 containment on TAB-LexGLUE."""
import sys, re, os, json, glob, pickle, random, collections, numpy as np
from common import *
from evalsparse import jaccard_pairs
from exp_v8 import uf, boot_dlost
os.makedirs('results/v11', exist_ok=True)
TH = 0.7
ZQ = re.compile(r'\bzq[a-z]{10}\b'); HEX = re.compile(r'\b[0-9a-f]{64}\b'); YEAR = re.compile(r'\b(?:1[89]|20)\d\d\b')

def release_year(d, base):
    text, spans = d['text'], d['spans']
    out, cur, st = [], 0, new_state()
    for idx, m in enumerate(spans):
        out.append(text[cur:m['start']])
        if m['type'] == 'DATETIME':
            out.append(' '.join(YEAR.findall(m['surface'])))
        else:
            out.append(replace_span(m, idx, base, d['key'], d['case'], st))
        cur = m['end']
    out.append(text[cur:])
    return ''.join(out)

def text_for(d, rep):
    if rep == 'SUR-DOC-SKIP': return ZQ.sub(' ', release(d['text'], d['spans'], 'SUR-DOC', d['key'], d['case']))
    if rep == 'HASH-SKIP': return HEX.sub(' ', release(d['text'], d['spans'], 'HASH-ENTITY', d['key'], d['case']))
    if rep == 'SUR-DOC-COLLAPSED': return ZQ.sub('[PII]', release(d['text'], d['spans'], 'SUR-DOC', d['key'], d['case']))
    if rep == 'YEAR': return release_year(d, 'SUR-DOC')
    if rep == 'YEAR-FAKER': return release_year(d, 'FAKER-DOC')
    return release(d['text'], d['spans'], rep, d['key'], d['case'])

def decide(docs, rep, n=5):
    sets = [shingles(text_for(d, rep), n) for d in docs]
    I, J, V = jaccard_pairs(sets, TH)
    N = len(docs); comp = uf(N, list(zip(I.tolist(), J.tolist())))
    case = np.array([d['case'] for d in docs]); cases = sorted(set(case.tolist()))
    kept = np.where(comp == np.arange(N))[0]; kc = collections.Counter(case[kept].tolist())
    lost = np.array([kc.get(c, 0) == 0 for c in cases])
    oi = {d['case']: i for i, d in enumerate(docs) if not d['is_copy']}
    copies = [i for i, d in enumerate(docs) if d['is_copy']]
    det = np.array([comp[oi[case[i]]] == comp[i] for i in copies])
    resid = int(sum(1 for i in copies if kc.get(case[i], 0) >= 2))
    return dict(comp=comp, lost=lost, recall=int(det.sum()), n_copies=len(copies), lost_n=int(lost.sum()), residual=resid,
                oi=oi, cases=cases)

def row(name, r, ref, docs):
    out = dict(recall=f"{r['recall']}/{r['n_copies']}", lost=r['lost_n'], residual=r['residual'])
    if ref is not None: out['vs_RAW5'] = boot_dlost(ref, r, docs)
    print(name, out, flush=True); return out

def e7():
    from build_corpus import build
    docs = build(); res = {}
    ref = decide(docs, 'RAW', 5); res['RAW n5'] = row('RAW n5', ref, None, docs)
    for n in (3, 2):
        res[f'RAW n{n}'] = row(f'RAW n{n}', decide(docs, 'RAW', n), ref, docs)
    for rep in ('SUR-DOC', 'FAKER-DOC'):
        for n in (5, 3, 2):
            res[f'{rep} n{n}'] = row(f'{rep} n{n}', decide(docs, rep, n), ref, docs)
    for rep in ('SUR-DOC-SKIP', 'HASH-SKIP', 'YEAR', 'YEAR-FAKER', 'ALIAS-DOC', 'DROP'):
        res[rep] = row(rep, decide(docs, rep, 5), ref, docs)
    json.dump(res, open('results/v11/e7_tab.json', 'w'), indent=1)

def e9():
    from build_corpus import build
    from detector import detect_many
    gold = build(); det = pickle.load(open('results/tab_detected.pkl', 'rb'))
    ref = decide(gold, 'RAW', 5)
    cidx = [i for i, d in enumerate(det) if d['is_copy']]
    sm = detect_many([det[i]['text'] for i in cidx], model='en_core_web_sm')
    lgsm = [dict(d) for d in det]
    for i, s_ in zip(cidx, sm): lgsm[i]['spans'] = s_
    res = {}
    for nm, docs in (('lg/lg', det), ('lg/sm', lgsm)):
        for rep in ('SUR-DOC-COLLAPSED', 'SUR-DOC-SKIP', 'SUR-DOC'):
            res[f'{nm} {rep}'] = row(f'{nm} {rep}', decide(docs, rep, 5), ref, gold)
    json.dump(res, open('results/v11/e9_mismatch.json', 'w'), indent=1)

def e8():
    from build_scale import build_scale
    out = {}
    for nm in ('ecthr', 'c4'):
        docs = build_scale(nm)
        sp = [x for f in sorted(glob.glob(f'results/det/{nm}_*.pkl')) for x in pickle.load(open(f, 'rb'))]
        for d, x in zip(docs, sp): d['spans'] = x
        orig = {d['case']: d for d in docs if not d['is_copy']}
        pairs = [(orig[d['case']], d) for d in docs if d['is_copy'] and d.get('copy_kind') == 'real']
        rng = random.Random(20261006); ol = list(orig.values())
        neg = [tuple(rng.sample(range(len(ol)), 2)) for _ in range(20000)]
        del docs, sp
        res = {}
        for n in (5, 3, 2):
            fp = np.mean([len((A := shingles(ol[i]['text'], n)) & (B := shingles(ol[j]['text'], n))) / max(1, len(A | B)) >= TH for i, j in neg])
            res[f'random_pairs_ge_0.7 n{n}'] = float(fp); print(nm, n, 'fp', fp, flush=True)
        reps = [(r, n) for r in ('RAW', 'SUR-DOC', 'HASH-ENTITY', 'FAKER-DOC') for n in (5, 3, 2)] + \
               [(r, 5) for r in ('SUR-DOC-SKIP', 'HASH-SKIP', 'YEAR', 'YEAR-FAKER')]
        for rep, n in reps:
            js = []
            for o, c in pairs:
                A, B = shingles(text_for(o, rep), n), shingles(text_for(c, rep), n)
                js.append(len(A & B) / max(1, len(A | B)))
            res[f'{rep} n{n}'] = float(np.mean(np.array(js) >= TH)); print(nm, rep, n, res[f'{rep} n{n}'], flush=True)
            json.dump(dict(out, **{nm: res}), open('results/v11/e8_pairs.json', 'w'), indent=1)
        out[nm] = res
    json.dump(out, open('results/v11/e8_pairs.json', 'w'), indent=1)

def e10():
    rows = json.load(open('results/v2/dose_pairs.json')); out = {}
    for s in sorted(set(r['setting'] for r in rows)):
        E, MS = 1e-9, 0.3 / 1.7   # same definitions as dose_metrics.py (J_surdoc, EPS tolerance)
        R = [r for r in rows if r['setting'] == s and r['J_raw'] >= TH - E]
        drop = [r for r in R if r['J_surdoc'] < TH - E]; keep = [r for r in R if r['J_surdoc'] >= TH - E]
        out[s] = dict(raw_ge=len(R), drops=len(drop), flagged=sum(r['m_doc'] > MS + E for r in drop),
                      missed=sum(r['m_doc'] <= MS + E for r in drop), false_alarms=sum(r['m_doc'] > MS + E for r in keep),
                      median_m_doc=float(np.median([r['m_doc'] for r in rows if r['setting'] == s])))
        print(s, out[s])
    json.dump(out, open('results/v11/e10_screen.json', 'w'), indent=1)

def e11():
    pairs = pickle.load(open('results/v9/natural_pairs.pkl', 'rb')); out = {}
    for s in ('RAW', 'MASK', 'SUR-CORPUS', 'SUR-DOC', 'HASH-ENTITY', 'FAKER-DOC'):
        c = []
        for a, b in pairs:
            A = shingles(release(a['text'], a['spans'], s, a['key'], a['case'])); B = shingles(release(b['text'], b['spans'], s, b['key'], b['case']))
            c.append(len(A & B) / max(1, len(B)))
        c = np.array(c); out[s] = dict(share_ge_08=float(np.mean(c >= 0.8)), share_ge_05=float(np.mean(c >= 0.5)), median=float(np.median(c)))
        print(s, out[s], flush=True)
    json.dump(out, open('results/v11/e11_containment.json', 'w'), indent=1)

if __name__ == '__main__':
    for p in sys.argv[1:]:
        globals()[p]()
