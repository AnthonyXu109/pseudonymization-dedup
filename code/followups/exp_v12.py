"""v12 (PROTOCOL_V12_20261006.md): E12 curator re-detection+collapse, E13 keyed MinHash on raw text, E14 Faker-dict baseline."""
import sys, re, os, json, glob, pickle, hmac, hashlib, collections, numpy as np
from common import *
from evalsparse import jaccard_pairs
from exp_v8 import uf, boot_dlost
os.makedirs('results/v12', exist_ok=True)
TH = 0.7
ALIAS = re.compile(r'\[([A-Z_]+?)_\d+\]')

def collapse_detected(texts):
    from detector import detect_many
    out = []
    for t, sp in zip(texts, detect_many(texts)):
        parts, cur = [], 0
        for m in sp:
            parts.append(t[cur:m['start']]); parts.append(f"[{m['type']}]"); cur = m['end']
        parts.append(t[cur:]); out.append(''.join(parts))
    return out

def released(docs, rep):
    return [release(d['text'], d['spans'], rep, d['key'], d['case']) for d in docs]

def decide_sets(docs, sets):
    I, J, V = jaccard_pairs(sets, TH)
    N = len(docs); comp = uf(N, list(zip(I.tolist(), J.tolist())))
    case = np.array([d['case'] for d in docs]); cases = sorted(set(case.tolist()))
    kept = np.where(comp == np.arange(N))[0]; kc = collections.Counter(case[kept].tolist())
    lost = np.array([kc.get(c, 0) == 0 for c in cases])
    oi = {d['case']: i for i, d in enumerate(docs) if not d['is_copy']}
    copies = [i for i, d in enumerate(docs) if d['is_copy']]
    det = np.array([comp[oi[case[i]]] == comp[i] for i in copies])
    return dict(comp=comp, lost=lost, recall=int(det.sum()), n_copies=len(copies), lost_n=int(lost.sum()),
                residual=int(sum(1 for i in copies if kc.get(case[i], 0) >= 2)), oi=oi, cases=cases)

def row(name, r, ref, docs):
    out = dict(recall=f"{r['recall']}/{r['n_copies']}", lost=r['lost_n'], residual=r['residual'])
    if ref is not None: out['vs_RAW'] = boot_dlost(ref, r, docs)
    print(name, out, flush=True); return out

KEY = b'experimental-shared-key-20261006'
def keyed_shingles(t, n=5):
    w = words(t)
    return {int.from_bytes(hmac.new(KEY, ' '.join(w[i:i + n]).encode(), hashlib.sha256).digest()[:8], 'little') for i in range(max(1, len(w) - n + 1))}

def e12_13_tab():
    from build_corpus import build
    docs = build(); res = {}
    ref = decide_sets(docs, [shingles(d['text']) for d in docs]); res['RAW'] = row('RAW', ref, None, docs)
    # E13: keyed shingles on raw text (64-bit ids; jaccard_pairs expects uint32 -> fold)
    ks = [{(h ^ (h >> 32)) & 0xffffffff for h in keyed_shingles(d['text'])} for d in docs]
    res['KEYED-RAW'] = row('KEYED-RAW', decide_sets(docs, ks), ref, docs)
    for rep in ('FAKER-DOC', 'SUR-DOC'):
        T = collapse_detected(released(docs, rep))
        res[f'{rep} redetect-collapse'] = row(f'{rep} redetect', decide_sets(docs, [shingles(t) for t in T]), ref, docs)
    T = [ALIAS.sub(r'[\1]', t) for t in released(docs, 'ALIAS-DOC')]
    res['ALIAS-DOC collapse'] = row('ALIAS collapse', decide_sets(docs, [shingles(t) for t in T]), ref, docs)
    json.dump(res, open('results/v12/e12_e13_tab.json', 'w'), indent=1)

def e12_pairs():
    from build_scale import build_scale
    out = {}
    for nm in ('ecthr', 'c4'):
        docs = build_scale(nm)
        sp = [x for f in sorted(glob.glob(f'results/det/{nm}_*.pkl')) for x in pickle.load(open(f, 'rb'))]
        for d, x in zip(docs, sp): d['spans'] = x
        orig = {d['case']: d for d in docs if not d['is_copy']}
        pairs = [(orig[d['case']], d) for d in docs if d['is_copy'] and d.get('copy_kind') == 'real']
        del docs, sp
        res = {}
        def rec(A, B): return float(np.mean([len((a := shingles(x)) & (b := shingles(y))) / max(1, len(a | b)) >= TH for x, y in zip(A, B)]))
        for rep in ('FAKER-DOC',):
            A = released([o for o, _ in pairs], rep); B = released([c for _, c in pairs], rep)
            res[f'{rep} plain'] = rec(A, B)
            res[f'{rep} redetect-collapse'] = rec(collapse_detected(A), collapse_detected(B)); print(nm, rep, res, flush=True)
        A = [ALIAS.sub(r'[\1]', t) for t in released([o for o, _ in pairs], 'ALIAS-DOC')]
        B = [ALIAS.sub(r'[\1]', t) for t in released([c for _, c in pairs], 'ALIAS-DOC')]
        res['ALIAS-DOC plain'] = rec(released([o for o, _ in pairs], 'ALIAS-DOC'), released([c for _, c in pairs], 'ALIAS-DOC'))
        res['ALIAS-DOC collapse'] = rec(A, B)
        res['RAW'] = rec([o['text'] for o, _ in pairs], [c['text'] for _, c in pairs])
        print(nm, res, flush=True); out[nm] = res
        json.dump(out, open('results/v12/e12_pairs.json', 'w'), indent=1)

def e14():
    import exp_v10 as E  # recomputes the E6 setting on import
    from faker import Faker
    f = Faker('en_US'); f.seed_instance(777)
    D = set()
    for gen in (f.name, f.company, f.city, lambda: f.date(pattern='%d %B %Y'), f.country):
        for _ in range(200000):
            D.update(t for t in re.findall(r'[A-Z][a-z]+', gen()))
    res = {'dict_size': len(D)}
    keep = lambda t: E.df[t] <= 5
    agg = collections.Counter()
    for i, j in E.pairs:
        L = E.leaked(i) & E.names_raw[j]; sur = E.names_rel[i] - E.names_raw[i]
        cand = {t for t in E.names_rel[i] if keep(t)}
        rules = {'one': cand, 'one_minus_dict': {t for t in cand if t not in D},
                 'two': {t for t in cand if t in E.names_rel[j]}, 'two_minus_dict': {t for t in cand if t in E.names_rel[j] and t not in D}}
        for nm, F in rules.items():
            agg[nm + '_flag'] += len(F); agg[nm + '_tp'] += len(F & L); agg[nm + '_sur'] += len(F & sur)
        agg['leaked'] += len(L)
    r = dict(agg)
    for nm in ('one', 'one_minus_dict', 'two', 'two_minus_dict'):
        r[nm + '_precision'] = agg[nm + '_tp'] / max(1, agg[nm + '_flag']); r[nm + '_recall'] = agg[nm + '_tp'] / max(1, agg['leaked'])
    res.update(r); print(json.dumps(res), flush=True)
    json.dump(res, open('results/v12/e14_fakerdict.json', 'w'), indent=1)

if __name__ == '__main__':
    for p in sys.argv[1:]: globals()[p]()
