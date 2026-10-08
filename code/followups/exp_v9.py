"""v9 follow-ups (PROTOCOL_V9_20261006.md): E4 natural cross-source pairs TAB<->LexGLUE, E5 embedding baseline."""
import sys, os, glob, json, pickle, random, collections, numpy as np, pandas as pd
from common import *
os.makedirs('results/v9', exist_ok=True)
SAMPLE_BITS = 16

def jac(a, b): return len(a & b) / len(a | b) if a or b else 0.0

def natural_pairs():
    """TAB judgment paired with the LexGLUE document of the same case (containment >= 0.8); detector spans on both."""
    cache = 'results/v9/natural_pairs.pkl'
    if os.path.exists(cache): return pickle.load(open(cache, 'rb'))
    tab = [d for d in pickle.load(open('results/tab_detected.pkl', 'rb')) if not d['is_copy']]
    from build_scale import build_scale
    ec = [d for d in build_scale('ecthr') if not d['is_copy']]
    sp = [x for f in sorted(glob.glob('results/det/ecthr_*.pkl')) for x in pickle.load(open(f, 'rb'))][:len(ec)]
    for d, x in zip(ec, sp): d['spans'] = x
    S = [shingles(d['text']) for d in ec]
    inv = collections.defaultdict(list)
    for j, s in enumerate(S):
        for h in s:
            if h % SAMPLE_BITS == 0: inv[h].append(j)
    pairs = []
    for d in tab:
        A = shingles(d['text']); c = collections.Counter()
        for h in A:
            if h % SAMPLE_BITS == 0:
                for j in inv.get(h, ()): c[j] += 1
        if not c: continue
        j = c.most_common(1)[0][0]; B = S[j]
        if len(A & B) / len(B) >= 0.8:
            pairs.append((d, dict(ec[j], key='lexglue-' + ec[j]['key'])))
    pickle.dump(pairs, open(cache, 'wb'))
    return pairs

def e4():
    pairs = natural_pairs()
    from dose import marked_shingles
    out = {'n_pairs': len(pairs)}
    for s in ('RAW', 'MASK', 'SUR-CORPUS', 'SUR-DOC', 'HASH-ENTITY'):
        js = [jac(shingles(release(a['text'], a['spans'], s, a['key'], a['case'])), shingles(release(b['text'], b['spans'], s, b['key'], b['case']))) for a, b in pairs]
        js = np.array(js)
        out[s] = dict(recall_05=float(np.mean(js >= 0.5)), recall_07=float(np.mean(js >= 0.7)), median_J=float(np.median(js)))
        print(s, out[s], flush=True)
    ms = []
    for a, _ in pairs:
        allh, unm = marked_shingles(a['text'], a['spans']); ms.append(1 - len(unm) / max(1, len(allh)))
    out['median_m_doc_tab_side'] = float(np.median(ms))
    json.dump(out, open('results/v9/e4_natural.json', 'w'), indent=1)

def embedders():
    import spacy
    trf = spacy.load('en_core_web_trf', exclude=['tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'])
    lg = spacy.load('en_core_web_lg', exclude=['tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner', 'senter'])
    def emb_trf(texts):
        texts = [' '.join(t.split()[:300]) for t in texts]
        out = []
        for doc in trf.pipe(texts, batch_size=8):
            v = doc._.trf_data.last_hidden_layer_state.dataXd.mean(axis=0)
            out.append(np.asarray(v, dtype=np.float32))
        return np.vstack(out)
    def emb_lg(texts):
        return np.vstack([doc.vector for doc in lg.pipe(texts, batch_size=32)]).astype(np.float32)
    return {'roberta': emb_trf, 'static': emb_lg}

def norm(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)

def samples():
    rng = random.Random(20261006)
    from build_scale import build_scale
    sets = {}
    for nm, k in (('ecthr', 300), ('c4', 500)):
        docs = build_scale(nm)
        sp = [x for f in sorted(glob.glob(f'results/det/{nm}_*.pkl')) for x in pickle.load(open(f, 'rb'))]
        orig = {}
        for d, x in zip(docs, sp):
            d['spans'] = x
            if not d['is_copy']: orig[d['case']] = d
        pr = [(orig[d['case']], d) for d in docs if d['is_copy'] and d.get('copy_kind') == 'real']
        sets[nm] = rng.sample(pr, k)
        del docs, sp, orig
    nat = natural_pairs(); sets['natural'] = rng.sample(nat, min(300, len(nat)))
    return sets

def e5(strats=('RAW', 'MASK', 'SUR-DOC'), only=None, path='results/v9/e5_embed.json'):
    sets = samples()
    if only: sets = {k: v for k, v in sets.items() if k in only}
    E = embedders(); out = {}
    for nm, pr in sets.items():
        out[nm] = {}
        for s in strats:
            A = [release(a['text'], a['spans'], s, a['key'], a['case']) for a, _ in pr]
            B = [release(b['text'], b['spans'], s, b['key'], b['case']) for _, b in pr]
            for en, f in E.items():
                XA, XB = norm(f(A)), norm(f(B))
                pos = (XA * XB).sum(1)
                G = XA @ XA.T; iu = np.triu_indices(len(pr), 1); neg = G[iu]
                tau = float(np.quantile(neg, 0.999))
                r = dict(tau=tau, recall=float(np.mean(pos >= tau)), median_pos=float(np.median(pos)), n=len(pr))
                out[nm][f'{s}|{en}'] = r
                print(nm, s, en, r, flush=True)
                json.dump(out, open(path, 'w'), indent=1)

if __name__ == '__main__':
    {'e4': e4, 'e5': e5,
     'e5b': lambda: e5(('FAKER-DOC', 'HASH-ENTITY'), ('ecthr',), 'results/v9/e5b_embed.json')}[sys.argv[1]]()
