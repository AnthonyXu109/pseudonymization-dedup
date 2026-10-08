"""Sparse version of evaluate() for large corpora: only pairs sharing >=1 shingle are scored."""
import collections, numpy as np, scipy.sparse as sp
from common import *
from evalcore import mention_release, cross_pairs, linkage

def jaccard_pairs(sets, th_min):
    """sets: iterable of shingle sets or uint32 arrays (unique). Memory-lean: numpy column ids."""
    arrs = [np.fromiter(x, dtype=np.uint32, count=len(x)) if not isinstance(x, np.ndarray) else x for x in sets]
    sz = np.array([len(x) for x in arrs], dtype=np.float64)
    rows = np.repeat(np.arange(len(arrs), dtype=np.int32), sz.astype(np.int64))
    allh = np.concatenate(arrs) if arrs else np.zeros(0, np.uint32)
    del arrs
    uniq, cols = np.unique(allh, return_inverse=True); del allh, uniq
    M = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols.astype(np.int32))), shape=(len(sz), int(cols.max()) + 1 if len(cols) else 1))
    del rows, cols
    MT = M.T.tocsr()
    I, Jx, V = [], [], []
    for a in range(0, M.shape[0], 500):
        C = (M[a:a + 500] @ MT).tocoo()
        r = C.row + a; mask = C.col > r
        r, c, inter = r[mask], C.col[mask], C.data[mask]
        j = inter / (sz[r] + sz[c] - inter)
        keep = j >= th_min
        I.append(r[keep]); Jx.append(c[keep]); V.append(j[keep])
        del C
    return np.concatenate(I), np.concatenate(Jx), np.concatenate(V)

def evaluate_sparse(docs, strats=STRATS, ths=(0.7, 0.8), verbose=True, vectors=None):
    n = len(docs)
    case = np.array([d['case'] for d in docs]); is_copy = np.array([d['is_copy'] for d in docs])
    ncase = len(set(case.tolist()))
    copy_idx = np.where(is_copy)[0]
    orig_idx = {d['case']: i for i, d in enumerate(docs) if not d['is_copy']}
    res = {}
    for s in strats:
        I, Jc, V = jaccard_pairs((np.fromiter(x, dtype=np.uint32, count=len(x)) for x in (shingles(release(d['text'], d['spans'], s, d['key'], d['case'])) for d in docs)), min(ths))
        pairJ = {(a, b): v for a, b, v in zip(I.tolist(), Jc.tolist(), V.tolist())}
        r = {}
        for th in ths:
            sel = V >= th
            a, b = I[sel], Jc[sel]
            fm = int((case[a] != case[b]).sum())
            det = float(np.mean([pairJ.get((min(orig_idx[case[i]], i), max(orig_idx[case[i]], i)), 0) >= th for i in copy_idx]))
            comp = components(n, list(zip(a.tolist(), b.tolist())))
            kept = [i for i, c in enumerate(comp) if c == i]
            kc = collections.Counter(case[kept].tolist())
            lost_cases = [c for c in set(case.tolist()) if kc.get(c, 0) == 0]
            if vectors is not None:
                vectors[(s, th)] = dict(lost=np.array([kc.get(c, 0) == 0 for c in sorted(set(case.tolist()))]),
                                        det=np.array([pairJ.get((min(orig_idx[case[i]], i), max(orig_idx[case[i]], i)), 0) >= th for i in copy_idx]))
            detc = float(np.mean([comp[orig_idx[case[i]]] == comp[i] for i in copy_idx]))
            r[str(th)] = dict(false_merge_pairs=fm, cases_lost=len(lost_cases), dup_recall_comp=detc, dup_recall=det,
                              residual_dups=int(sum(1 for i in copy_idx if kc.get(case[i], 0) >= 2)),
                              docs_kept=len(kept), cases_in_false_merge=int(len(set(case[a[case[a] != case[b]]].tolist()) | set(case[b[case[a] != case[b]]].tolist()))))
        r['link'] = linkage(docs, s)
        r['link_person'] = linkage(docs, s, 'PERSON')
        res[s] = r
        if verbose: print(s, {k: r[k] for k in r}, flush=True)
    return res
