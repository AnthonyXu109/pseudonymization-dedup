"""E1: effect of release strategy on near-duplicate decisions + cross-document linkage."""
import json, sys, collections
import numpy as np
from common import *
from build_corpus import build

mode = sys.argv[1] if len(sys.argv) > 1 else 'first'
docs = build(mode)
n = len(docs)
case = np.array([d['case'] for d in docs])
is_copy = np.array([d['is_copy'] for d in docs])
ncase = case.max() + 1
same = case[:, None] == case[None, :]
iu = np.triu_indices(n, 1)
pos_mask = same[iu]
copy_cases = sorted(set(case[is_copy]))
THS = [0.5, 0.6, 0.7, 0.8]
out = {'mode': mode, 'n_docs': n, 'n_cases': int(ncase), 'n_copies': int(is_copy.sum()), 'strategies': {}}
percase = {}
for s in STRATS:
    rel = [release(d['text'], d['spans'], s, d['key']) for d in docs]
    J = jaccard_matrix([shingles(r) for r in rel])
    pairJ = J[iu]
    res = {'pos_pair_J_mean': float(pairJ[pos_mask].mean())}
    for th in THS:
        hit = pairJ >= th
        fm_pairs = int((hit & ~pos_mask).sum())
        rec = float((hit & pos_mask).sum() / pos_mask.sum())
        # per-case indicators
        A = (J >= th) & ~same
        fm_case = np.zeros(ncase, bool)
        rows = np.where(A.any(1))[0]
        fm_case[case[rows]] = True
        # dedup: components, keep lowest index per component
        ei = np.argwhere(np.triu(J >= th, 1))
        comp = components(n, ei.tolist())
        kept = sorted({c: i for i, c in reversed(list(enumerate(comp)))}.values())
        kept_cases = collections.Counter(case[kept])
        lost = int(ncase - len(kept_cases))
        resid = int(sum(1 for c in copy_cases if kept_cases.get(c, 0) >= 2))
        res[str(th)] = dict(false_merge_pairs=fm_pairs, dup_recall=rec, cases_in_false_merge=int(fm_case.sum()),
                            cases_lost=lost, residual_dups=resid, docs_kept=len(kept))
        percase[(s, th)] = dict(fm=fm_case, det=np.array([bool(((J[case == c][:, case == c]) >= th).any()) for c in copy_cases]),
                                lostv=np.array([kept_cases.get(c, 0) == 0 for c in range(ncase)]))
    # linkage attacker on originals: link two masked mentions in different cases iff released strings identical
    groups_rel = collections.defaultdict(collections.Counter)  # released string -> Counter(true surface)
    for d, r in zip(docs, rel):
        if d['is_copy']:
            continue
        for m in d['spans']:
            rs = m['surface'] if s == 'RAW' else None
            groups_rel  # placeholder
    # recompute released string per mention
    mention_rel = []
    for d in docs:
        if d['is_copy']:
            continue
        # replicate per-mention replacement
        alias, cnt = {}, collections.Counter()
        for m in d['spans']:
            if s == 'RAW': r = m['surface']
            elif s == 'DROP': r = ''
            elif s == 'MASK': r = '[PII]'
            elif s == 'TYPE': r = f"[{m['type']}]"
            elif s == 'ALIAS-DOC':
                if m['ent'] not in alias:
                    cnt[m['type']] += 1; alias[m['ent']] = f"[{m['type']}_{cnt[m['type']]}]"
                r = alias[m['ent']]
            elif s == 'SUR-DOC': r = surrogate(m['surface'], d['key'])
            else: r = surrogate(m['surface'], 'CORPUS')
            mention_rel.append((d['case'], norm_surface(r) if not r.startswith('[') else r, norm_surface(m['surface']), m['type']))
    # count cross-case pairs: predicted links (same released, nonempty), true links (same true surface)
    def cross_pairs(keyfn):
        g = collections.defaultdict(collections.Counter)
        for mr in mention_rel:
            k = keyfn(mr)
            if k is None: continue
            g[k][mr[0]] += 1
        tot = 0
        for cc in g.values():
            t = sum(cc.values()); tot += (t * t - sum(v * v for v in cc.values())) // 2
        return tot
    true_links = cross_pairs(lambda mr: mr[2])
    pred_links = cross_pairs(lambda mr: mr[1] if mr[1] else None)
    correct = cross_pairs(lambda mr: (mr[1], mr[2]) if mr[1] else None)
    res['link'] = dict(true=true_links, predicted=pred_links, correct=correct,
                       recall=correct / true_links, precision=(correct / pred_links if pred_links else float('nan')))
    out['strategies'][s] = res
    print(s, json.dumps({k: v for k, v in res.items() if k in ('0.7', '0.8', 'link', 'pos_pair_J_mean')}), flush=True)

# paired bootstrap vs RAW over cases (per-case indicators), 2000 resamples
rng = np.random.default_rng(20261001)
B = 2000
bs = {}
for th in THS:
    for s in STRATS:
        if s == 'RAW': continue
        a, b = percase[(s, th)], percase[('RAW', th)]
        idx = rng.integers(0, ncase, (B, ncase))
        dfm = (a['fm'][idx].mean(1) - b['fm'][idx].mean(1))
        dlost = (a['lostv'][idx].mean(1) - b['lostv'][idx].mean(1))
        k = len(copy_cases); idx2 = rng.integers(0, k, (B, k))
        ddet = (a['det'][idx2].mean(1) - b['det'][idx2].mean(1))
        bs[f'{s}@{th}'] = {nm: [float(np.mean(v0)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                          for nm, v, v0 in [('d_frac_cases_false_merge', dfm, a['fm'].mean() - b['fm'].mean()),
                                            ('d_frac_cases_lost', dlost, a['lostv'].mean() - b['lostv'].mean()),
                                            ('d_dup_recall', ddet, a['det'].mean() - b['det'].mean())]}
out['bootstrap_vs_raw'] = bs
json.dump(out, open(f'results/e1_{mode}.json', 'w'), indent=1)
