import json, sys, collections, pickle, numpy as np
from common import *
from fastlsh import signatures, lsh_edges
def run(docs, strats=STRATS, seeds=range(20)):
    case = [d['case'] for d in docs]; ncase = len(set(case))
    orig = {d['case']: i for i, d in enumerate(docs) if not d['is_copy']}
    copies = [i for i, d in enumerate(docs) if d['is_copy']]
    out = {}
    for s in strats:
        sets = [shingles(release(d['text'], d['spans'], s, d['key'])) for d in docs]
        rs = []
        for seed in seeds:
            E = lsh_edges(signatures(sets, seed=seed))
            fm = sum(1 for a, b in E if case[a] != case[b])
            det = np.mean([(min(orig[case[i]], i), max(orig[case[i]], i)) in E for i in copies])
            comp = components(len(docs), list(E))
            kc = collections.Counter(case[i] for i, c in enumerate(comp) if c == i)
            rs.append((fm, det, ncase - len(kc)))
        a = np.array(rs, float)
        out[s] = dict(false_merge_pairs=a[:, 0].tolist(), dup_recall=a[:, 1].tolist(), cases_lost=a[:, 2].tolist())
        print(s, 'fm %.1f±%.1f  rec %.3f  lost %.1f±%.1f' % (a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 2].mean(), a[:, 2].std()), flush=True)
    return out
if __name__ == '__main__':
    from build_corpus import build
    docs = build()
    json.dump(run(docs), open('results/lsh_tab_gold.json', 'w'), indent=1)
