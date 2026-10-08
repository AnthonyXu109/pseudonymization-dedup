import json,collections,numpy as np
from common import *
from build_corpus import build
from datasketch import MinHash, MinHashLSH
out={}
# MinHash-LSH with FineWeb-style parameters: 5-gram, 14 bands x 8 rows (112 perms)
docs = build()
def lsh_run(strat, seed=1):
    lsh = MinHashLSH(num_perm=112, params=(14, 8))
    mh = []
    for i, d in enumerate(docs):
        m = MinHash(num_perm=112, seed=seed)
        for h in shingles(release(d['text'], d['spans'], strat, d['key'])):
            m.update(h.to_bytes(4, 'little'))
        mh.append(m); lsh.insert(i, m)
    edges = set()
    for i, m in enumerate(mh):
        for j in lsh.query(m):
            if j != i: edges.add((min(i, j), max(i, j)))
    case = [d['case'] for d in docs]
    fm = sum(1 for a, b in edges if case[a] != case[b])
    pos = {(i, j) for i in range(len(docs)) for j in [] }
    copies = [i for i, d in enumerate(docs) if d['is_copy']]
    orig_of = {i: docs[i]['case'] for i in copies}
    det = sum(1 for i in copies if (orig_of[i], i) in edges)
    comp = components(len(docs), list(edges))
    kept_cases = collections.Counter(case[i] for i, c in enumerate(comp) if c == i)
    return dict(false_merge_pairs=fm, dup_recall=det / len(copies), cases_lost=1268 - len(kept_cases))
out['lsh'] = {}
for s in STRATS:
    r = [lsh_run(s, seed) for seed in (1, 2, 3)]
    out['lsh'][s] = {k: [x[k] for x in r] for k in r[0]}
    print('LSH', s, out['lsh'][s])
json.dump(out, open('results/lsh.json', 'w'), indent=1)
