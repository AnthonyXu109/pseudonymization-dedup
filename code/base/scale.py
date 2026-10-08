"""F4: scale-up on ECtHR (LexGLUE ecthr_a, 11k) and C4 validation shards (~91k) with detector spans,
two copy kinds (paragraph deletion, realistic second-source), plain and datatrove normalization."""
import sys, os, json, gzip, glob, pickle, collections
from common import *
from evalcore import add_copies_kind
from detector import detect_many
name = sys.argv[1]
if name == 'ecthr':
    import pandas as pd
    texts = ['\n'.join(x) for f in sorted(glob.glob('./ecthr_*.parquet')) for x in pd.read_parquet(f)['text']]
    domain = 'legal'; ncopy = 1000
else:
    texts = [json.loads(l)['text'] for f in sorted(glob.glob('./c4-validation.*.json.gz')) for l in gzip.open(f, 'rt')]
    domain = 'web'; ncopy = 2000
texts = [t for t in texts if len(t.split()) >= 20]
seen, uniq = set(), []
for t in texts:
    k = ' '.join(t.split())
    if k not in seen: seen.add(k); uniq.append(t)
print(name, 'docs', len(texts), 'unique', len(uniq), 'words', sum(len(t.split()) for t in uniq), flush=True)
docs = [dict(key=f'{name}-{i}', case=i, is_copy=False, text=t) for i, t in enumerate(uniq)]
docs += add_copies_kind(docs, ncopy, 'para', domain, 20261001) + add_copies_kind(docs, ncopy, 'real', domain, 20261002)
cache = f'results/{name}_detected.pkl'
if os.path.exists(cache):
    docs = pickle.load(open(cache, 'rb'))
else:
    for d, s in zip(docs, detect_many([d['text'] for d in docs], batch=32)): d['spans'] = s
    pickle.dump(docs, open(cache, 'wb'))
print('detected', flush=True)
