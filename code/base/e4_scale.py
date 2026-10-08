"""E4: scale-up with detector spans on ECtHR (LexGLUE ecthr_a) or Enron (AESLC)."""
import sys, json, pickle, os
from common import *
from evalcore import add_copies
from evalsparse import evaluate_sparse
from detector import detect_many
from para import run as para_run
name = sys.argv[1]; src = sys.argv[2]
if name == 'ecthr':
    import pandas as pd
    frames = [pd.read_parquet(os.path.join(src, f)) for f in sorted(os.listdir(src)) if f.startswith(name) and f.endswith('.parquet')]
    df = pd.concat(frames, ignore_index=True)
    texts = ['\n'.join(x) for x in df['text']]
else:
    texts = []
    for split in ('train', 'dev', 'test'):
        dd = os.path.join(src, split)
        for f in sorted(os.listdir(dd)):
            texts.append(open(os.path.join(dd, f), encoding='utf-8', errors='replace').read().split('@subject')[0].strip())
texts = [t for t in texts if len(t.split()) >= 20]
# exact-duplicate originals are collapsed first so that 'cases' are distinct texts
seen, uniq = set(), []
for t in texts:
    k = ' '.join(t.split())
    if k not in seen: seen.add(k); uniq.append(t)
print(name, 'docs', len(texts), 'unique', len(uniq), flush=True)
docs = [dict(key=f'{name}-{i}', case=i, is_copy=False, text=t) for i, t in enumerate(uniq)]
docs = add_copies(docs, min(1000, len(docs) // 10))
cache = f'results/{name}_detected.pkl'
if os.path.exists(cache):
    docs = pickle.load(open(cache, 'rb'))
else:
    for d, s in zip(docs, detect_many([d['text'] for d in docs], batch=32)): d['spans'] = s
    pickle.dump(docs, open(cache, 'wb'))
print('detected', flush=True)
res = evaluate_sparse(docs)
para = para_run(docs)
json.dump(dict(name=name, n_unique=len(uniq), n_docs=len(docs), results=res, para=para), open(f'results/e4_{name}.json', 'w'), indent=1)
