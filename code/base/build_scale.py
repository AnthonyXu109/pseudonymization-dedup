import json, gzip, glob
from common import *
from evalcore import add_copies_kind
def build_scale(name):
    if name == 'ecthr':
        import pandas as pd
        texts = ['\n'.join(x) for f in sorted(glob.glob('./ecthr_*.parquet')) for x in pd.read_parquet(f)['text']]
        domain, ncopy = 'legal', 1000
    else:
        texts = [json.loads(l)['text'] for f in sorted(glob.glob('./c4-validation.*.json.gz')) for l in gzip.open(f, 'rt')]
        domain, ncopy = 'web', 2000
    texts = [t for t in texts if len(t.split()) >= 20]
    seen, uniq = set(), []
    for t in texts:
        k = ' '.join(t.split())
        if k not in seen: seen.add(k); uniq.append(t)
    docs = [dict(key=f'{name}-{i}', case=i, is_copy=False, text=t) for i, t in enumerate(uniq)]
    docs += add_copies_kind(docs, ncopy, 'para', domain, 20261001) + add_copies_kind(docs, ncopy, 'real', domain, 20261002)
    return docs
