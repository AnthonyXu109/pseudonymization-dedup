import numpy as np
from scipy.stats import binomtest
def compare(vec, s, base='RAW', th=0.7, B=2000, seed=20261001):
    a, b = vec[(s, th)], vec[(base, th)]
    out = {}
    rng = np.random.default_rng(seed)
    for k in ('lost', 'det'):
        x, y = a[k].astype(int), b[k].astype(int)
        n = len(x); idx = rng.integers(0, n, (B, n))
        d = x[idx].mean(1) - y[idx].mean(1)
        n10 = int(((x == 1) & (y == 0)).sum()); n01 = int(((x == 0) & (y == 1)).sum())
        p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        out[k] = dict(diff=float(x.mean() - y.mean()), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                      only_strategy=n10, only_base=n01, mcnemar_p=float(p))
    return out
