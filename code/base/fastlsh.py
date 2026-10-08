"""Vectorized MinHash + banding LSH (FineWeb config: 5-grams, 112 hashes = 14 bands x 8 rows)."""
import numpy as np, collections
P = (1 << 61) - 1
def signatures(shingle_sets, num_perm=112, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 1 << 31, num_perm, dtype=np.uint64); b = rng.integers(0, 1 << 31, num_perm, dtype=np.uint64)
    sig = np.empty((len(shingle_sets), num_perm), dtype=np.uint64)
    for i, s in enumerate(shingle_sets):
        x = np.fromiter(s, dtype=np.uint64, count=len(s))
        sig[i] = ((x[:, None] * a[None, :] + b[None, :]) % np.uint64(4294967311)).min(0)
    return sig
def lsh_edges(sig, bands=14, rows=8):
    edges = set()
    for bi in range(bands):
        buckets = collections.defaultdict(list)
        block = sig[:, bi * rows:(bi + 1) * rows]
        for i, row in enumerate(map(bytes, block)):
            buckets[row].append(i)
        for ids in buckets.values():
            if len(ids) > 1:
                for x in range(len(ids)):
                    for y in range(x + 1, len(ids)):
                        edges.add((ids[x], ids[y]))
    return edges

def lsh_star_edges(sig, bands=14, rows=8):
    """datatrove-style: documents sharing any band bucket are linked (star per bucket), no pair verification."""
    edges = []
    for bi in range(bands):
        first = {}
        block = sig[:, bi * rows:(bi + 1) * rows]
        for i, row in enumerate(map(bytes, block)):
            j = first.setdefault(row, i)
            if j != i: edges.append((j, i))
    return edges
