"""Checks whether cached detector spans still line up with the copy texts that build_scale regenerates today.
Second-source copies draw header dates with Faker relative to the current date, so a copy's text can drift."""
import sys, glob, pickle, json
from build_scale import build_scale
out = {}
for nm in sys.argv[1:]:
    docs = build_scale(nm)
    sp = [x for f in sorted(glob.glob(f'results/det/{nm}_*.pkl')) for x in pickle.load(open(f, 'rb'))]
    assert len(sp) == len(docs)
    c = {}
    for d, s in zip(docs, sp):
        k = 'orig' if not d['is_copy'] else d['copy_kind']
        bad = sum(d['text'][m['start']:m['end']] != m['surface'] for m in s)
        e = c.setdefault(k, [0, 0, 0]); e[0] += 1; e[1] += bad > 0; e[2] += bad
    out[nm] = c; print(nm, c, flush=True)
    del docs, sp
json.dump(out, open('results/v2/span_align.json', 'w'), indent=1)
