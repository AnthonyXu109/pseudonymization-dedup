"""F2: realistic second-source copies (detector re-run on each copy) + Presidio-default hashing, Faker surrogates,
subject-scoped pseudonyms. TAB and Enron."""
import json, pickle, os, sys
from common import *
from evalcore import add_copies_kind, evaluate
from evalsparse import evaluate_sparse
from detector import detect_many
from stats import compare
ALL = STRATS + EXTRA_STRATS
out = {}
for name, src, domain, ncopy, sparse in (('TAB', 'results/tab_detected.pkl', 'legal', 400, False),
                                         ('Enron', 'results/aeslc_detected.pkl', 'email', 1000, True)):
    cache = f'results/{name.lower()}_real_detected.pkl'
    base = [d for d in pickle.load(open(src, 'rb')) if not d['is_copy']]
    if os.path.exists(cache):
        docs = pickle.load(open(cache, 'rb'))
    else:
        cps = add_copies_kind(base, ncopy, 'real', domain, 20261002)
        for d, s in zip(cps, detect_many([d['text'] for d in cps])): d['spans'] = s
        docs = base + cps
        pickle.dump(docs, open(cache, 'wb'))
    for mode in ('plain', 'datatrove'):
        NORM['mode'] = mode
        v = {}
        r = (evaluate_sparse if sparse else evaluate)(docs, strats=ALL, ths=(0.7, 0.75), verbose=False, vectors=v)
        out[f'{name}|{mode}'] = dict(results=r, compare={f'{s}@{t}': compare(v, s, th=t) for s in ALL if s != 'RAW' for t in (0.7, 0.75)})
        for s in ALL:
            print(name, mode, f"{s:12s} rec0.7={r[s]['0.7']['dup_recall']:.3f} rec0.75={r[s]['0.75']['dup_recall']:.3f} removed={r[s]['0.7']['cases_lost']} personP={r[s]['link_person']['precision']:.3f} personR={r[s]['link_person']['recall']:.3f}", flush=True)
NORM['mode'] = 'plain'
json.dump(out, open('results/v2/f2_real.json', 'w'), indent=1, default=float)
