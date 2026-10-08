"""G2a: exact-Jaccard pipeline curves (distinct records removed vs copy recall) over thresholds, TAB and Enron."""
import json, pickle
from common import *
from evalcore import evaluate
from evalsparse import evaluate_sparse
from build_corpus import build
THS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
S = ['RAW', 'MASK', 'SUR-DOC', 'SUR-CORPUS', 'HASH-ENTITY']
out = {}
sets = [('TAB gold para', build(), False),
        ('TAB second-source', pickle.load(open('results/tab_real_detected.pkl', 'rb')), False),
        ('Enron second-source', pickle.load(open('results/enron_real_detected.pkl', 'rb')), True)]
for mode in ('plain', 'datatrove'):
    NORM['mode'] = mode
    for name, docs, sparse in sets:
        r = (evaluate_sparse if sparse else evaluate)(docs, strats=S, ths=THS, verbose=False)
        out[f'{name}|{mode}'] = {s: {str(t): {k: r[s][str(t)][k] for k in ('cases_lost', 'dup_recall', 'dup_recall_comp')} for t in THS} for s in S}
        print(name, mode, {s: [(r[s][str(t)]['cases_lost'], round(r[s][str(t)]['dup_recall_comp'], 2)) for t in (0.6, 0.7, 0.8)] for s in S}, flush=True)
        json.dump(out, open('results/v2/curves_exact.json', 'w'), indent=1)
