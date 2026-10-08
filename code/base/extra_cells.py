import json, pickle, os, collections
from common import *
from evalcore import evaluate
from evalsparse import evaluate_sparse
from build_corpus import build
out = {}
NORM['mode'] = 'plain'
r = evaluate(build(), strats=['RAW'] + EXTRA_STRATS, ths=(0.7,), verbose=False)
out['TAB gold para'] = {s: dict(rec=r[s]['0.7']['dup_recall'], removed=r[s]['0.7']['cases_lost'], lp=r[s]['link_person']) for s in r}
print(out['TAB gold para'].keys(), {s: v['rec'] for s, v in out['TAB gold para'].items()}, flush=True)
en = pickle.load(open('results/aeslc_detected.pkl', 'rb'))
src = './enron_subject_line'
texts = [open(os.path.join(src, sp, f), encoding='utf-8', errors='replace').read().split('@subject')[0].strip()
         for sp in ('train', 'dev', 'test') for f in sorted(os.listdir(os.path.join(src, sp)))]
orig = {d['case']: d for d in en if not d['is_copy']}
k2c = {' '.join(d['text'].split()): c for c, d in orig.items()}
seen = collections.Counter(); nat = []
for i, t in enumerate(texts):
    k = ' '.join(t.split())
    if k in k2c:
        seen[k] += 1
        if seen[k] >= 2: nat.append({**orig[k2c[k]], 'key': f'aeslc-nat-{i}', 'is_copy': True, 'natural': True})
r = evaluate_sparse(list(orig.values()) + nat, strats=['RAW'] + EXTRA_STRATS, ths=(0.7,), verbose=False)
out['Enron natural'] = {s: dict(rec=r[s]['0.7']['dup_recall'], removed=r[s]['0.7']['cases_lost']) for s in r}
print({s: v['rec'] for s, v in out['Enron natural'].items()}, flush=True)
json.dump(out, open('results/v2/extra_cells.json', 'w'), indent=1, default=float)
