"""Enron (AESLC): detector spans; synthetic copies + natural exact duplicates as a second recall set;
Dolma-style masking (e-mail/phone/IP only); doc-level and paragraph-level dedup; plain and datatrove norm."""
import os, re, json, pickle, sys, collections, numpy as np
from common import *
from evalsparse import evaluate_sparse
from para import run as para_run
from stats import compare
src = './enron_subject_line'
texts = []
for split in ('train', 'dev', 'test'):
    for f in sorted(os.listdir(os.path.join(src, split))):
        texts.append(open(os.path.join(src, split, f), encoding='utf-8', errors='replace').read().split('@subject')[0].strip())
docs = pickle.load(open('results/aeslc_detected.pkl', 'rb'))   # unique originals (case=i) + 1000 synthetic copies
orig = {d['case']: d for d in docs if not d['is_copy']}
key2case = {' '.join(d['text'].split()): c for c, d in orig.items()}
# natural exact duplicates: every extra occurrence of an identical body
seen = collections.Counter(); nat = []
for i, t in enumerate(texts):
    k = ' '.join(t.split())
    if k not in key2case: continue
    seen[k] += 1
    if seen[k] >= 2:
        c = key2case[k]
        nat.append({**orig[c], 'key': f'aeslc-nat-{i}', 'is_copy': True, 'natural': True})
for d in docs: d.setdefault('natural', False)
print('natural duplicate records', len(nat))
docs = docs + nat
# Dolma-style spans: only e-mail, phone, IP via regex
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"); PHONE = re.compile(r"\+?\d[\d ()-]{7,}\d"); IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
for d in docs:
    ms = []
    for rx, t in ((EMAIL, 'EMAIL_ADDRESS'), (PHONE, 'PHONE_NUMBER'), (IP, 'IP_ADDRESS')):
        ms += [dict(start=m.start(), end=m.end(), type=t, surface=m.group(0), ent=(t, m.group(0).lower())) for m in rx.finditer(d['text'])]
    d['dolma_spans'] = resolve(ms)
out = {}
for mode in ('plain', 'datatrove'):
    NORM['mode'] = mode
    vec = {}
    # split copies: evaluate natural vs synthetic recall by running on two views sharing originals
    for view in ('synthetic', 'natural'):
        sub = [d for d in docs if not d['is_copy'] or (d['natural'] == (view == 'natural'))]
        v = {}
        r = evaluate_sparse(sub, verbose=False, vectors=v)
        out[f'{mode}/{view}'] = dict(results=r, compare={f'{s}@0.7': compare(v, s, th=0.7) for s in STRATS if s != 'RAW'})
        for s in STRATS:
            x = r[s]['0.7']; print(mode, view, f"{s:11s} lost={x['cases_lost']:4d} rec={x['dup_recall']:.3f}", flush=True)
        if view == 'synthetic': vec_syn = v
    # Dolma-style masking vs RAW, synthetic view
    sub = [dict(d, spans=d['dolma_spans']) for d in docs if not d['is_copy'] or not d['natural']]
    v = {}
    r = evaluate_sparse(sub, strats=['RAW', 'TYPE', 'SUR-DOC'], verbose=False, vectors=v)
    out[f'{mode}/dolma_types'] = dict(results=r, compare={f'{s}@0.7': compare(v, s, th=0.7) for s in ('TYPE', 'SUR-DOC')})
    print(mode, 'dolma-types', {s: (r[s]['0.7']['cases_lost'], round(r[s]['0.7']['dup_recall'], 3)) for s in r}, flush=True)
NORM['mode'] = 'plain'
out['para_detector'] = para_run([d for d in docs if not d['is_copy']])
out['para_dolma_types'] = para_run([dict(d, spans=d['dolma_spans']) for d in docs if not d['is_copy']], strats=['RAW', 'TYPE'])
out['n_natural'] = len(nat)
json.dump(out, open('results/v2/e4_enron.json', 'w'), indent=1, default=float)
