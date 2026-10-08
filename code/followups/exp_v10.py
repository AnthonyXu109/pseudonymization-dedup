"""v10 (PROTOCOL_V10_20261006.md): E6 diff exposure of missed identifiers under FAKER-DOC on TAB."""
import re, os, json, pickle, collections, numpy as np
from common import *
from build_corpus import build
os.makedirs('results/v10', exist_ok=True)
NAME = re.compile(r'(?<![\w])([A-Z][a-z]+)(?![\w])')
NAME_TYPES = {'PERSON', 'ORG', 'LOC'}

def name_tokens(text):
    """(string, start, end) of name-like tokens not at line start or after sentence punctuation."""
    out = []
    for m in NAME.finditer(text):
        pre = text[:m.start()].rstrip(' \t')
        if not pre or pre[-1] in '\n.!?:;"“(': continue
        out.append((m.group(1), m.start(), m.end()))
    return out

def overlaps(s, e, spans):
    return any(sp['start'] < e and s < sp['end'] for sp in spans)

gold = build(); det = pickle.load(open('results/tab_detected.pkl', 'rb'))
assert [g['key'] for g in gold] == [d['key'] for d in det]
rel = [release(d['text'], d['spans'], 'FAKER-DOC', d['key'], d['case']) for d in det]
names_rel = [set(t for t, _, _ in name_tokens(r)) for r in rel]
names_raw = [set(t for t, _, _ in name_tokens(g['text'])) for g in gold]
df = collections.Counter(t for s in names_rel for t in s)

def leaked(i):
    g, d = gold[i], det[i]
    gs = [sp for sp in g['spans'] if sp['type'] in NAME_TYPES]
    return {t for t, s, e in name_tokens(g['text']) if overlaps(s, e, gs) and not overlaps(s, e, d['spans'])}

orig = {g['case']: i for i, g in enumerate(gold) if not g['is_copy']}
pairs = [(orig[g['case']], j) for j, g in enumerate(gold) if g['is_copy']]
res = {}
for filt in ('df<=5', 'none'):
    keep = (lambda t: df[t] <= 5) if filt == 'df<=5' else (lambda t: True)
    agg = collections.Counter()
    for i, j in pairs:
        L = leaked(i) & names_raw[j]                      # leaked strings present in both raw texts
        sur = names_rel[i] - names_raw[i]                 # surrogate strings in the released original
        cand = {t for t in names_rel[i] if keep(t)}
        one, two = cand, {t for t in cand if t in names_rel[j]}
        for nm, F in (('one', one), ('two', two)):
            agg[nm + '_flag'] += len(F); agg[nm + '_tp'] += len(F & L); agg[nm + '_sur'] += len(F & sur)
        agg['leaked'] += len(L); agg['pairs_with_leak'] += bool(L)
        agg['surrogates_in_orig'] += len(sur)
    r = dict(agg)
    for nm in ('one', 'two'):
        r[nm + '_precision'] = agg[nm + '_tp'] / max(1, agg[nm + '_flag'])
        r[nm + '_recall'] = agg[nm + '_tp'] / max(1, agg['leaked'])
    res[filt] = r
    print(filt, json.dumps(r), flush=True)
sh = [shingles(r) for r in rel]
J = [len(sh[i] & sh[j]) / len(sh[i] | sh[j]) for i, j in pairs]
res['released_pair_jaccard'] = dict(median=float(np.median(J)), ge_03=int(sum(x >= 0.3 for x in J)), ge_05=int(sum(x >= 0.5 for x in J)), n=len(J))
print(res['released_pair_jaccard'])
json.dump(res, open('results/v10/e6_diff.json', 'w'), indent=1)
