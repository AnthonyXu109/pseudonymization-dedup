"""Build corpus: 1,268 TAB cases + near-duplicate copies (paragraph deletion) for a seeded sample."""
import json, random, re
from common import *

SEED = 20261001
N_COPIES = 400

def make_copy(doc, spans, rng):
    text = doc['text']
    # paragraph boundaries
    paras, start = [], 0
    for mo in re.finditer(r"\n+", text):
        paras.append((start, mo.end())); start = mo.end()
    paras.append((start, len(text)))
    p = rng.choice([0.05, 0.10, 0.15, 0.20])
    keep = [i == 0 or rng.random() > p for i in range(len(paras))]
    if all(keep) and len(paras) > 2:
        keep[rng.randrange(1, len(paras))] = False
    new, newspans, shift = [], [], 0
    pos = 0
    for (a, b), k in zip(paras, keep):
        if k:
            new.append(text[a:b])
            for m in spans:
                if a <= m['start'] and m['end'] <= b:
                    newspans.append({**m, 'start': m['start'] - a + pos, 'end': m['end'] - a + pos})
            pos += b - a
    return ''.join(new), newspans, p

def build(mode='first'):
    D = load_tab()
    rng = random.Random(SEED)
    docs = []
    for i, d in enumerate(D):
        docs.append(dict(key=d['doc_id'], case=i, is_copy=False, text=d['text'], spans=gold_spans(d, mode),
                         country=d['meta'].get('countries'), branch=d['meta'].get('legal_branch'), year=d['meta'].get('year')))
    idx = sorted(rng.sample(range(len(D)), N_COPIES))
    for i in idx:
        t, s, p = make_copy(D[i], docs[i]['spans'], rng)
        docs.append(dict(key=D[i]['doc_id'] + '#copy', case=i, is_copy=True, text=t, spans=s, p_del=p,
                         country=docs[i]['country'], branch=docs[i]['branch'], year=docs[i]['year']))
    return docs

if __name__ == '__main__':
    docs = build()
    print(len(docs), sum(d['is_copy'] for d in docs))
