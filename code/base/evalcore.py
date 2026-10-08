"""Generic evaluation of release strategies on a corpus with near-duplicate copies."""
import collections, random, re, numpy as np
from common import *

def paragraph_copy(text, rng):
    paras, start = [], 0
    for mo in re.finditer(r"\n+", text):
        paras.append((start, mo.end())); start = mo.end()
    paras.append((start, len(text)))
    p = rng.choice([0.05, 0.10, 0.15, 0.20])
    keep = [i == 0 or rng.random() > p for i in range(len(paras))]
    if all(keep) and len(paras) > 2:
        keep[rng.randrange(1, len(paras))] = False
    return ''.join(text[a:b] for (a, b), k in zip(paras, keep) if k)

def add_copies(docs, n_copies, seed=20261001):
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(docs)), n_copies))
    out = list(docs)
    for i in idx:
        out.append({**docs[i], 'key': docs[i]['key'] + '#copy', 'is_copy': True, 'text': paragraph_copy(docs[i]['text'], rng)})
    return out

def mention_release(d, s):
    st, res = new_state(), []
    for idx, m in enumerate(d['spans']):
        r = replace_span(m, idx, s, d['key'], d['case'], st)
        res.append((d['case'], r if r.startswith('[') else norm_surface(r), norm_surface(m['surface']), m['type']))
    return res

def cross_pairs(items, keyfn):
    g = collections.defaultdict(collections.Counter)
    for it in items:
        k = keyfn(it)
        if k is None or k == '': continue
        g[k][it[0]] += 1
    tot = 0
    for cc in g.values():
        t = sum(cc.values()); tot += (t * t - sum(v * v for v in cc.values())) // 2
    return tot

def linkage(docs, s, only_type=None):
    items = [x for d in docs if not d['is_copy'] for x in mention_release(d, s) if only_type is None or x[3] == only_type]
    t = cross_pairs(items, lambda it: it[2]); p = cross_pairs(items, lambda it: it[1])
    c = cross_pairs(items, lambda it: (it[1], it[2]) if it[1] else None)
    return dict(true=t, predicted=p, correct=c, recall=c / t if t else float('nan'), precision=c / p if p else float('nan'))

def evaluate(docs, strats=STRATS, ths=(0.5, 0.6, 0.7, 0.8), verbose=True, vectors=None):
    n = len(docs)
    case = np.array([d['case'] for d in docs]); is_copy = np.array([d['is_copy'] for d in docs])
    cases = sorted(set(case.tolist())); ncase = len(cases)
    copy_idx = np.where(is_copy)[0]
    orig_idx = {d['case']: i for i, d in enumerate(docs) if not d['is_copy']}
    res = {}
    for s in strats:
        J = jaccard_matrix([shingles(release(d['text'], d['spans'], s, d['key'], d['case'])) for d in docs])
        r = {'pos_J_mean': float(np.mean([J[orig_idx[case[i]], i] for i in copy_idx]))}
        for th in ths:
            A = J >= th
            same = case[:, None] == case[None, :]
            fm = int(np.triu(A & ~same, 1).sum())
            det = float(np.mean([A[orig_idx[case[i]], i] for i in copy_idx]))
            ei = np.argwhere(np.triu(A, 1))
            comp = components(n, ei.tolist())
            kept = [i for i, c in enumerate(comp) if c == i]
            kc = collections.Counter(case[kept].tolist())
            if vectors is not None:
                vectors[(s, th)] = dict(lost=np.array([kc.get(c, 0) == 0 for c in cases]),
                                        det=np.array([bool(A[orig_idx[case[i]], i]) for i in copy_idx]))
            detc = float(np.mean([comp[orig_idx[case[i]]] == comp[i] for i in copy_idx]))
            r[str(th)] = dict(false_merge_pairs=fm, cases_lost=ncase - len(kc), dup_recall=det, dup_recall_comp=detc,
                              residual_dups=int(sum(1 for i in copy_idx if kc.get(case[i], 0) >= 2)),
                              docs_kept=len(kept))
            del A, same
        r['link'] = linkage(docs, s)
        r['link_person'] = linkage(docs, s, 'PERSON')
        res[s] = r
        del J
        if verbose:
            print(s, {k: r[k] for k in r if k != 'pos_J_mean'}, flush=True)
    return res

# ---- realistic duplicate generator (F2) ----
_OCR = {'l': '1', 'O': '0', 'rn': 'm', 'e': 'c', 'i': 'l'}
import datetime as _dt
REF_NOW = _dt.datetime(2026, 9, 1, 0, 0, 0)

def realistic_copy(text, rng, domain):
    """A second-source copy: light paragraph loss, a source header and footer that carry their own identifiers,
    quote/whitespace reflow and ~0.5% OCR-like character noise. The detector is re-run on the copy."""
    from faker import Faker
    f = Faker('en_US'); f.seed_instance(rng.randrange(2**31))
    paras = [p for p in re.split(r"\n+", text) if p.strip()]
    keep = [p for i, p in enumerate(paras) if i == 0 or rng.random() > rng.choice([0.0, 0.05, 0.10])]
    body = '\n'.join(keep)
    body = body.replace('“', '"').replace('”', '"').replace('’', "'")
    words_ = body.split(' ')
    for i in range(len(words_)):
        if rng.random() < 0.005:
            w = words_[i]
            for a, b in _OCR.items():
                if a in w:
                    words_[i] = w.replace(a, b, 1); break
    body = ' '.join(words_)
    # dates are drawn up to a fixed reference instant: Faker's default end is "now", which made copies drift
    # between runs (fixed in v6)
    date = f.date(pattern='%d %B %Y', end_datetime=REF_NOW)
    if domain == 'legal':
        head = f"Source: HUDOC database. Retrieved on {date} by {f.name()}."
        foot = f"Downloaded from the Court's website on {date}."
    elif domain == 'email':
        head = f"-----Original Message-----\nFrom: {f.name()}\nSent: {f.date(pattern='%A, %B %d, %Y', end_datetime=REF_NOW)} {f.time(pattern='%I:%M %p', end_datetime=REF_NOW)}\nTo: {f.name()}\nSubject: FW:"
        foot = f"{f.name()}\n{f.phone_number()}"
    else:
        head = f"Posted by {f.name()} on {date}"
        foot = f"Copyright {f.year()} {f.company()}. All rights reserved."
    return head + '\n' + body + '\n' + foot

def add_copies_kind(docs, n_copies, kind, domain, seed):
    rng = random.Random(seed)
    idx = sorted(rng.sample([i for i, d in enumerate(docs) if not d['is_copy']], n_copies))
    out = []
    for i in idx:
        t = paragraph_copy(docs[i]['text'], rng) if kind == 'para' else realistic_copy(docs[i]['text'], rng, domain)
        out.append({**docs[i], 'key': docs[i]['key'] + f'#{kind}', 'is_copy': True, 'copy_kind': kind, 'text': t, 'spans': None})
    return out
