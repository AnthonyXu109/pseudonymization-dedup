"""Shared code: TAB loading, release strategies, shingling, dedup, linkage metrics."""
import json, re, hashlib, random, zlib, collections
import numpy as np, scipy.sparse as sp

TAB = './tab'
MASKED = ('DIRECT', 'QUASI')
STRATS = ['RAW', 'DROP', 'MASK', 'TYPE', 'ALIAS-DOC', 'SUR-DOC', 'SUR-CORPUS']
EXTRA_STRATS = ['HASH-ENTITY', 'FAKER-DOC', 'SUR-SUBJECT']
SALT = 'xcur-2026-secret-salt'
TOK = re.compile(r"\[[A-Z_0-9]+\]|\w+")

def load_tab():
    D = []
    for f in ['train', 'dev', 'test']:
        D += json.load(open(f'{TAB}/echr_{f}.json'))
    return D

def resolve(ms):
    """longest span first, then earliest start; drop overlaps."""
    ms = sorted(ms, key=lambda m: (-(m['end'] - m['start']), m['start']))
    acc = []
    for m in ms:
        if any(m['start'] < o['end'] and o['start'] < m['end'] for o in acc):
            continue
        acc.append(m)
    return sorted(acc, key=lambda m: m['start'])

def gold_spans(doc, mode='first', types=None, idtypes=MASKED):
    """mode='first': first annotator (sorted); 'union': union over annotators."""
    anns = sorted(doc['annotations'])
    use = anns[:1] if mode == 'first' else anns
    ms = []
    for a in use:
        for m in doc['annotations'][a]['entity_mentions']:
            if m['identifier_type'] not in idtypes:
                continue
            if types and m['entity_type'] not in types:
                continue
            ms.append(dict(start=m['start_offset'], end=m['end_offset'], type=m['entity_type'],
                           ent=(a, m['entity_id']), surface=doc['text'][m['start_offset']:m['end_offset']]))
    return resolve(ms)

def norm_surface(s):
    return ' '.join(TOK.findall(s.lower()))

def _h(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'big')

import hmac, unicodedata
def keyed_token(tok, scope_key):
    """Keyed pseudonym token: 'zq' + 10 base-26 letters of HMAC-SHA256(SALT|scope, token).
    Injective with overwhelming probability, letters only (survives number normalization),
    and disjoint from ordinary words via the reserved prefix."""
    d = hmac.new(f'{SALT}|{scope_key}'.encode(), tok.lower().encode(), 'sha256').digest()
    n = int.from_bytes(d[:8], 'big'); out = []
    for _ in range(10):
        n, r = divmod(n, 26); out.append(chr(97 + r))
    return 'zq' + ''.join(out)

def surrogate(surface, scope_key):
    # token-level keyed mapping; scope_key = doc id (SUR-DOC) or 'CORPUS' (SUR-CORPUS)
    return re.sub(r"\w+", lambda mo: keyed_token(mo.group(0), scope_key), surface)

def replace_span(m, idx, strat, doc_key, subject, st):
    """Replacement string for one span; st holds per-document state (alias counters, faker memo)."""
    if strat == 'RAW': return m['surface']
    if strat == 'DROP': return ''
    if strat == 'MASK': return '[PII]'
    if strat == 'TYPE': return f"[{m['type']}]"
    if strat == 'ALIAS-DOC':
        k = m['ent']
        if k not in st['alias']:
            st['cnt'][m['type']] += 1
            st['alias'][k] = f"[{m['type']}_{st['cnt'][m['type']]}]"
        return st['alias'][k]
    if strat == 'SUR-DOC': return surrogate(m['surface'], doc_key)
    if strat == 'SUR-CORPUS': return surrogate(m['surface'], 'CORPUS')
    if strat == 'SUR-SUBJECT': return surrogate(m['surface'], f'SUBJECT|{subject}')
    if strat == 'HASH-ENTITY':
        salt = hashlib.sha256(f'{SALT}|{doc_key}|{idx}'.encode()).digest()
        return hashlib.sha256(m['surface'].encode() + salt).hexdigest()
    if strat == 'FAKER-DOC':
        k = m['ent']
        if k not in st['faked']:
            st['faked'][k] = fake_value(m['type'], m['surface'], _h(SALT, doc_key, k))
        return st['faked'][k]
    raise ValueError(strat)

def new_state():
    return {'alias': {}, 'cnt': collections.Counter(), 'faked': {}}

def release(text, spans, strat, doc_key, subject=None):
    if strat == 'RAW':
        return text
    out, cur, st = [], 0, new_state()
    for idx, m in enumerate(spans):
        out.append(text[cur:m['start']])
        out.append(replace_span(m, idx, strat, doc_key, subject, st))
        cur = m['end']
    out.append(text[cur:])
    return ''.join(out)

NORM = {'mode': 'plain'}   # 'plain' or 'datatrove'
NUM_RE = re.compile(r"\d+([.,]\d+)?")
PUNCT_RE = re.compile(r"[^\w\s]|_")
def datatrove_norm(s):
    """Approximation of datatrove simplify_text defaults: lowercase, numbers->0,
    punctuation->space, strip diacritics, collapse whitespace."""
    s = s.lower()
    s = NUM_RE.sub('0', s)
    s = PUNCT_RE.sub(' ', s)
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    return s.split()

def words(s):
    if NORM['mode'] == 'datatrove':
        return datatrove_norm(s)
    return [w if w.startswith('[') else w.lower() for w in TOK.findall(s)]

NGRAM = {'n': 5}
def shingles(s, n=None):
    n = n or NGRAM['n']
    w = words(s)
    return {zlib.crc32(' '.join(w[i:i + n]).encode()) for i in range(max(1, len(w) - n + 1))}

def jaccard_matrix(sets):
    vocab, rows, cols = {}, [], []
    for i, s in enumerate(sets):
        for h in s:
            rows.append(i); cols.append(vocab.setdefault(h, len(vocab)))
    M = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(sets), len(vocab)))
    M.sum_duplicates(); M.data[:] = 1
    inter = (M @ M.T).toarray()
    sz = np.asarray(M.sum(1)).ravel()
    J = inter / (sz[:, None] + sz[None, :] - inter)
    np.fill_diagonal(J, 0)
    return J

def components(n, edges):
    p = list(range(n))
    def f(x):
        while p[x] != x:
            p[x] = p[p[x]]; x = p[x]
        return x
    for a, b in edges:
        ra, rb = f(a), f(b)
        if ra != rb:
            p[max(ra, rb)] = min(ra, rb)
    return [f(i) for i in range(n)]

_POOLS = {}
def _pool(kind):
    """Pre-generated Faker value pools (seeded), so per-entity draws are cheap and reproducible."""
    if kind not in _POOLS:
        from faker import Faker
        f = Faker('en_US'); f.seed_instance(20261001)
        n = 20000
        gen = {'name': f.name, 'company': f.company, 'city': f.city, 'date': lambda: f.date(pattern='%d %B %Y'),
               'dow': f.day_of_week, 'country': f.country, 'email': f.email, 'phone': f.phone_number, 'ip': f.ipv4, 'word': f.word}[kind]
        _POOLS[kind] = [gen() for _ in range(n if kind not in ('dow',) else 50)]
    return _POOLS[kind]

def fake_value(etype, surface, seed):
    """Realistic per-document surrogate (Faker pools), consistent within a document; token count may differ."""
    pick = lambda k: (lambda p: p[seed % len(p)])(_pool(k))
    if etype == 'PERSON': return pick('name')
    if etype == 'ORG': return pick('company')
    if etype == 'LOC': return pick('city')
    if etype == 'DATETIME': return pick('date') if any(c.isdigit() for c in surface) else pick('dow')
    if etype == 'CODE':
        r = random.Random(seed)
        return re.sub(r'\d', lambda _: str(r.randrange(10)), surface) if any(c.isdigit() for c in surface) else 'XQ-%04d' % r.randrange(10000)
    if etype == 'QUANTITY': return str(1 + seed % 99999)
    if etype == 'DEM': return pick('country')
    if etype == 'EMAIL_ADDRESS': return pick('email')
    if etype == 'PHONE_NUMBER': return pick('phone')
    if etype == 'IP_ADDRESS': return pick('ip')
    return pick('word')
