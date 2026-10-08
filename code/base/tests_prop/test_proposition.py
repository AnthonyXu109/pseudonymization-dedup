"""Micro-cases for Proposition 1 (run: python3 tests_prop/test_proposition.py from xcur/)."""
import sys, re; sys.path.insert(0, '.')
from common import *
from dose import jac
def spans_for(text, words, only_first=False):
    out = []
    for w in words:
        for k, mo in enumerate(re.finditer(r'\b%s\b' % re.escape(w), text)):
            if only_first and k > 0: break
            out.append(dict(start=mo.start(), end=mo.end(), type='PERSON', surface=mo.group(0), ent=('t', w.lower())))
    return resolve(out)
def case(name, a, b, sa, sb, expect_exact):
    A, B = shingles(a), shingles(b); S = A & B; J = jac(A, B)
    ra, rb = release(a, sa, 'SUR-DOC', 'docA'), release(b, sb, 'SUR-DOC', 'docB')
    obs = jac(shingles(ra), shingles(rb))
    # m_sh by literal definition: common shingle types containing a marked token at some occurrence
    def marked_types(t, sp):
        toks = [(mo.start(), mo.end(), mo.group(0).lower()) for mo in TOK.finditer(t)]
        mk = [any(s['start'] < e and st < s['end'] for s in sp) for st, e, _ in toks]
        return {zlib.crc32(' '.join(x[2] for x in toks[i:i + 5]).encode()) for i in range(len(toks) - 4) if any(mk[i:i + 5])}
    m = len(S & (marked_types(a, sa) | marked_types(b, sb))) / len(S)
    pred = (1 - m) * J / (1 + m * J)
    ok = abs(obs - pred) < 1e-12
    print(f"{name:55s} J={J:.3f} m_sh={m:.3f} pred={pred:.3f} obs={obs:.3f} exact={ok}")
    assert ok == expect_exact, name
t = 'Alice went to court today Alice went to court today'
case('exact copy, every occurrence marked', t, t, spans_for(t, ['Alice']), spans_for(t, ['Alice']), True)
case('exact copy, only first occurrence marked (violates C1)', t, t, spans_for(t, ['Alice'], True), spans_for(t, ['Alice'], True), False)
u = 'the applicant Alice Smith lodged the application on the same day as the others did'
v = u + ' and Bob Jones was her lawyer in the proceedings'
case('near-duplicate, identifiers in shared and unshared parts', u, v, spans_for(u, ['Alice', 'Smith']), spans_for(v, ['Alice', 'Smith', 'Bob', 'Jones']), True)
w = 'Alice lodged a complaint and ALICE was heard by the court on the same day'
case('case variants of one name, all marked (C1 after lowercasing)', w, w, spans_for(w, ['Alice', 'ALICE']), spans_for(w, ['Alice', 'ALICE']), True)
case('name marked in one copy only (different marking across copies)', u, u, spans_for(u, ['Alice']), [], True)
print('all micro-cases behave as stated')
