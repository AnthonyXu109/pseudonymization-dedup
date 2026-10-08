import json, os
L = lambda f: json.load(open(f)) if os.path.exists(f) else None
f2 = L('results/v2/f2_real.json'); ex = L('results/v2/extra_cells.json'); e1 = L('results/v2/e1_gold_first_plain.json')
en = L('results/v2/e4_enron.json'); ec = L('results/v2/scale_ecthr.json'); c4 = L('results/v2/scale_c4.json'); sw = L('results/v2/sweep.json')
e3 = {m: L(f'results/v2/e3_tab_detector_{m}.json') for m in ('plain', 'datatrove')}
rows = ['RAW', 'DROP', 'MASK', 'TYPE', 'ALIAS-DOC', 'SUR-DOC', 'HASH-ENTITY', 'FAKER-DOC', 'SUR-CORPUS', 'SUR-SUBJECT']
dag = {'SUR-DOC', 'HASH-ENTITY', 'FAKER-DOC'}
def g(res, s, key='dup_recall', th='0.7'):
    try: return res[s][th][key]
    except Exception: return None
cols = [
    lambda s: g(e1['results'], s) if s in e1['results'] else (ex['TAB gold para'][s]['rec'] if s in ex['TAB gold para'] else None),
    lambda s: g(f2['TAB|plain']['results'], s),
    lambda s: g(en['plain/natural']['results'], s) if s in en['plain/natural']['results'] else (ex['Enron natural'][s]['rec'] if ex and 'Enron natural' in ex and s in ex['Enron natural'] else None),
    lambda s: g(f2['Enron|plain']['results'], s),
    lambda s: g(ec['real|plain']['results'], s) if ec and 'real|plain' in ec else None,
    lambda s: g(c4['real|plain']['results'], s) if c4 and 'real|plain' in c4 else None,
]
fmt = lambda v: '--' if v is None else f'{v:.2f}' if v >= 0.095 or v == 0 else f'{v:.3f}'
print('% recall table')
for s in rows:
    DG = '$^\\dagger$' if s in dag else ''
    print('\\st{' + s + '}' + DG + ' & ' + ' & '.join(fmt(c(s)) for c in cols) + ' \\\\')
print('% merge table')
def mrow(label, dedup, th, raw, mask, p):
    return f"{label} & {dedup} & {th} & {raw} & {mask} & {p}\\\\"
def pf(p): return '$<$0.0001' if p < 0.0001 else f'{p:.4f}' if p < 0.001 else f'{p:.3f}' if p < 0.01 else f'{p:.2f}'
for mode, lab in (('plain', 'tokens'), ('datatrove', 'normalized')):
    r = e3[mode]['results']; c = e3[mode]['compare']
    print(mrow('TAB', lab, '0.7', r['RAW']['0.7']['cases_lost'], r['MASK']['0.7']['cases_lost'], pf(c['MASK@0.7']['lost']['mcnemar_p'])))
    k = f'TAB detector|{mode}|n5'; o = sw[k]['0.75']
    print(mrow('TAB', lab, '0.75', o['raw_removed'], o['mask_removed'], pf(o['mask_p'])))
for mode, lab in (('plain', 'tokens'), ('datatrove', 'normalized')):
    r = en[f'{mode}/synthetic']['results']; c = en[f'{mode}/synthetic']['compare']
    print(mrow('Enron', lab, '0.7', r['RAW']['0.7']['cases_lost'], r['MASK']['0.7']['cases_lost'], pf(c['MASK@0.7']['lost']['mcnemar_p'])))
    o = sw[f'Enron detector|{mode}|n5']['0.75']
    print(mrow('Enron', lab, '0.75', o['raw_removed'], o['mask_removed'], pf(o['mask_p'])))
for nm, d in (('ECtHR', ec), ('Web', c4)):
    if not d: continue
    for mode, lab in (('plain', 'tokens'), ('datatrove', 'normalized')):
        k = f'real|{mode}'
        if k not in d or 'MASK' not in d[k]['results']: continue
        r = d[k]['results']; c = d[k]['compare']
        print(mrow(nm, lab, 'LSH', r['RAW']['0.7']['cases_lost'], r['MASK']['0.7']['cases_lost'], pf(c['MASK@0.7']['lost']['mcnemar_p'])))
print('% extra')
for nm, d in (('ECtHR', ec), ('Web', c4)):
    if d and 'dose' in d: print(nm, 'dose', d['dose'])
