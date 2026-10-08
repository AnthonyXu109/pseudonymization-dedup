import json
L = lambda f: json.load(open(f))
g = {m: L(f'results/v2/e1_gold_first_{m}.json') for m in ('plain', 'datatrove')}
dt = {m: L(f'results/v2/e3_tab_detector_{m}.json') for m in ('plain', 'datatrove')}
en = L('results/v2/e4_enron.json')
print('TABLE I (TAB gold, 0.7)')
for s in g['plain']['results']:
    a, b = g['plain']['results'][s], g['datatrove']['results'][s]
    print(f"{s:11s} rem {a['0.7']['cases_lost']:3d}/{b['0.7']['cases_lost']:3d}  rec {a['0.7']['dup_recall']:.3f}/{b['0.7']['dup_recall']:.3f}  personP {a['link_person']['precision']:.3f} personR {a['link_person']['recall']:.3f} allP {a['link']['precision']:.3f}")
print('\nTABLE II MASK vs RAW @0.7 (removed RAW, MASK, only, only_base, p)')
def row(name, res, cmp, s='MASK'):
    c = cmp[f'{s}@0.7']['lost']
    print(f"{name:28s} RAW {res['RAW']['0.7']['cases_lost']:4d} {s} {res[s]['0.7']['cases_lost']:4d}  +{c['only_strategy']}/-{c['only_base']} p={c['mcnemar_p']:.2g}  diff={c['diff']*100:.2f}pp [{c['lo']*100:.2f},{c['hi']*100:.2f}]")
for m in ('plain', 'datatrove'):
    row(f'TAB gold {m}', g[m]['results'], g[m]['compare'])
    row(f'TAB detector {m}', dt[m]['results'], dt[m]['compare'])
    row(f'Enron detector {m}', en[f'{m}/synthetic']['results'], en[f'{m}/synthetic']['compare'])
    row(f'Enron Dolma-scope {m}', en[f'{m}/dolma_types']['results'], en[f'{m}/dolma_types']['compare'], 'TYPE')
print('\nRecall SUR-DOC vs RAW @0.7')
for m in ('plain', 'datatrove'):
    for nm, res in [('TAB gold', g[m]['results']), ('TAB det', dt[m]['results']), ('Enron synth', en[f'{m}/synthetic']['results']), ('Enron natural', en[f'{m}/natural']['results']), ('Enron Dolma synth', en[f'{m}/dolma_types']['results'])]:
        print(m, f"{nm:18s} RAW {res['RAW']['0.7']['dup_recall']:.3f} SUR-DOC {res['SUR-DOC']['0.7']['dup_recall']:.3f} ALIAS {res.get('ALIAS-DOC',{}).get('0.7',{}).get('dup_recall',float('nan')):.3f} SURC {res.get('SUR-CORPUS',{}).get('0.7',{}).get('dup_recall',float('nan')):.3f} MASK {res.get('MASK',{}).get('0.7',{}).get('dup_recall',float('nan')):.3f}")
print('\nEnron natural n', en['n_natural'])
for s in en['plain/natural']['results']:
    r = en['plain/natural']['results'][s]; print('Enron link', s, round(r['link_person']['precision'], 3), round(r['link_person']['recall'], 3))
print('\nthreshold sweep TAB gold plain RAW/MASK:', [(th, g['plain']['results']['RAW'][str(th)]['cases_lost'], g['plain']['results']['MASK'][str(th)]['cases_lost']) for th in (0.5,0.6,0.7,0.8)])
print('threshold sweep TAB gold dt RAW/MASK:', [(th, g['datatrove']['results']['RAW'][str(th)]['cases_lost'], g['datatrove']['results']['MASK'][str(th)]['cases_lost']) for th in (0.5,0.6,0.7,0.8)])
p = L('results/v2/para_tab.json'); print('para TAB', {k: {s: round(v['frac_words_removed']*100, 1) for s, v in p[k].items()} for k in p})
print('para Enron', {s: round(v['frac_words_removed']*100, 2) for s, v in en['para_detector'].items()}, {s: round(v['frac_words_removed']*100, 2) for s, v in en['para_dolma_types'].items()})
