import json, pickle, sys
from common import *
from evalcore import evaluate
from evalsparse import evaluate_sparse
from build_corpus import build
from stats import compare
THS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9)
S = ['RAW', 'MASK', 'SUR-DOC', 'SUR-CORPUS', 'ALIAS-DOC']
out = {}
tabg, tabd = build(), pickle.load(open('results/tab_detected.pkl', 'rb'))
for mode in ('plain', 'datatrove'):
    NORM['mode'] = mode
    for n in ((3, 5, 7) if mode == 'plain' else (5,)):
        NGRAM['n'] = n
        for name, docs in (('TAB gold', tabg), ('TAB detector', tabd)):
            v = {}; r = evaluate(docs, strats=S, ths=THS, verbose=False, vectors=v)
            key = f'{name}|{mode}|n{n}'
            out[key] = {str(t): dict(raw_removed=r['RAW'][str(t)]['cases_lost'], mask_removed=r['MASK'][str(t)]['cases_lost'],
                                     mask_p=compare(v, 'MASK', th=t)['lost']['mcnemar_p'],
                                     rec_raw=r['RAW'][str(t)]['dup_recall'], rec_surdoc=r['SUR-DOC'][str(t)]['dup_recall'],
                                     rec_alias=r['ALIAS-DOC'][str(t)]['dup_recall'], rec_surc=r['SUR-CORPUS'][str(t)]['dup_recall']) for t in THS}
            print(key, {t: (o['raw_removed'], o['mask_removed'], round(o['mask_p'], 3), round(o['rec_raw'], 2), round(o['rec_surdoc'], 3)) for t, o in out[key].items()}, flush=True)
NGRAM['n'] = 5
en = pickle.load(open('results/aeslc_detected.pkl', 'rb'))
for mode in ('plain', 'datatrove'):
    NORM['mode'] = mode
    v = {}; r = evaluate_sparse(en, strats=['RAW', 'MASK', 'SUR-DOC'], ths=THS, verbose=False, vectors=v)
    key = f'Enron detector|{mode}|n5'
    out[key] = {str(t): dict(raw_removed=r['RAW'][str(t)]['cases_lost'], mask_removed=r['MASK'][str(t)]['cases_lost'],
                             mask_p=compare(v, 'MASK', th=t)['lost']['mcnemar_p'], rec_raw=r['RAW'][str(t)]['dup_recall'],
                             rec_surdoc=r['SUR-DOC'][str(t)]['dup_recall']) for t in THS}
    print(key, {t: (o['raw_removed'], o['mask_removed'], round(o['mask_p'], 3), round(o['rec_raw'], 2), round(o['rec_surdoc'], 3)) for t, o in out[key].items()}, flush=True)
json.dump(out, open('results/v2/sweep.json', 'w'), indent=1)
