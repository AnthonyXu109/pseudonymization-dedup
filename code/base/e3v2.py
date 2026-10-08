import json, sys, pickle
from common import *
from evalcore import evaluate
from stats import compare
mode = sys.argv[1]; NORM['mode'] = mode
docs = pickle.load(open('results/tab_detected.pkl', 'rb'))
vec = {}
res = evaluate(docs, vectors=vec, verbose=False)
cmp = {f'{s}@{th}': compare(vec, s, th=th) for s in STRATS if s != 'RAW' for th in (0.5, 0.6, 0.7, 0.8)}
for s in STRATS:
    r = res[s]
    print(f"{s:11s}", ' '.join(f"θ{th}: lost={r[str(th)]['cases_lost']:3d} rec={r[str(th)]['dup_recall']:.3f}" for th in (0.6,0.7,0.8)),
          f"| linkP={r['link']['precision']:.3f} personP={r['link_person']['precision']:.3f} personR={r['link_person']['recall']:.3f}")
print('MASK@0.7', cmp['MASK@0.7']['lost'])
json.dump(dict(norm=mode, results=res, compare=cmp), open(f'results/v2/e3_tab_detector_{mode}.json', 'w'), indent=1)
