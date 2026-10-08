import json, sys
from common import *
from evalcore import evaluate
from build_corpus import build
from stats import compare
mode = sys.argv[1]; spans_mode = sys.argv[2] if len(sys.argv) > 2 else 'first'
NORM['mode'] = mode
docs = build(spans_mode)
vec = {}
res = evaluate(docs, vectors=vec, verbose=False)
cmp = {f'{s}@{th}': compare(vec, s, th=th) for s in STRATS if s != 'RAW' for th in (0.5, 0.6, 0.7, 0.8)}
for s in STRATS:
    r = res[s]
    print(f"{s:11s}", ' '.join(f"θ{th}: lost={r[str(th)]['cases_lost']:3d} rec={r[str(th)]['dup_recall']:.3f}" for th in (0.5,0.6,0.7,0.8)),
          f"| linkP={r['link']['precision']:.3f} personP={r['link_person']['precision']:.3f} personR={r['link_person']['recall']:.3f}")
for k in ('MASK@0.7','MASK@0.8','SUR-DOC@0.7'):
    print(k, cmp[k])
json.dump(dict(norm=mode, spans=spans_mode, results=res, compare=cmp), open(f'results/v2/e1_gold_{spans_mode}_{mode}.json', 'w'), indent=1)
