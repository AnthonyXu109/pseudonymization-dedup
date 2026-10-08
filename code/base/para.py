"""Dolma-style exact paragraph dedup (first occurrence kept) on released text, originals only."""
import json, re, pickle, collections
from common import *
def run(docs, strats=STRATS, min_words=0):
    out = {}
    for s in strats:
        seen = set(); tot = rem = totw = remw = 0; rem_docs = collections.Counter()
        for d in docs:
            if d['is_copy']: continue
            t = release(d['text'], d['spans'], s, d['key'], d['case'])
            for p in re.split(r"\n+", t):
                w = words(p)
                if len(w) < max(1, min_words): continue
                k = ' '.join(w)
                tot += 1; totw += len(w)
                if k in seen:
                    rem += 1; remw += len(w); rem_docs[d['case']] += 1
                else:
                    seen.add(k)
        out[s] = dict(paragraphs=tot, removed=rem, frac_removed=rem / tot, frac_words_removed=remw / totw, docs_affected=len(rem_docs))
        print(f"{s:11s} removed {rem:6d}/{tot} paras ({rem/tot:.3%}), words {remw/totw:.3%}, docs affected {len(rem_docs)}", flush=True)
    return out
if __name__ == '__main__':
    from build_corpus import build
    res = {'gold': run(build())}
    docs = pickle.load(open('results/tab_detected.pkl', 'rb'))
    print('-- detector'); res['detector'] = run(docs)
    json.dump(res, open('results/para_tab.json', 'w'), indent=1)
