import json, pickle, numpy as np
from common import *
from evalcore import *
from detector import detect_many
D = load_tab()
docs = [dict(key=d['doc_id'], case=i, is_copy=False, text=d['text']) for i, d in enumerate(D)]
docs = add_copies(docs, 400)
spans = detect_many([d['text'] for d in docs])
for d, s in zip(docs, spans): d['spans'] = s
pickle.dump(docs, open('results/tab_detected.pkl', 'wb'))
# detector quality vs gold (char-level, originals, DIRECT+QUASI of first annotator)
tp = fn = fp = 0; direct_miss = direct_tot = 0
for i, d in enumerate(D):
    g = np.zeros(len(d['text']), bool); p = np.zeros(len(d['text']), bool)
    for m in gold_spans(d): g[m['start']:m['end']] = True
    for m in docs[i]['spans']: p[m['start']:m['end']] = True
    tp += int((g & p).sum()); fn += int((g & ~p).sum()); fp += int((~g & p).sum())
    for m in gold_spans(d, idtypes=('DIRECT',)):
        direct_tot += 1; direct_miss += int(not p[m['start']:m['end']].all())
qual = dict(char_recall=tp / (tp + fn), char_precision=tp / (tp + fp), direct_mentions=direct_tot,
            direct_mentions_not_fully_masked=direct_miss)
print(qual, flush=True)
res = evaluate(docs)
json.dump(dict(detector='spaCy en_core_web_lg 3.8 + regex', quality=qual, results=res), open('results/e3_tab.json', 'w'), indent=1)
