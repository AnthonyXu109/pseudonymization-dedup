"""v6: second-source copies are now generated with a fixed reference date (evalcore.REF_NOW). Re-run the detector on
those copies only and patch the cached span chunks in results/det (the v5 cache is kept in results/det_v5).
Also records a SHA-256 of every regenerated copy so later runs can verify that the texts have not drifted."""
import sys, glob, pickle, hashlib, json, os
from build_scale import build_scale
from detector import detect_many
name, CH = sys.argv[1], int(sys.argv[2])
docs = build_scale(name)
idx = [i for i, d in enumerate(docs) if d['is_copy'] and d.get('copy_kind') == 'real']
spans = detect_many([docs[i]['text'] for i in idx], batch=16)
files = sorted(glob.glob(f'results/det/{name}_*.pkl')); chunks = [pickle.load(open(f, 'rb')) for f in files]
flat = [x for c in chunks for x in c]; assert len(flat) == len(docs)
for i, s in zip(idx, spans): flat[i] = s
k = 0
for f, c in zip(files, chunks):
    pickle.dump(flat[k:k + len(c)], open(f, 'wb')); k += len(c)
man = 'results/v2/copies_manifest.json'
M = json.load(open(man)) if os.path.exists(man) else {}
M[name] = {docs[i]['key']: hashlib.sha256(docs[i]['text'].encode()).hexdigest() for i in idx}
M[name + '_all_sha256'] = hashlib.sha256(''.join(d['text'] for d in docs).encode()).hexdigest()
json.dump(M, open(man, 'w'))
print(name, 'redetected', len(idx), flush=True)
