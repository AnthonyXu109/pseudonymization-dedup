import sys, pickle, os
from build_scale import build_scale
from detector import detect_many
name, i, CH = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
docs = build_scale(name)
part = docs[i * CH:(i + 1) * CH]
if not part: sys.exit(3)
spans = detect_many([d['text'] for d in part], batch=16)
os.makedirs('results/det', exist_ok=True)
pickle.dump(spans, open(f'results/det/{name}_{i:04d}.pkl', 'wb'))
print(name, i, len(part), flush=True)
