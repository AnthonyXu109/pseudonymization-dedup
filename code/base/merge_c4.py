import json, os
out = {}
for p in ('a', 'b'):
    f = f'results/v2/scale_c4{p}.json'
    if not os.path.exists(f): continue
    d = json.load(open(f))
    for k, v in d.items():
        if k == 'dose': out['dose'] = v; continue
        o = out.setdefault(k, {'results': {}, 'compare': {}})
        o['results'].update(v['results']); o['compare'].update(v['compare'])
json.dump(out, open('results/v2/scale_c4.json', 'w'), indent=1)
print({k: list(v['results']) if k != 'dose' else v for k, v in out.items()})
