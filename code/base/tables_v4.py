import json, os, subprocess
L = lambda f: json.load(open(f)) if os.path.exists(f) else None
out = subprocess.run(['python3', 'tables_v3.py'], capture_output=True, text=True).stdout
rec = out.split('% recall table\n')[1].split('% merge table')[0].strip()
cb = L('results/v2/cboot_exact.json')
rows = []
def fmt(raw, s, d, ci, add, res):
    return f"{raw} & {s} & {add}/{res} & {d:+.0f} [{ci[0]:.0f}, {ci[1]:.0f}]"
for name in ('TAB', 'Enron'):
    for mode, lab in (('plain', 'tokens'), ('datatrove', 'norm.')):
        r = cb[f'{name}|{mode}|0.7']
        rows.append(f"{name} & {lab} & exact 0.7 & " + fmt(r['raw'], r['strat'], r['diff'], r['ci'], r['added'], r['restored']) + r' \\')
for name, nm in (('ecthr', 'ECtHR'), ('c4', 'Web')):
    for mode, lab in (('plain', 'tokens'), ('datatrove', 'norm.')):
        P = L(f'results/v2/pipeline_{name}_{mode}.json')
        if not P: rows.append(f"{nm} & {lab} & MinHash & \\multicolumn{{4}}{{c}}{{pending}} \\\\"); continue
        b = P['boot']['MASK']; raw = P['results']['RAW|0|14x8']['removed']; s = P['results']['MASK|0|14x8']['removed']
        rows.append(f"{nm} & {lab} & MinHash & " + fmt(raw, s, b['d_removed'], b['d_removed_ci'], b['added'], b['restored']) + r' \\')
print('% RECALL\n' + rec + '\n% MERGE\n' + '\n'.join(rows))
