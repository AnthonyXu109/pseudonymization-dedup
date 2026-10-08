import json, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows = json.load(open('results/v2/dose_pairs.json'))
plt.rcParams.update({'font.size': 7.5, 'font.family': 'serif', 'axes.linewidth': 0.6})
fig, ax = plt.subplots(figsize=(3.4, 2.35))
cols = {'TAB gold': '#2a78d6', 'TAB detector': '#1baf7a', 'Enron detector': '#eb6834', 'Enron natural': '#e87ba4', 'Enron Dolma-like': '#8a8a8a'}
rng = np.random.default_rng(0)
for s, c in cols.items():
    R = [r for r in rows if r['setting'] == s]
    if len(R) > 500: R = [R[i] for i in rng.choice(len(R), 500, replace=False)]
    ax.scatter([r['m'] for r in R], [r['J_surdoc'] for r in R], s=4, color=c, alpha=0.55, linewidths=0, label=s, zorder=3)
m = np.linspace(0, 1, 200)
ax.plot(m, (1 - m) / (1 + m), color='#222222', lw=1.0, zorder=4, label='prediction for exact copies')
ax.axhline(0.7, color='#999999', lw=0.7, ls='--'); ax.axvline(0.3 / 1.7, color='#999999', lw=0.7, ls=':')
ax.text(0.19, 0.93, r'$m^*=0.176$', fontsize=6.5, color='#555555')
ax.text(0.20, 0.715, r'$\theta=0.7$', fontsize=6.5, color='#555555')
ax.set_xlabel('Share of shingles touching an identifier ($m$)'); ax.set_ylabel('Jaccard after SUR-DOC')
ax.set_xlim(0, 0.92); ax.set_ylim(0, 1.02)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
ax.grid(color='#e5e5e5', lw=0.4); ax.set_axisbelow(True)
ax.legend(fontsize=5.8, frameon=False, loc='upper right', markerscale=2.2, handletextpad=0.3)
fig.tight_layout(pad=0.3); fig.savefig('./dose.pdf'); fig.savefig('./dose.png', dpi=200)
