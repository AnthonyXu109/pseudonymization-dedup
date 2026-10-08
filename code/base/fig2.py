import json, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
from matplotlib.lines import Line2D
L = lambda f: json.load(open(f))
g = {m: L(f'results/v2/e1_gold_first_{m}.json')['results'] for m in ('plain', 'datatrove')}
en = L('results/v2/e4_enron.json')
plt.rcParams.update({'font.size': 7.5, 'font.family': 'serif', 'axes.linewidth': 0.6})
C = {'plain': '#2a78d6', 'datatrove': '#eb6834'}
# Figure 1: trade-off scatter, TAB gold
fig, ax = plt.subplots(figsize=(3.4, 2.3))
for m, mk in (('plain', 'o'), ('datatrove', 's')):
    for s_, r in g[m].items():
        x, y = r['0.7']['cases_lost'], r['0.7']['dup_recall']
        p = r['link_person']['precision']; filled = p == p and p > 0.5
        ax.scatter(x + (0.35 if m == 'datatrove' else 0), y, s=24, marker=mk, facecolor=C[m] if filled else 'white', edgecolor=C[m], linewidth=1.1, zorder=3)
A = lambda t, xy, off: ax.annotate(t, xy, xytext=off, textcoords='offset points', fontsize=6.3, color='#333333')
A('SUR-CORPUS', (5, 0.91), (-14, -11)); A('RAW', (11, 0.91), (-4, 6)); A('RAW (norm.)', (19, 0.915), (-22, 6))
A('MASK / TYPE / DROP', (24, 0.905), (-30, -11)); A('ALIAS-DOC\n(norm.)', (26, 0.917), (2, 5))
A('ALIAS-DOC', (13, 0.593), (6, -2)); A('SUR-DOC', (0, 0.043), (7, 2))
ax.set_xlabel('Distinct judgments removed by deduplication (of 1,268)')
ax.set_ylabel('Recall of injected copies')
ax.set_xlim(-2, 32); ax.set_ylim(-0.05, 1.12)
ax.grid(color='#dddddd', linewidth=0.4); ax.set_axisbelow(True)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
h = [Line2D([], [], marker='o', ls='', mfc='white', mec=C['plain'], label='token shingles'),
     Line2D([], [], marker='s', ls='', mfc='white', mec=C['datatrove'], label='datatrove-normalized'),
     Line2D([], [], marker='o', ls='', mfc='#666666', mec='#666666', label='filled: person names linkable')]
ax.legend(handles=h, loc='center right', bbox_to_anchor=(1.0, 0.35), fontsize=6, frameon=False)
fig.tight_layout(pad=0.3); fig.savefig('./tradeoff.pdf'); fig.savefig('./tradeoff.png', dpi=200)
# Figure 2: duplicate recall across duplicate sets, plain tokens
sets = [('TAB injected\n(gold spans)', g['plain']), ('Enron injected\n(detector)', en['plain/synthetic']['results']), ('Enron natural\nexact copies', en['plain/natural']['results'])]
strats = ['RAW', 'MASK', 'ALIAS-DOC', 'SUR-DOC', 'SUR-CORPUS']
cols = ['#8a8a8a', '#2a78d6', '#1baf7a', '#eb6834', '#e87ba4']
fig, ax = plt.subplots(figsize=(3.4, 1.9))
w = 0.16; x = np.arange(len(sets))
for k, (s, c) in enumerate(zip(strats, cols)):
    vals = [r[s]['0.7']['dup_recall'] for _, r in sets]
    ax.bar(x + (k - 2) * w, vals, w * 0.9, color=c, label=s, zorder=3)
ax.set_xticks(x); ax.set_xticklabels([n for n, _ in sets], fontsize=6.5)
ax.set_ylabel('Duplicate recall ($\\theta$=0.7)'); ax.set_ylim(0, 1.05)
ax.grid(axis='y', color='#dddddd', linewidth=0.4); ax.set_axisbelow(True)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
ax.legend(fontsize=5.8, ncol=5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.2), handlelength=1, columnspacing=0.8)
fig.tight_layout(pad=0.3); fig.savefig('./recall.pdf'); fig.savefig('./recall.png', dpi=200)
