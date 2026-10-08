import json, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
g = json.load(open('results/e1_first.json'))['strategies']; dt = json.load(open('results/e3_tab.json'))['results']
plt.rcParams.update({'font.size': 7.5, 'font.family': 'serif', 'axes.linewidth': 0.6})
fig, ax = plt.subplots(figsize=(3.4, 2.25))
C = {'gold': '#2a78d6', 'detector': '#eb6834'}
off = {'RAW': (4, 3), 'DROP': (4, -8), 'MASK': (4, 3), 'TYPE': (-22, 3), 'ALIAS-DOC': (4, 3), 'SUR-DOC': (4, 3), 'SUR-CORPUS': (4, -8)}
for tag, res, mk in [('gold', g, 'o'), ('detector', dt, 's')]:
    for s, r in res.items():
        x, y = r['0.7']['cases_lost'], r['0.7']['dup_recall']; linkable = r['link']['precision'] > 0.5 if r['link']['precision'] == r['link']['precision'] else False
        ax.scatter(x, y, s=26, marker=mk, facecolor=C[tag] if linkable else 'white', edgecolor=C[tag], linewidth=1.2, zorder=3)
        if tag == 'gold':
            ax.annotate(s, (x, y), xytext=off[s], textcoords='offset points', fontsize=6.5, color='#333333')
ax.set_xlabel('Distinct cases removed by deduplication (of 1,268)')
ax.set_ylabel('Recall of injected near-duplicates')
ax.set_xlim(-2, 30); ax.set_ylim(-0.05, 1.05)
ax.grid(color='#dddddd', linewidth=0.4); ax.set_axisbelow(True)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
from matplotlib.lines import Line2D
h = [Line2D([], [], marker='o', ls='', mfc='white', mec=C['gold'], label='gold spans'),
     Line2D([], [], marker='s', ls='', mfc='white', mec=C['detector'], label='detector spans'),
     Line2D([], [], marker='o', ls='', mfc='#666666', mec='#666666', label='filled: cross-document linkable')]
ax.legend(handles=h, loc='lower right', fontsize=6, frameon=False)
fig.tight_layout(pad=0.3); fig.savefig('./tradeoff.pdf'); fig.savefig('./tradeoff.png', dpi=200)
