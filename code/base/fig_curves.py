import json, os, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 7.5, 'font.family': 'serif', 'axes.linewidth': 0.6})
ex = json.load(open('results/v2/curves_exact.json'))
S = ['RAW', 'MASK', 'SUR-CORPUS', 'SUR-DOC', 'HASH-ENTITY']
col = {'RAW': '#222222', 'MASK': '#2a78d6', 'SUR-CORPUS': '#e87ba4', 'SUR-DOC': '#eb6834', 'HASH-ENTITY': '#1baf7a'}
mk = {'RAW': 'o', 'MASK': 's', 'SUR-CORPUS': 'D', 'SUR-DOC': 'v', 'HASH-ENTITY': '^'}
fig, axs = plt.subplots(1, 4, figsize=(7.1, 1.9))
def exact_panel(ax, key, title, N):
    d = ex[key]
    for s in S:
        th = sorted(d[s], key=float)
        x = [d[s][t]['cases_lost'] for t in th]; y = [d[s][t]['dup_recall_comp'] for t in th]
        ax.plot(x, y, color=col[s], marker=mk[s], ms=2.6, lw=0.9, label=s)
    ax.set_title(title, fontsize=7)
def lsh_panel(ax, f, title):
    if not os.path.exists(f): ax.set_title(title + ' (pending)', fontsize=7); return
    R = json.load(open(f))['results']
    for s in S:
        xs, ys, xl, xh, yl, yh = [], [], [], [], [], []
        for band in ('7x16', '8x14', '14x8', '16x7', '28x4'):
            v = [R[k] for k in R if k.startswith(s + '|') and k.endswith('|' + band)]
            if not v: continue
            x = np.array([t['removed'] for t in v]); y = np.array([t['recall'] for t in v])
            xs.append(x.mean()); ys.append(y.mean()); xl.append(x.mean() - x.min()); xh.append(x.max() - x.mean()); yl.append(y.mean() - y.min()); yh.append(y.max() - y.mean())
        if xs: ax.errorbar(xs, ys, xerr=[xl, xh], yerr=[yl, yh], color=col[s], marker=mk[s], ms=2.6, lw=0.9, elinewidth=0.5, capsize=0, label=s)
    ax.set_title(title, fontsize=7); ax.set_xscale('symlog', linthresh=10); ax.set_xlim(left=-1)
exact_panel(axs[0], 'TAB second-source|plain', '(a) TAB, exact Jaccard', 1268)
exact_panel(axs[1], 'Enron second-source|plain', '(b) Enron, exact Jaccard', 16963)
lsh_panel(axs[2], 'results/v2/pipeline_ecthr_plain.json', '(c) ECtHR, MinHash')
lsh_panel(axs[3], 'results/v2/pipeline_c4_plain.json', '(d) Web, MinHash')
for i, ax in enumerate(axs):
    ax.set_ylim(-0.03, 1.03); ax.grid(color='#e5e5e5', lw=0.4); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.set_xlabel('Distinct records removed', fontsize=6.5)
axs[0].set_ylabel('Pipeline recall')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=5, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 1.04))
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.93)); fig.savefig('./curves.pdf', bbox_inches='tight', pad_inches=0.02); fig.savefig('./curves.png', dpi=200, bbox_inches='tight', pad_inches=0.02)
