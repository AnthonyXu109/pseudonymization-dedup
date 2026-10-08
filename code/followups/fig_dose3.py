import json, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows = json.load(open('results/v2/dose_pairs.json'))
C1 = [r['c1'] for r in json.load(open('results/v2/dose_c1_pairs.json'))]
for r, c in zip(rows, C1): r['c1'] = c
plt.rcParams.update({'font.size': 7, 'font.family': 'serif', 'axes.linewidth': 0.6})
groups = {'TAB (gold/detector)': ('TAB gold', 'TAB detector', 'TAB second-source'),
          'Enron': ('Enron detector', 'Enron natural', 'Enron second-source', 'Enron Dolma-like'),
          'ECtHR': ('ECtHR second-source',), 'Web': ('Web second-source',)}
cols = dict(zip(groups, ('#2a78d6', '#eb6834', '#1baf7a', '#e87ba4')))
rng = np.random.default_rng(0)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.45, 1.85))
for g, sets in groups.items():
    R = [r for r in rows if r['setting'] in sets]
    if len(R) > 700: R = [R[i] for i in rng.choice(len(R), 700, replace=False)]
    a1.scatter([r['m_shared'] for r in R], [r['J_SUR-DOC'] for r in R], s=2.5, color=cols[g], alpha=0.5, linewidths=0, label=g, zorder=3)

m = np.linspace(0, 1, 200)
a1.plot(m, (1 - m) / (1 + m), color='#222222', lw=0.9, zorder=4)
a1.axhline(0.7, color='#999999', lw=0.6, ls='--'); a1.axvline(0.3 / 1.7, color='#999999', lw=0.6, ls=':')
a1.text(0.19, 0.03, r'$m^*$', fontsize=6.5, color='#555555')
a1.set_xlabel(r'$m_{\mathrm{sh}}$'); a1.set_ylabel(r"Measured $J'$ (SUR-DOC)")
a1.set_xlim(0, 0.95); a1.set_ylim(0, 1.02); a1.set_title('(a) bound $(1-m)/(1+m)$', fontsize=7)
for lab, flag, col in (('consistent marking', True, '#222222'), ('inconsistent', False, '#eb6834')):
    R = [r for r in rows if r['c1'] == flag]
    a2.scatter([r['J_pred_shared'] for r in R], [r['J_SUR-DOC'] - r['J_pred_shared'] for r in R], s=2.2, color=col, alpha=0.45, linewidths=0, label=f"{lab} (n={len(R):,})", zorder=3 if flag else 2)
a2.axhline(0, color='#999999', lw=0.6)
a2.set_xlabel(r"Predicted $J'$, Eq. (1)"); a2.set_ylabel('Measured $-$ predicted', labelpad=1)
a2.set_xlim(0, 1.0); a2.set_ylim(-0.05, 0.05); a2.set_title('(b) residuals', fontsize=7)
a2.legend(fontsize=5.2, frameon=False, loc='upper left', markerscale=3, handletextpad=0.2, borderpad=0.1)
for ax in (a1, a2):
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.grid(color='#e5e5e5', lw=0.4); ax.set_axisbelow(True)
fig.legend(*a1.get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.5, 0.98), ncol=4, columnspacing=0.8, markerscale=3, handletextpad=0.2, borderpad=0.1)
fig.tight_layout(pad=0.25, w_pad=0.6); fig.savefig('./dose.pdf', bbox_inches='tight', pad_inches=0.02); fig.savefig('./dose.png', dpi=220, bbox_inches='tight', pad_inches=0.02)
# residual stats for the paper
out = {}
for s in dict.fromkeys(r['setting'] for r in rows):
    X = [r for r in rows if r['setting'] == s]
    e = np.array([r['J_SUR-DOC'] - r['J_pred_shared'] for r in X]); ed = np.array([r['J_SUR-DOC'] - r['J_pred'] for r in X])
    out[s] = dict(n=len(X), mae_shared=float(np.abs(e).mean()), max_shared=float(np.abs(e).max()), mae_doc=float(np.abs(ed).mean()),
                  median_m_doc=float(np.median([r['m_doc'] for r in X])), median_m_shared=float(np.median([r['m_shared'] for r in X])),
                  rec_obs=float(np.mean([r['J_SUR-DOC'] >= 0.7 for r in X])), rec_pred=float(np.mean([r['J_pred_shared'] >= 0.7 for r in X])),
                  viol_doc=int(sum(r['m_doc'] > 0.3/1.7 + 1e-9 and r['J_SUR-DOC'] >= 0.7 - 1e-9 for r in X)), viol_shared=int(sum(r['m_shared'] > 0.3/1.7 + 1e-9 and r['J_SUR-DOC'] >= 0.7 - 1e-9 for r in X)))
pass
print(json.dumps({k: {kk: round(v, 4) if isinstance(v, float) else v for kk, v in d.items()} for k, d in out.items()}, indent=0)[:1500])
