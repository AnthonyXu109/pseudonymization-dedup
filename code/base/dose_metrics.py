"""Separates the three things the text reports about the prediction: numeric error, agreement at theta=0.7,
and the per-document warning rule m_doc > (1-theta)/(1+theta)."""
import json, numpy as np
R = json.load(open('results/v2/dose_pairs.json')); TH = 0.7; MSTAR = (1 - TH) / (1 + TH); EPS = 1e-9
J = np.array([r['J_surdoc'] for r in R]); Ps = np.array([r['J_pred_shared'] for r in R]); Pd = np.array([r['J_pred'] for r in R])
md = np.array([r['m_doc'] for r in R]); ms = np.array([r['m_shared'] for r in R])
reach = J >= TH - EPS
out = dict(n=len(R), reach=int(reach.sum()),
    mae_shared=float(np.mean(np.abs(J - Ps))), max_shared=float(np.max(np.abs(J - Ps))),
    disagree_theta_shared=int(np.sum((Ps >= TH - EPS) != reach)),
    disagree_theta_doc=int(np.sum((Pd >= TH - EPS) != reach)),
    warn_false_alarm=int(np.sum((md > MSTAR + EPS) & reach)),      # warned, but the pair still reaches theta
    warn_miss=int(np.sum((md <= MSTAR + EPS) & ~reach)),           # no warning, pair falls below theta
    warned=int(np.sum(md > MSTAR + EPS)), not_warned=int(np.sum(md <= MSTAR + EPS)),
    bound_violations_shared=int(np.sum((ms > MSTAR + EPS) & reach)))
per = {}
for s in dict.fromkeys(r['setting'] for r in R):
    k = np.array([r['setting'] == s for r in R])
    per[s] = dict(n=int(k.sum()), disagree_shared=int(np.sum(((Ps >= TH - EPS) != reach) & k)), disagree_doc=int(np.sum(((Pd >= TH - EPS) != reach) & k)),
                  false_alarm=int(np.sum((md > MSTAR + EPS) & reach & k)), miss=int(np.sum((md <= MSTAR + EPS) & ~reach & k)),
                  not_warned=int(np.sum((md <= MSTAR + EPS) & k)))
out['per_setting'] = per
json.dump(out, open('results/v2/dose_metrics.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'per_setting'}, indent=0)); [print(k, v) for k, v in per.items()]
# warning rule evaluated where it is meant to apply: pairs that the unscrubbed text keeps at theta
JR = np.array([r['J_raw'] for r in R]); base = JR >= TH - EPS
warn = md > MSTAR + EPS; lostp = base & ~reach
conf = dict(raw_pairs=int(base.sum()), lost=int(lostp.sum()), warned_and_lost=int((warn & lostp).sum()),
            warned_but_kept=int((warn & base & reach).sum()), not_warned_but_lost=int((~warn & lostp).sum()),
            not_warned_kept=int((~warn & base & reach).sum()))
# numeric error of the m_doc form
conf['mae_doc_formula'] = float(np.mean(np.abs(J - Pd)))
out['warning_rule_on_raw_pairs'] = conf
json.dump(out, open('results/v2/dose_metrics.json', 'w'), indent=1)
print(conf)
for s in per:
    k = np.array([r['setting'] == s for r in R]) & base
    print(s, int(k.sum()), 'lost', int((k & ~reach).sum()), 'warned&lost', int((k & ~reach & warn).sum()), 'FA', int((k & reach & warn).sum()), 'miss', int((k & ~reach & ~warn).sum()))
