"""Enron crossing point: which distinct messages RAW removes at theta=0.9 and SUR-DOC at theta=0.55 (second-source copies)."""
import sys; pass
import pickle, numpy as np, json, collections
from common import *
from evalsparse import jaccard_pairs
docs=pickle.load(open('results/enron_real_detected.pkl','rb')); n=len(docs); case=np.array([d['case'] for d in docs])
def comps(I,J):
    p=list(range(n))
    def f(x):
        while p[x]!=x: p[x]=p[p[x]]; x=p[x]
        return x
    for a,b in zip(I,J):
        ra,rb=f(a),f(b)
        if ra!=rb: p[max(ra,rb)]=min(ra,rb)
    return np.array([f(i) for i in range(n)])
def lost(s,th):
    sets=[shingles(release(d['text'],d['spans'],s,d['key'],d['case'])) for d in docs]
    I,J,V=jaccard_pairs(sets,th); c=comps(I.tolist(),J.tolist())
    kept=set(case[np.where(c==np.arange(n))[0]].tolist()); return {x for x in set(case.tolist()) if x not in kept}, sets, c
LR,SR,cR=lost('RAW',0.9); LS,SS,cS=lost('SUR-DOC',0.55)
print(len(LR),len(LS),len(LR&LS))
orig={d['case']:i for i,d in enumerate(docs) if not d['is_copy']}
def stats(L):
    ln=[len(docs[orig[c]]['text'].split()) for c in L]; md=[]
    for c in L:
        d=docs[orig[c]]; w=len(d['text'].split()); md.append(sum(s['end']-s['start'] for s in d['spans'])/max(1,len(d['text'])))
    return np.median(ln), np.median(md)
R=dict(raw_090_removed=len(LR), surdoc_055_removed=len(LS), common=len(LR&LS), raw_only=len(LR-LS), surdoc_only=len(LS-LR),
       raw_only_median_words_and_id_char_share=stats(LR-LS), surdoc_only_median=stats(LS-LR), common_median=stats(LR&LS))
print(R); json.dump({k: (list(map(float, v)) if isinstance(v, tuple) else v) for k, v in R.items()}, open('results/v2/enron_cross.json', 'w'), indent=1)
import random; random.seed(1)
for c in random.sample(sorted(LR-LS),4):
    i=orig[c]; mem=[j for j in np.where(cR==cR[i])[0] if case[j]!=c][:1]
    print('----RAWonly', repr(docs[i]['text'][:250])); 
    for j in mem: print('  partner', repr(docs[j]['text'][:250]))
for c in random.sample(sorted(LS-LR),3):
    i=orig[c]; mem=[j for j in np.where(cS==cS[i])[0] if case[j]!=c][:1]
    print('----SURonly', repr(docs[i]['text'][:250]))
    for j in mem: print('  partner', repr(docs[j]['text'][:250]))
