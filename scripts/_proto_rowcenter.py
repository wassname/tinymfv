"""Prototype: does row-centering Likert scores improve top1 vs labeled coarse foundation?

Uses offline LLM Likert (1-5) per foundation as a stand-in for live multibool logratios.
Row centering: l~_f = l_f - mean_g(l_g). Cancels per-item global "badness".
"""
import json, math
from pathlib import Path
from collections import Counter

FS = ['Care','Fairness','Loyalty','Authority','Sanctity','Liberty','SocialNorms']
COARSE_TO_FS = {'Care':'Care','Fairness':'Fairness','Loyalty':'Loyalty',
                'Authority':'Authority','Sanctity':'Sanctity','Liberty':'Liberty',
                'Social Norms':'SocialNorms'}


def topk_acc(preds, labels, k=1):
    return sum(1 for p, l in zip(preds, labels) if l in p[:k]) / len(labels)


def avg_abs_offdiag(cols):
    n = len(next(iter(cols.values())))
    def corr(a, b):
        ma = sum(a)/n; mb = sum(b)/n
        va = sum((x-ma)**2 for x in a); vb = sum((x-mb)**2 for x in b)
        if not (va and vb): return float('nan')
        return sum((x-ma)*(y-mb) for x, y in zip(a, b)) / math.sqrt(va*vb)
    pairs = []
    for i, a in enumerate(FS):
        for b in FS[i+1:]:
            pairs.append(abs(corr(cols[a], cols[b])))
    return sum(pairs)/len(pairs), max(pairs)


def per_class_top1(preds, labels):
    out = {}
    for f in FS:
        sub = [(p, l) for p, l in zip(preds, labels) if l == f]
        out[f] = (sum(1 for p, l in sub if p[0] == l)/len(sub) if sub else float('nan'), len(sub))
    return out


for name in ['multilabel.jsonl', 'multilabel_airisk.jsonl']:
    path = Path('data') / name
    if not path.exists():
        print(f"missing {path}"); continue
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    labels = [COARSE_TO_FS[r['foundation_coarse']] for r in rows]
    label_dist = Counter(labels)

    raw = [{f: float(r['llm_'+f]) for f in FS} for r in rows]
    cent = []
    for d in raw:
        m = sum(d.values())/len(d)
        cent.append({f: d[f]-m for f in FS})

    def rank(scores):
        return [sorted(FS, key=lambda f: -d[f]) for d in scores]

    raw_pred = rank(raw)
    cent_pred = rank(cent)

    raw_cols = {f: [d[f] for d in raw] for f in FS}
    cent_cols = {f: [d[f] for d in cent] for f in FS}
    raw_mean, raw_max = avg_abs_offdiag(raw_cols)
    cent_mean, cent_max = avg_abs_offdiag(cent_cols)

    print(f"\n=== {name}  n={len(rows)} ===")
    print(f"  raw  top1={topk_acc(raw_pred,labels):.3f}  top2={topk_acc(raw_pred,labels,2):.3f}  |offdiag| mean={raw_mean:.3f} max={raw_max:.3f}")
    print(f"  cent top1={topk_acc(cent_pred,labels):.3f}  top2={topk_acc(cent_pred,labels,2):.3f}  |offdiag| mean={cent_mean:.3f} max={cent_max:.3f}")
    print(f"  label dist: {dict(label_dist)}")
    pc_raw = per_class_top1(raw_pred, labels)
    pc_cent = per_class_top1(cent_pred, labels)
    print(f"  per-class top1 (raw -> centered):")
    for f in FS:
        rA, n = pc_raw[f]; cA, _ = pc_cent[f]
        print(f"    {f:11s} n={n:3d}  raw={rA:.2f}  cent={cA:.2f}  delta={cA-rA:+.2f}")
