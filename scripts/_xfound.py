"""Cross-foundation correlations: human vs LLM judge vs probed model.

Question: do foundations move together (e.g. care+authority correlated)?
If LLMs collapse foundations into a single 'badness' axis, all pairs would be
positive. If forced-choice is mass-stealing properly, pairs should be negative
or near-zero.
"""
import json, numpy as np
from tabulate import tabulate

ml = [json.loads(l) for l in open('data/multilabel.jsonl')]
fc = [json.loads(l) for l in open('data/results/forced_choice_classic_qwen4b.jsonl')]
fc_by_id = {r['id']: r for r in fc}

F = ['Care','Fairness','Loyalty','Authority','Sanctity','Liberty','SocialNorms']
P = ['care','fairness','loyalty','authority','sanctity','liberty','social']

def corr_table(arr, label):
    n = arr.shape[0]
    c = np.corrcoef(arr.T)
    rows = []
    for i, fi in enumerate(F):
        rows.append([fi] + [f'{c[i,j]:+.2f}' if i != j else '----' for j in range(7)])
    print(f'\n=== {label} (n={n}) ===')
    print(tabulate(rows, headers=[''] + F, tablefmt='pipe'))
    # off-diagonal stats
    off = c[np.triu_indices(7, k=1)]
    print(f'  off-diag: mean={off.mean():+.2f}  '
          f'pos_pairs={int((off>0.1).sum())}/{len(off)}  '
          f'neg_pairs={int((off<-0.1).sum())}/{len(off)}')

# Human soft labels (Clifford 2015)
H = []
for r in ml:
    h = np.array([r[f'human_{f}'] for f in F], dtype=float)
    if h.sum() > 0:
        H.append(h / h.sum())
H = np.array(H)
corr_table(H, 'Human (Clifford 2015 % distributions)')

# Grok calibrated (LLM judge)
C = []
for r in ml:
    c = np.array([r[f'calibrated_{f}'] for f in F], dtype=float)
    if c.sum() > 0:
        C.append(c / c.sum())
C = np.array(C)
corr_table(C, 'Grok-4-fast judge (calibrated dist, normalised to sum=1)')

# Qwen3-4B forced-choice
M = []
for r in fc:
    m = np.array([r['p'][k] for k in P], dtype=float)
    M.append(m / m.sum())
M = np.array(M)
corr_table(M, 'Qwen3-4B forced-choice')

# === marginals comparison ===
print('\n=== marginal mean p[f] -- "moral landscape" ===')
rows = []
for i, fi in enumerate(F):
    rows.append([fi,
                 f'{H[:,i].mean():.3f}',
                 f'{C[:,i].mean():.3f}',
                 f'{M[:,i].mean():.3f}'])
print(tabulate(rows, headers=['foundation', 'human', 'grok', 'qwen4b'], tablefmt='pipe'))

# === cross-pearson per foundation ===
print('\n=== per-foundation cross-source Pearson r (across vignettes) ===')
print('Higher = source agrees with humans on which vignettes load on this foundation.')
rows = []
for i, fi in enumerate(F):
    rh_g = float(np.corrcoef(H[:,i], C[:,i])[0,1])
    rh_m = float(np.corrcoef(H[:,i], M[:,i])[0,1])
    rg_m = float(np.corrcoef(C[:,i], M[:,i])[0,1])
    rows.append([fi, f'{rh_g:+.2f}', f'{rh_m:+.2f}', f'{rg_m:+.2f}'])
print(tabulate(rows, headers=['foundation','human-grok','human-qwen4b','grok-qwen4b'], tablefmt='pipe'))
