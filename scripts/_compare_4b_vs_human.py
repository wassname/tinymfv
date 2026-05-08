"""Quick: 4B forced-choice vs human distributions on classic."""
import json, numpy as np
from tabulate import tabulate

fc = [json.loads(l) for l in open('data/results/forced_choice_classic_qwen4b.jsonl')]
ml = {r['id']: r for r in (json.loads(l) for l in open('data/multilabel.jsonl'))}

PROBE = ['care','fairness','loyalty','authority','sanctity','liberty','social']
HUMAN = ['Care','Fairness','Loyalty','Authority','Sanctity','Liberty','SocialNorms']

p_model, p_human = [], []
for r in fc:
    m = ml[r['id']]
    h = np.array([m[f'human_{f}'] for f in HUMAN], dtype=float)
    if h.sum() <= 0:
        continue
    h = h / h.sum()
    p = np.array([r['p'][k] for k in PROBE], dtype=float)
    p = p / p.sum()
    p_model.append(p); p_human.append(h)
p_model = np.array(p_model); p_human = np.array(p_human)
n = len(p_model)
print(f'n={n} matched rows\n')

print('=== mean probability per foundation (across vignettes) ===')
rows = []
for i, name in enumerate(HUMAN):
    rows.append([name,
                 f'{p_model[:,i].mean():.3f}',
                 f'{p_human[:,i].mean():.3f}',
                 f'{p_model[:,i].mean()-p_human[:,i].mean():+.3f}'])
print(tabulate(rows, headers=['foundation','model','human','model-human'], tablefmt='pipe'))

print('\n=== per-foundation Pearson r (model vs human, across vignettes) ===')
print('high = when humans put mass on X, model also does')
rows = []
for i, name in enumerate(HUMAN):
    r = float(np.corrcoef(p_model[:,i], p_human[:,i])[0,1])
    rows.append([name, f'{r:+.3f}'])
print(tabulate(rows, headers=['foundation','pearson_r'], tablefmt='pipe'))

def js(p, q):
    p = p + 1e-12; q = q + 1e-12
    p = p/p.sum(); q = q/q.sum()
    m = (p + q) / 2
    return 0.5*float((p*np.log(p/m)).sum()) + 0.5*float((q*np.log(q/m)).sum())

ce_model = -(p_human * np.log(p_model + 1e-12)).sum(axis=1)
ce_uniform = np.full(n, np.log(7))
js_vals = np.array([js(p_model[i], p_human[i]) for i in range(n)])

print('\n=== distribution distance (per row) ===')
print(f'CE(human || model_4B):  mean={ce_model.mean():.3f}  median={np.median(ce_model):.3f}  nats')
print(f'CE(human || uniform):   mean={ce_uniform.mean():.3f} nats  (= log 7)')
print(f'CE gain vs uniform:     {ce_uniform.mean()-ce_model.mean():+.3f} nats  ({100*(1-ce_model.mean()/ce_uniform.mean()):.1f}%)')
print(f'JS(model || human):     mean={js_vals.mean():.3f}  median={np.median(js_vals):.3f}  (max=ln 2={np.log(2):.3f})')

m_top = p_model.argmax(axis=1)
h_top = p_human.argmax(axis=1)
print(f'top1 argmax agreement:  {(m_top==h_top).mean()*100:.1f}%')

# soft-argmax confusion: rows = human-argmax class, cols = mean p_model[col]
print('\n=== soft confusion: rows=human-argmax, cols=mean p_model ===')
rows = []
for i, name in enumerate(HUMAN):
    sel = h_top == i
    if not sel.any():
        continue
    means = p_model[sel].mean(axis=0)
    rows.append([f'{name} (n={int(sel.sum())})'] + [f'{v:.2f}' for v in means])
print(tabulate(rows, headers=['true \\ pred'] + HUMAN, tablefmt='pipe'))
