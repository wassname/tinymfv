# tinymfv (tiny moral/value eval for local LLMs)

A fast, sensitive eval that measures a local model's moral profile and whether an intervention
(weight steering, a prompt, a fine-tune) moves it. Instead of sampling and parsing an answer, it
prefills the answer slot and reads the next-token logprobs over the seven moral foundations, so a
small steering vector shows up as a shift in nats long before it would flip a sampled argmax.

The default instrument is the Clifford moral-foundation vignettes (MFV): 132 scenarios, each with a
human distribution over the foundations care / fairness / loyalty / authority / sanctity / liberty /
social (Clifford et al. 2015). Three configs (`classic`, `scifi`, `ai-actor`), two framings each
(`other_violate`, `self_violate`). [[HF dataset](https://huggingface.co/datasets/wassname/tiny-mfv)]

![LLM vs 19 human societies on the moral-foundations map](docs/img/showcase/mfq2/map_pca_ipsative.png)

## Install

```bash
uv pip install git+https://github.com/wassname/tinymfv
```

## Core API

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from tinymfv import load_vignettes, evaluate

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B").cuda()

vignettes = load_vignettes("classic")              # list[dict], one per scenario
report = evaluate(model, tok, vignettes=vignettes) # dev mode: N=1, 64 think tokens
print(report["top1_acc"], report["mean_pmass_allowed"])
print(report["profile"])                           # mean p[foundation] vs the human profile
```

`load_vignettes(name)` returns the scenarios (each a dict with the prompt per framing and the human
label distribution). `evaluate(model, tok, ...)` runs a forced-choice probe per (vignette,
condition) and returns a dict with:

- `profile`: mean `p[foundation]` across vignettes, on the same 7-way simplex as the human profile.
- `top1_acc`, `informedness`, `mean_nll_T`: agreement vs the human label (None if the config is unlabeled).
- `mean_pmass_allowed`: mean full-vocab probability mass on the allowed answer tokens at the answer
  slot. This is answer-slot format coherence: it drops when the model wants to emit prose, refusal,
  punctuation, or another out-of-space token, independent of which valid answer is top.
- `mean_nll_prefill`: mean NLL/token of the forced assistant prefill that leads into the answer slot.
- `per_row` (with `return_per_row=True`): the per-row 7-vec `p`, raw `score` (nats), `pmass_allowed`,
  `nll_prefill`, `top1`, `margin`. This is what the steering metrics below consume.

To measure a steering intervention, run `evaluate` twice (base vs steered, same vignettes) and diff
the reports. The steering-lite package wraps this as `evaluate_with_vector(model, tok, vector=v)`,
which returns `raw_logratios[vid|cond][foundation] = logit(p[foundation])` for the two metrics below.

## The two steering metrics

Both compare a steered report against a base report, per foundation.

**dlogit / dlogprob** (`dlogit_per_foundation`). The paired delta in the logit of the foundation
probability:

$$\Delta(\text{vid},\text{cond},f) = \mathrm{logit}\,p_{\text{steer}}[f] - \mathrm{logit}\,p_{\text{base}}[f]$$

averaged over all (vignette, condition) pairs. Units are nats. Positive means the steer made the
model more likely to call that foundation the violation; negative, less likely. It is paired
(same vignette base vs steer) and calibration-free, so it does not saturate the way a probability
delta would near 0 or 1. This is the continuous effect-size signal.

**SI -- Surgical Informedness** (`si_per_foundation` in steering-lite, the canonical implementation).
A bidirectional, reference-anchored score that asks "did the steer move the model toward the intended
direction at `+C` and away from it at `-C`, without breaking the rows that were already right?" Per
foundation:

$$\mathrm{SI} = \mathrm{mean}(\mathrm{SI_{fwd}}, \mathrm{SI_{rev}}) \times \text{pmass\_scale}$$

where `SI_fwd = fix_rate - k * broke_rate` (fixes are rows the steer flipped toward intent; broke are
rows it flipped away, penalized `k`x), and `SI_rev` is the same on the `-C` pole. The reference is the
base model's per-row decision at threshold logit=0 (p=0.5), so SI > 0 always means "moved toward
intent at `+C` and away at `-C`". `pmass_scale = tanh(min margin)^2` softly drops methods whose K-way
decision has collapsed. SI moves only when an answer flips, so it is less sensitive than dlogit but
more robust: use dlogit for effect size, SI for "did the steer do the intended surgical thing".

## Design notes

- Logprobs, not sampled answers. Prefill the answer slot, read the next-token distribution; small
  interventions register in nats before changing an argmax.
- Position-bias control. Each row is scored twice (options forward and reversed) and the logprob
  vectors averaged, cancelling option-order effects ([Pezeshkpour & Hruschka 2023](https://arxiv.org/abs/2308.11483)).
- A sliding think budget. `max_think_tokens` (0 / 64 dev default / 4096 / unbounded) is a setting you
  sweep: steering accrues over the think trace, so the same vector moves the profile more with more
  think, up to the point (~512) where the model closes `</think>` on its own and the readout collapses.
- Two modes. dev (N=1, greedy, 64 think) is fast and granular, the default. full (N=4 sampled traces
  per ordering, Bayesian-model-averaged, high think) adds sampling-variance uncertainty and SI.

## Instruments

The reader is answer-space-agnostic: it gathers logprobs over answer tokens at a prefilled slot
(`src/tinymfv/instrument.py`).

- Nominal instruments, the MFV vignettes, read a foundation category and reduce to mean category
  probability.
- Ordinal instruments, MFQ-2 / Big-Five / 16PF / humor-styles, read a 1..M scale point and reduce to
  keyed expected score `E`, logit contrast `C`, `logodds_agree`, entropy, and `pmass_allowed`.

## Scope

A fast sensitive eval for small steering interventions on local models, not a full moral-reasoning
evaluation. For behaviour-heavy evals see
[machiavelli](https://huggingface.co/datasets/wassname/machiavelli),
[AIRiskDilemmas](https://huggingface.co/datasets/kellycyy/AIRiskDilemmas),
[ethics_expression_preferences](https://huggingface.co/datasets/wassname/ethics_expression_preferences).

Used in [steering-lite](https://github.com/wassname/steering-lite),
[lora-lite](https://github.com/wassname/lora-lite),
[w2schar-mini](https://github.com/wassname/w2schar-mini).

## Citation

```bibtex
@misc{clark2026tinymfv,
  title = {tinymfv: tiny moral/value eval for local LLMs},
  author = {Michael Clark},
  year = {2026},
  url = {https://github.com/wassname/tinymfv/}
}
```
