# tinymfv (tiny moral/value eval for local LLMs)

tinymfv reads answer-token logprobs from local chat LMs and turns them into moral/value profiles.
It is meant for fast steering experiments: prefill the answer slot, read the next-token
distribution, and compare base vs steered runs in nats instead of waiting for sampled answers to
flip.

There are two instrument kinds:

- Nominal MFV vignettes: the answer is a foundation category. The profile is the mean probability
  of care / fairness / loyalty / authority / sanctity / liberty / social.
- Ordinal questionnaires: the answer is a scale point 1..M. MFQ-2, Big Five, 16PF, and Humor Styles
  all use the same reader and reduce to per-factor Likert profiles.

The default nominal instrument is the Clifford moral-foundation vignettes (MFV): 132 scenarios, each
with a human distribution over foundations (Clifford et al. 2015). Three configs (`classic`,
`scifi`, `ai-actor`), two framings each (`other_violate`, `self_violate`).
[[HF dataset](https://huggingface.co/datasets/wassname/tiny-mfv)]

The bundled showcase steers Qwen3-4B with one activation vector, extracted once from
`wassname/moral_stories_foundations` fairness situations (mean activation difference, fairness vs a
balanced sample of the other foundations, situations only, no completions), iso-KL calibrated, then
read across every instrument at base / +c / -c. On MFQ-2 it selectively raises the equality factor:
equality is the only factor that rises at the gentlest +c, every other factor falls, a clean
directional move on the fairness-equality axis. On the MFV vignettes, though, the same vector moves
Care, not Fairness, so it is a care-leaning moral-salience direction that reads as fairness on the
questionnaire rather than a clean single-foundation isolator. Coherence holds across the sweep
(`frac_unscorable` 0, `mean_margin` ~10-12 nats), so the readouts below are not steering artifacts.

The same vector is read on all five bundled instruments. Range plots show the per-factor steer path
(red `+c` up, blue `-c` down) against human society ranges; maps are PCA culture maps with the c-sweep
trajectory. The 16PF map is omitted (16 axes do not lay out as a readable 2-D map; its range is kept).

![MFQ-2 culture map: base and fairness-steered points against human societies](docs/img/showcase/mfq2/map_pca_ipsative.png)

![MFQ-2 range plot: human society ranges beside the fairness steer path; equality rises at +c](docs/img/showcase/mfq2/range.png)

![Big Five culture map with the c-sweep trajectory](docs/img/showcase/big5/map_pca_ipsative.png)

![Big Five range: fairness steer path per trait against human society ranges](docs/img/showcase/big5/range.png)

![Humor Styles culture map with the c-sweep trajectory](docs/img/showcase/humor_styles/map_pca_ipsative.png)

![Humor Styles range: fairness steer path per style against human ranges](docs/img/showcase/humor_styles/range.png)

![16PF range: fairness steer path across all 16 factors](docs/img/showcase/16pf/range.png)

![MFV culture map with the c-sweep trajectory](docs/img/showcase/mfv/map_pca_ipsative.png)

![MFV range: per-foundation emphasis; Care moves most under the fairness vector, Fairness barely](docs/img/showcase/mfv/range.png)

## Install

```bash
uv pip install git+https://github.com/wassname/tinymfv
```

For maps:

```bash
uv pip install "tiny-mfv[maps] @ git+https://github.com/wassname/tinymfv"
```

For repo development:

```bash
git clone https://github.com/wassname/tinymfv
cd tinymfv
uv sync --extra maps --dev
just smoke
```

## Use it

Nominal MFV vignettes use `evaluate`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from tinymfv import load_vignettes, evaluate

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B").cuda()

vignettes = load_vignettes("classic")              # list[dict], one per scenario
report = evaluate(model, tok, vignettes=vignettes, return_per_row=True)
print(report["mean_pmass_allowed"])
print(report["per_row"][0]["score"])               # logprob score per foundation, nats
print(report["profile"])                           # mean p[foundation], easier to read
```

Ordinal questionnaires use `administer`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from tinymfv import administer, get_instrument

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B").cuda()

instr = get_instrument("mfq2")       # "mfq2", "big5", "16pf", or "humor_styles"
report = administer(model, tok, instr)
print(report["dimensions"])
print(report["per_item_frame"][0]["lp"])           # raw logprobs for the answer tokens
print(report["profile_E"])           # human-comparable factor means
print(report["profile_C"])           # steer-sensitive logit contrast per factor
print(report["mean_pmass_allowed"])
```

From a checkout, the small functional run is:

```bash
just smoke
just eval Qwen/Qwen3-0.6B classic
```

The plotting helpers live in `tinymfv.maps`; `scripts/plot_steer_showcase.py` shows the current
map/range pipeline for steering-lite outputs.

## Reader math

For an answer set $A = \{a_1,\ldots,a_K\}$, tinymfv gathers the full-vocab logprobs at the answer
tokens:

$$\ell_k = \log P(a_k \mid \text{prompt}, \text{think trace}, \text{assistant prefill})$$

The raw gathered vector $\ell$ is the primitive. Everything else is a pure readout:

$$\mathrm{pmass\_allowed} = \sum_{k=1}^{K} \exp(\ell_k)$$

$$p_k = \frac{\exp(\ell_k)}{\mathrm{pmass\_allowed}}$$

`pmass_allowed` is answer-format coherence: mass on valid answer tokens at the answer slot. The
slot is force-prefilled, so the scaffold primes a valid token and `pmass_allowed` is pinned high
under the forced-choice path. Treat it as a format floor, not a sensitive coherence gate: a steered
run that has broken the model can still score `pmass_allowed` near 1.0. The two signals that carry
coherence instead are `frac_unscorable` (rows with no scorable slot, a self-closed think with no
answer or a blown-up forward; near 0 once the end-of-answer tokens are suppressed over the budget)
and `nll_prefill` below (does the scaffold still fit the model). pmass does not say which valid
answer is right.

`nll_prefill` measures whether the forced assistant prefill itself fits the model:

$$\mathrm{nll\_prefill} = -\frac{1}{J}\sum_{j=1}^{J} \log P(u_j \mid \text{context}, u_{<j})$$

where $u_1,\ldots,u_J$ are the prefill tokens before the answer token. This catches scaffold
friction before the answer slot; `pmass_allowed` catches the slot itself.

## What to use

Most research code should use the logprob-level outputs. They are sensitive and easy to interpret
in relative terms: positive means the steer moved the answer up on that axis, negative means down,
and the units are nats. For MFV, use per-row `score` or the paired `dlogit_per_foundation`. For
ordinal questionnaires, use raw per-frame `lp` when you want the primitive, or `profile_C` when you
want a factor-level steer readout.

The value summaries are less sensitive but easier to interpret. MFV `profile` is mean
foundation probability. Ordinal `profile_E` is the expected Likert score, in the human scale
direction:

$$E = \sum_{k=1}^{M} k p_k$$

For reverse-keyed items, `keyed_E = M + 1 - E`. `E` is bounded, so it is good for comparing to human
survey means but can hide small steering effects near confident answers. That is why the ordinal
steering readout is the rank-centered logit contrast:

$$C = \sum_{k=1}^{M} \left(k - \frac{M + 1}{2}\right)\ell_k$$

The weights sum to zero, so any constant shift in the logprobs cancels. $C$ is invariant to
renormalizing over the allowed answer tokens and linear in logprob changes:

$$\Delta C = \sum_{k=1}^{M} \left(k - \frac{M + 1}{2}\right)(\ell^{\mathrm{steer}}_k - \ell^{\mathrm{base}}_k)$$

For MFV, the equivalent sensitive readout is the paired change in foundation logit:

$$\Delta_f(i,c) = \mathrm{logit}\,p^{\mathrm{steer}}_{i,c,f} - \mathrm{logit}\,p^{\mathrm{base}}_{i,c,f}$$

averaged over vignette $i$ and condition $c$. Positive means the steer made the model more likely to
call foundation $f$ the violation; negative means less likely.

`logodds_agree` is the easier ordinal direction summary:

$$\mathrm{logodds\_agree} =
\log\sum_{k \in \mathrm{agree}} \exp(\ell_k)
-
\log\sum_{k \in \mathrm{disagree}} \exp(\ell_k)$$

It drops the neutral middle option on odd scales. It is more readable than $C$, but it throws away
rank information.

Full return schemas live in the `TypedDict`s in `src/tinymfv/eval.py` and
`src/tinymfv/administer.py`.

## Steering plots

The per-instrument range and map figures are at the top of this README. A stricter, thresholded flip
metric, `si_per_foundation`, lives in steering-lite: did the steer move rows toward the intended
foundation at `+C` and away at `-C`, while penalizing rows that were already right and got broken?

$$\mathrm{SI} = \mathrm{mean}(\mathrm{SI_{fwd}}, \mathrm{SI_{rev}}) \times \text{pmass\_scale}$$

Use delta logit or delta $C$ for effect size. Use SI only when you care about thresholded decisions.

## Design notes

- Logprobs, not sampled answers. Prefill the answer slot, read the next-token distribution; small
  interventions register in nats before changing an argmax.
- Position-bias control. Each row is scored twice (options forward and reversed) and the logprob
  vectors averaged, cancelling option-order effects ([Pezeshkpour & Hruschka 2023](https://arxiv.org/abs/2308.11483)).
- A sliding think budget. `max_think_tokens` (0 / 64 dev default / 4096 / unbounded) is a setting you
  sweep: steering accrues over the think trace, so the same vector moves the profile more with more
  think. The model's end-of-answer tokens (eos and `</think>`) are suppressed over the budget, so it
  cannot self-close mid-trace into an unscorable state; every read is taken at the same forced slot,
  comparable across base and steered, instead of the old readout collapse once the model self-closed.
- Two modes. dev (N=1, greedy, 64 think) is fast and granular, the default. full (N=4 sampled traces
  per ordering, Bayesian-model-averaged, high think) adds sampling-variance uncertainty and SI.

## Instruments

The reader is answer-space-agnostic: it gathers logprobs over answer tokens at a prefilled slot
(`src/tinymfv/instrument.py`). Ordinal instruments read a 1..M Likert point and reduce to a keyed
expected score `E`, the rank-centered logit contrast `C`, `logodds_agree`, entropy, and
`pmass_allowed`; the nominal MFV vignettes read a foundation category and reduce to the log-prob over
foundations (`dlogit`).

"Ways asked" is the per-item debias: ordinal items are presented in three reworded framings
(forward / scale-inverted / content-negated, canonicalized and averaged, which cancels acquiescence);
MFV runs two option-order passes (forward / reversed enumeration, which cancels position bias).

| instrument | items | ways asked | measure | per-respondent | per-country | per-item human | source |
| :--- | ---: | :--- | :--- | :--- | ---: | :--- | :--- |
| MFQ-2 (Moral Foundations) | 36 | 3 reworded framings | log-odds + contrast `C`, 1-5 Likert | yes (raw) | 19 | no | Atari et al. 2023 |
| Big Five | 50 | 3 reworded framings | log-odds + contrast `C`, 1-5 Likert | no | 24 | no | BFI (country source not recorded in-repo) |
| 16PF | 162 | 3 reworded framings | log-odds + contrast `C`, 1-5 Likert | no | 34 | no | Cattell 16PF (country source not recorded) |
| Humor Styles | 32 | 3 reworded framings | log-odds + contrast `C`, model 1-5 / human 1-7 | no | 28 | no | Martin et al. 2003 |
| MFV (Moral Foundations Vignettes) | 132 | 2 option-order passes | log-prob over 7-way categorical (`dlogit`) | no | 5 | yes (per-vignette) | Clifford et al. 2015; norms JimenezLeal2025 + Yamada2025 |

Items is the unique-item count; ordinal items are each scored x3 framings. Per-respondent = raw
individual human data is bundled (MFQ-2 ships Atari et al. Study 2 respondents, used for the SPLOM and
the PCA basis); per-country = number of societies with published mean+sd; per-item human = per-question
human distribution (only the MFV vignettes carry Clifford's per-vignette ratings).

The bundled public showcase figures (per instrument, identical layout) are:

- `docs/img/showcase/<instr>/map_pca_ipsative.png`: culture map, model base and steer poles against
  human societies.
- `docs/img/showcase/<instr>/range.png`: per-factor human ranges beside the model steer path.

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
