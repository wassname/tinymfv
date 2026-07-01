You are reading the document below for the FIRST time (a cold reader).
Answer ONLY from what it says; where something is unstated or ambiguous, say so.
Output ONE JSON object, no prose, no fences:
{
 "summary": "<2-3 sentences: restate the thesis/method in your OWN words>",
 "mechanism": "<the doc's single hardest/central mechanism, reconstructed in your OWN words as if to a colleague; say 'unclear' if the doc does not let you rebuild it>",
 "scores": {"clarity": "<1-5>", "conciseness": "<1-5>", "technical_accuracy": "<1-5>"},
 "reason": "<one sentence on the scores>",
 "unclear": ["<what was confusing, ambiguous, or you had to guess>"],
 "misunderstandings": ["<places the text contradicts itself or invites a misread>"],
 "missing_to_implement": ["<what a reader still needs to reproduce or act on this>"],
 "questions": ["<a question the doc left you with, then your best-guess answer>"],
 "suggestions": ["<concrete edit that would help>"],
 "rewrites": [{"section": "<quote a sentence/section that reads as AI-written>", "rewrite": "<rewrite it to strip the AI voice and read like a plain, direct human draft -- SAME meaning and facts, do not add claims>", "why": "<which AI tell it removes>"}]
}

DOCUMENT:
# tinymfv

tinymfv is a small set of fast value evals for local LLM steering work. It asks moral vignettes and survey questions, reads answer-token probabilities, and turns them into one model profile.

Use it when you want to know whether a steer moved the intended values, moved nearby values too, and still lands near real human response patterns. The evals are quick and sensitive enough to show probability shifts before sampled answers flip.

The plots compare that profile to human data. Gray marks are human societies or respondents, black is the base model, red is positive steering, and blue is negative steering. This showcase uses an Authority contrast vector built from `authority-respecting` versus `authority-disregarding` personas. steering-lite built and applied the vector; tinymfv measures the resulting model profiles. Here `c` is the steering-lite multiplier on that vector. Red is `+c`, more Authority; blue is `-c`, less Authority.

MFV comes first because it is the direct moral-vignette readout. MFV is nominal: the answer is a moral foundation category. MFQ-2 is the Moral Foundations Questionnaire 2 survey, where the answer is a 1-5 scale point.

![MFV culture map: Authority steering against human countries](docs/img/showcase/mfv/map_pca_ipsative.png)

![MFV range plot: foundation emphasis beside Authority steering](docs/img/showcase/mfv/range.png)

MFV uses the same map and range plotters as the surveys, after converting nominal foundation probabilities into relative foundation emphasis. Each profile is z-scored across foundations before mapping, so the plot compares which foundations are high or low within that profile.

![Humor Styles range plot: human society ranges beside Authority steering](docs/img/showcase/humor_styles/range.png)

![Humor Styles culture map: Authority steering against human societies](docs/img/showcase/humor_styles/map_pca_ipsative.png)

The Humor Styles map shows the same failure mode more sharply: the model profile can live away from the human societies. That is the useful warning sign, a model can answer in the requested format and still be a moral or psychological alien on the measured profile.

![Big Five range plot: human society ranges beside Authority steering](docs/img/showcase/big5/range.png)

![Big Five culture map: Authority steering against human societies](docs/img/showcase/big5/map_pca_ipsative.png)

Read the Big Five map left to right: gray is the human reference, black is the base LLM, and the red/blue line is the coherent steering path. Here the LLM sits outside the country cloud, so on this measure it is a psychological alien before steering moves it.

MFQ-2 means Moral Foundations Questionnaire 2, the short survey instrument. It is separate from MFV, the moral-vignette foundation reader. MFQ-2 has fewer items per axis than the longer personality surveys, so the showcase averages 8 sampled reads per item before treating small path wiggles as signal.

![MFQ-2 range plot: human society ranges beside Authority steering](docs/img/showcase/mfq2/range.png)

![MFQ-2 culture map: Authority steering against human societies](docs/img/showcase/mfq2/map_pca_ipsative.png)

The path shows only usable coefficients: `c=0`, then each positive and negative side until one of the plot gates fails. This run kept the full path `c=-1,-0.5,0,+0.5,+1`. For surveys, collapse can mean the answer distribution loses its factor structure even when answer mass stays high.

This is the same run as the plots. The contrast used to build the vector varied Authority; the table shows every measured axis that moved.

`profile shift / human SD` is the distance from `c=-1` to `c=+1`, divided by the standard deviation of country means in the bundled human reference for that axis. `100%` means one human SD. `profile shift` is in plot units: MFV relative-emphasis z-score, or survey expected 1-5 score. `reader-logit shift` is the direct answer-logit readout for the same endpoints: MFV uses the per-foundation logit change, surveys use the rank-logit contrast `C`. The `+/-` term is the propagated item uncertainty.

| dataset | axis | profile shift / human SD | profile shift | reader-logit shift |
| --- | --- | --- | --- | --- |
| MFV vignettes | Care | -242% | -0.59 | -0.07 +/- 0.28 |
| MFV vignettes | Sanctity | -9% | -0.05 | +0.53 +/- 0.21 |
| MFV vignettes | Authority | +401% | +1.17 | +1.82 +/- 0.32 |
| MFV vignettes | Loyalty | -18% | -0.05 | +0.55 +/- 0.18 |
| MFV vignettes | Fairness | -5% | -0.02 | +0.56 +/- 0.20 |
| MFV vignettes | Liberty | -75% | -0.46 | +0.11 +/- 0.20 |
| Humor Styles | affiliative | +144% | +0.40 | +15.00 +/- 3.98 |
| Humor Styles | selfenhancing | +162% | +0.25 | +13.52 +/- 4.86 |
| Humor Styles | aggressive | -23% | -0.04 | +0.08 +/- 7.58 |
| Humor Styles | selfdefeating | +32% | +0.06 | +13.62 +/- 6.55 |
| Big Five | extraversion | +105% | +0.17 | +7.21 +/- 4.79 |
| Big Five | neuroticism | +2% | +0.00 | +7.69 +/- 3.56 |
| Big Five | agreeableness | +269% | +0.34 | +16.03 +/- 4.17 |
| Big Five | conscientiousness | +435% | +0.45 | +17.21 +/- 4.03 |
| Big Five | openness | +41% | +0.07 | +17.77 +/- 2.48 |
| MFQ-2 survey | care | +305% | +0.93 | +23.20 +/- 6.08 |
| MFQ-2 survey | equality | +21% | +0.07 | +16.06 +/- 7.89 |
| MFQ-2 survey | proportionality | +270% | +0.77 | +24.48 +/- 5.87 |
| MFQ-2 survey | loyalty | +202% | +0.82 | +25.88 +/- 6.33 |
| MFQ-2 survey | authority | +338% | +1.17 | +29.73 +/- 2.83 |
| MFQ-2 survey | purity | +137% | +0.74 | +23.53 +/- 8.02 |

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

## Datasets

| dataset | bundled data | human reference | profile used in plots |
|---|---|---|---|
| MFV classic | [132 moral vignettes, other](src/tinymfv/data/vignettes_classic_other_violate.jsonl) / [self](src/tinymfv/data/vignettes_classic_self_violate.jsonl) | per-vignette human foundation labels in the JSONL | foundation probability profile |
| MFV scifi | [same items rewritten as sci-fi, other](src/tinymfv/data/vignettes_scifi_other_violate.jsonl) / [self](src/tinymfv/data/vignettes_scifi_self_violate.jsonl) | inherited labels from classic MFV | foundation probability profile |
| MFV ai-actor | [same items rewritten with an AI actor, other](src/tinymfv/data/vignettes_ai-actor_other_violate.jsonl) / [self](src/tinymfv/data/vignettes_ai-actor_self_violate.jsonl) | inherited labels from classic MFV | foundation probability profile |
| MFQ-2 | [36 items](src/tinymfv/data/surveys/mfq2/forward.json), plus inverted and negated frames | [country means](src/tinymfv/data/human/mfq2_country_foundations.csv), plus [raw respondents](src/tinymfv/data/atari_study2_raw.csv) | expected 1-5 score per foundation |
| Big Five | [50 items](src/tinymfv/data/surveys/big5/questionnaire.json), plus inverted and negated frames | [country means](src/tinymfv/data/human/big5_country_factors.csv) | expected 1-5 score per trait |
| 16PF | [162 items](src/tinymfv/data/surveys/16pf/questionnaire.json), plus inverted and negated frames | [country means](src/tinymfv/data/human/16pf_country_factors.csv) | expected 1-5 score per factor |
| Humor Styles | [32 items](src/tinymfv/data/surveys/humor_styles/questionnaire.json), plus inverted and negated frames | [country means](src/tinymfv/data/human/humor_styles_country_factors.csv), originally 1-7 | expected 1-5 score per style |

MFV is nominal: the answer is the category. The survey instruments are ordinal: the answer is a scale point.

Each MFV item is asked in two perspectives, `other_violate` and `self_violate`. Each survey item is asked three ways, forward, scale-inverted, and content-negated. tinymfv canonicalizes these frames before averaging, so the profile is less tied to one wording.

## API

Run MFV vignettes with `evaluate`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from tinymfv import evaluate, load_vignettes

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B").cuda()

vignettes = load_vignettes("classic")  # "classic", "scifi", "ai-actor", or "all"
report = evaluate(model, tok, vignettes=vignettes)

print(report["profile"])              # mean probability per foundation
print(report["mean_pmass_allowed"])   # format check: mass on valid answer tokens
```

Run survey instruments with `administer`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from tinymfv import administer, get_instrument

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B").cuda()

instr = get_instrument("mfq2")  # "mfq2", "big5", "16pf", or "humor_styles"
report = administer(model, tok, instr)

print(report["dimensions"])
print(report["profile"])                  # expected 1-5 score per factor
print(report["mean_pmass_allowed"])       # format check: mass on valid answer tokens
```

Generate the bundled range plots and culture maps from a steering-lite all-instrument run:

```bash
uv run python scripts/plot_steer_showcase.py \
  --run-dir ../steering-lite/outputs/20260630T222000Z_pure_authority_mundane15_pca_readme_mfv_mfq2_humor_big5_n8 \
  --out docs/img/showcase \
  --vec-label "Authority contrast, PCA (+c = more Authority)" \
  --coherence-frac 0.99 \
  --contrast-frac 0.000001 \
  --margin-frac 0.50
```

The plot gate keeps only coefficients that all plotted instruments can still read. `coherence-frac` checks answer mass against base. `contrast-frac` checks that survey answers still have rank-logit structure rather than flattening. `margin-frac` checks that MFV still has a top foundation separated from the runner-up.

## Measurement

The measurement on the maps is the profile.

For MFV, the profile is the model's mean probability on each moral foundation:

$$\mathrm{profile}_f = \mathbb{E}_i P(f \mid i)$$

For survey instruments, the profile is the mean expected 1-5 answer for each factor, after reverse-keying:

$$\mathrm{profile}_d = \mathbb{E}_{i \in d}\sum_{k=1}^{M} k P(k \mid i)$$

where $i$ is an item, $d$ is a survey factor, $k$ is a scale point, and $M$ is the largest scale value.

This is what the survey maps and range plots show. In the showcase CSVs, this is the `mean` column. For MFV showcase plots, model and human units differ, so the plotted quantity is relative foundation emphasis: each foundation profile is z-scored across foundations before mapping.

The table's reader-logit shift uses a more sensitive log-space readout.

For MFV:

$$\Delta_f = \mathbb{E}_i \left(\ell_{i,f}^{(+1)} - \ell_{i,f}^{(-1)}\right)$$

where $\ell_{i,f}^{(c)}$ is the forced-choice logit for foundation $f$ on item $i$ at coefficient $c$.

For survey instruments:

$$C_d(c) = \mathbb{E}_{i \in d}\sum_{k=1}^{M} \left(k - \frac{M+1}{2}\right)\ell_{i,k}^{(c)}$$

and the table reports $C_d(+1)-C_d(-1)$.

For paired steering runs, compare the base profile to the steered profile path. Answer mass is a coherence check, not a value score:

$$m(c) = \mathbb{E}_i \sum_{a \in A_i} P_c(a \mid i)$$

where $A_i$ is the valid answer-token set for item $i$. The showcase also checks survey rank-logit contrast and MFV top-foundation margin, because a steered reader can keep answer mass while losing useful structure.

## Scope

tinymfv is for fast paired steering comparisons, not full moral reasoning evaluation. It is useful when you want to compare base, positive-steer, and negative-steer runs against the same human reference plots.

For behavior-heavy moral evals, see [machiavelli](https://huggingface.co/datasets/wassname/machiavelli), [AIRiskDilemmas](https://huggingface.co/datasets/kellycyy/AIRiskDilemmas), and [ethics_expression_preferences](https://huggingface.co/datasets/wassname/ethics_expression_preferences).

Used in [steering-lite](https://github.com/wassname/steering-lite), [lora-lite](https://github.com/wassname/lora-lite), and [w2schar-mini](https://github.com/wassname/w2schar-mini).

## Citation

```bibtex
@misc{clark2026tinymfv,
  title = {tinymfv: tiny moral/value eval for local LLMs},
  author = {Michael Clark},
  year = {2026},
  url = {https://github.com/wassname/tinymfv/}
}
```
