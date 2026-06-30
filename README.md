# tinymfv

tinymfv is a small set of fast value evals for local LLM steering work. It asks moral vignettes and survey questions, reads answer-token probabilities, and turns them into one model profile.

Use it when you want to know whether a steer moved the intended values, moved nearby values too, and still lands near real human response patterns. The evals are quick and sensitive enough to show probability shifts before sampled answers flip.

The plots compare that profile to human data. Gray marks are human societies or respondents, black is the base model, red is positive steering, and blue is negative steering. In this README run, `+c` is oriented by a declared MFV Authority anchor. In other runs, `c` is just the signed vector coefficient unless the run declares its own anchor.

MFV comes first because it is the direct moral-vignette readout. MFV is nominal: the answer is a moral foundation category. The survey plots are ordinal: the answer is a 1-5 scale point.

![MFV culture map: dignity/authority steering against human countries](docs/img/showcase/mfv/map_pca_ipsative.png)

![MFV range plot: foundation emphasis beside dignity/authority steering](docs/img/showcase/mfv/range.png)

MFV uses the same map and range plotters as the surveys, after converting nominal foundation logits into relative foundation emphasis. Each profile is z-scored across foundations before mapping, so the plot compares which foundations are high or low within that profile.

![Humor Styles range plot: human society ranges beside dignity/authority steering](docs/img/showcase/humor_styles/range.png)

![Humor Styles culture map: dignity/authority steering against human societies](docs/img/showcase/humor_styles/map_pca_ipsative.png)

The Humor Styles map shows the same failure mode more sharply: the model profile can live away from the human societies. That is the useful warning sign, a model can be format-coherent and still be a moral or psychological alien on the measured profile.

![Big Five range plot: human society ranges beside dignity/authority steering](docs/img/showcase/big5/range.png)

![Big Five culture map: dignity/authority steering against human societies](docs/img/showcase/big5/map_pca_ipsative.png)

Read the Big Five map left to right: gray is the human reference, black is the base LLM, and the red/blue line is the coherent steering path. Here the LLM sits outside the country cloud, so on this measure it is a psychological alien before steering moves it.

MFQ-2 means Moral Foundations Questionnaire 2, the short survey instrument. It is separate from MFV, the moral-vignette foundation reader. MFQ-2 has fewer items per axis than the longer personality surveys, so the showcase uses sampled think traces before treating small path wiggles as signal.

![MFQ-2 range plot: human society ranges beside dignity/authority steering](docs/img/showcase/mfq2/range.png)

![MFQ-2 culture map: dignity/authority steering against human societies](docs/img/showcase/mfq2/map_pca_ipsative.png)

The path shows only usable coefficients: `c=0`, then each positive and negative side until the reader starts to collapse. For surveys, collapse can mean the answer distribution loses its factor structure even when answer mass stays high.

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
  --run-dir ../steering-lite/outputs/20260630_dignity_authority_strict22_local_sspace_mfvgrid_n8 \
  --out docs/img/showcase \
  --vec-label="MFV Authority anchor (+c intended higher Authority)" \
  --coherence-frac 0.99 \
  --contrast-frac 0.50 \
  --margin-frac 0.50
```

## Measurement

The measurement on the maps is the profile.

For MFV, the profile is the model's mean probability on each moral foundation:

$$\mathrm{profile}_f = \mathbb{E}_i P(f \mid i)$$

For survey instruments, the profile is the mean expected 1-5 answer for each factor, after reverse-keying:

$$\mathrm{profile}_d = \mathbb{E}_{i \in d}\sum_{k=1}^{M} k P(k \mid i)$$

where $i$ is an item, $d$ is a survey factor, $k$ is a scale point, and $M$ is the largest scale value.

This is what the survey maps and range plots show. In the showcase CSVs, this is the `mean` column. For MFV showcase plots, model and human units differ, so the plotted quantity is relative foundation emphasis: each foundation profile is z-scored across foundations before mapping.

For paired steering runs, compare the base profile to the steered profile path. Answer mass is a coherence check, not a value score:

$$m(c) = \mathbb{E}_i \sum_{a \in A_i} P_c(a \mid i)$$

where $A_i$ is the valid answer-token set for item $i$. The showcase also checks survey contrast and MFV forced-choice margin, because a steered reader can keep answer mass while losing useful structure.

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
