# tinymfv

tinymfv is a small set of fast value evals for local LLM steering work. It asks moral vignettes and survey questions, reads answer-token probabilities, and turns them into model profiles that you can compare to humans.

When comparing models or checkpoints you can use it to check three things: did the intended value move?, what else moved?, how does this compare to human responses? The evals are quick and sensitive enough to show probability shifts.

## The plots - Are model moral aliens?

One interesting thing we can do with this repo is compare AI's models on human psychological and anthropological surveys. Are they like us?

One thing jumps out of the plots below. Before any steering the base model is already a psychological alien on some axes: on the culture maps it sits away from most humans, most sharply on Big5 openness (low) and conscientiousness (high), and on humor style (high aggressive, low affiliative). And steering is strong relative to human variation, on the headline axes a single sweep moves the profile further than any two countries differ, not just a US-vs-Australia nudge.

### The whole field, on the world's value map

The clearest single answer comes from the World Values Survey, the standard culture map of the world. Since 1981 it has asked people in about ninety countries the same questions, and two axes drawn from it sort societies by how traditional or secular they are and how much they weigh survival over self-expression. We put seventeen frontier models through the same questions.

![WVS culture map: 17 frontier models among about 90 human societies](docs/img/wvs/wvs_map_iw.png)

Every model lands in the top-left: more secular and more self-expressive than almost any country on earth, deep in the rich-world corner and often past its edge, and none of them sits near the African or Muslim societies. This is a different reading from the rest of the page. These frontier models are closed APIs with no answer probabilities to read, so each is scored by rated sampling (rate every option one to five, twelve times, with the option order shuffled), and the human positions are approximated from the GlobalOpinionQA question set. The steering plots below instead follow one open model we can push, Qwen3-4B.

What happens when we steer them? Below we steer models with `authority-respecting` versus `authority-disregarding` personas.

### Value maps: where a model sits, on named axes

The clearest view is the value ("quadrant") map. Each has two named axes borrowed from the psychology that built the survey, the human societies drawn as cultural regions, and the model as a black dot with a coloured path showing where steering takes it. Steering here means nudging the model's internal state toward the authority-respecting side (red, more Authority) or away from it (blue, less), without retraining. Every map keeps one orientation, the cultural West to the west and the global South to the south, so they all read the same way.

![MFQ-2 value map: individual-first vs group-first morality, with the Authority steer path](docs/img/showcase/mfq2/map_value.png)

Moral-foundations theory (Jonathan Haidt's) holds that our moral sense runs on a few basic concerns: caring for others, fairness, loyalty to the group, respect for authority, and a sense of the sacred. The MFQ-2 survey scores a person, or a model, on each. On this map, left to right runs from an individual-first morality (care, equality) to a group-first one (loyalty, authority, purity); bottom to top splits fairness into equal-shares versus earned-shares. The base model sits in the Western, individual-first corner, and pushing it toward Authority walks it clear across to the group-first corner shared by the African-Islamic and East-Asian societies.

![Big Five value map: outgoing/open vs even-keeled axes, with the Authority steer path](docs/img/showcase/big5/map_value.png)

Big Five personality collapses to two broad traits: how outgoing and open a person is (reserved to exploratory, left to right) and how even-keeled they are (volatile to stable, bottom to top). The Authority push barely moves the base model here, which is the point: it shifts values, not personality.

![Humor Styles value map: adaptive vs maladaptive humor, with the Authority steer path](docs/img/showcase/humor_styles/map_value.png)

Humor is the honest negative result. On its axes (warm, healthy humor versus put-down humor; joking at yourself versus at others) the human regions overlap almost completely: humor style does not sort societies the way values do. Worth knowing a survey can't tell societies apart at all before reading anything into a steer on it.

### Range plots: one factor at a time

A range plot takes an instrument one factor at a time: the spread of human societies is a grey strip, their middle a black line, and the steer a red-to-blue sweep, so even a small model move stays visible against the whole human range.

![MFV range plot: foundation emphasis beside Authority steering](docs/img/showcase/mfv/range.png)

MFV (moral-foundation vignettes, the repo's namesake) hands the model a short story about someone breaking a moral rule and asks which kind of wrong it is: cruelty, cheating, betrayal, defiance of authority, or defiling the sacred. Pushed toward Authority, the model does just what you would hope, it calls out the authority violations far more often and the others less. The move is large: about as wide as the gap between the least and the most authority-minded societies on earth, not a nudge.

![MFQ-2 range plot: human society ranges beside Authority steering](docs/img/showcase/mfq2/range.png)

![Big Five range plot: human society ranges beside Authority steering](docs/img/showcase/big5/range.png)

![Humor Styles range plot: human society ranges beside Authority steering](docs/img/showcase/humor_styles/range.png)

The surveys echo their maps: MFQ-2's binding factors climb under the steer, while Big Five and humor stay flat.

### Maps with data-picked axes

When a survey has no ready-made named axes (MFV), or just as a cross-check on the value maps, we let the data pick the axes instead: find the two directions along which human societies differ most, and place the model in them. A small compass shows which traits each axis is built from, and an inset shows where the zoomed-in frame sits within the full crowd of human respondents.

![MFV culture map: Authority steering against human countries](docs/img/showcase/mfv/map_pca_ipsative.png)

![MFQ-2 culture map: Authority steering against human societies](docs/img/showcase/mfq2/map_pca_ipsative.png)

![Big Five culture map: Authority steering against human societies](docs/img/showcase/big5/map_pca_ipsative.png)

![Humor Styles culture map: Authority steering against human societies](docs/img/showcase/humor_styles/map_pca_ipsative.png)

The story survives without the named axes: on Big Five and humor the base model already sits outside the human crowd before any steering, an alien on the measured profile. For MFV each profile is read relative to the model's own average, so the map shows which foundation a model leans on more than usual, not how much it endorses everything.


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
| MFV classic | [132 moral vignettes, other](src/tinymfv/data/vignettes_classic_other_violate.jsonl) / [self](src/tinymfv/data/vignettes_classic_self_violate.jsonl) | per-vignette human foundation labels in the JSONL | forced-choice foundation probability profile |
| MFV scifi | [same items rewritten as sci-fi, other](src/tinymfv/data/vignettes_scifi_other_violate.jsonl) / [self](src/tinymfv/data/vignettes_scifi_self_violate.jsonl) | inherited labels from classic MFV | forced-choice foundation probability profile |
| MFV ai-actor | [same items rewritten with an AI actor, other](src/tinymfv/data/vignettes_ai-actor_other_violate.jsonl) / [self](src/tinymfv/data/vignettes_ai-actor_self_violate.jsonl) | inherited labels from classic MFV | forced-choice foundation probability profile |
| MFQ-2 | [36 items](src/tinymfv/data/surveys/mfq2/forward.json), plus inverted and negated frames | [country means](src/tinymfv/data/human/mfq2_country_foundations.csv), plus [raw respondents](src/tinymfv/data/atari_study2_raw.csv) | expected 1-5 score per foundation |
| Big Five | [50 items](src/tinymfv/data/surveys/big5/questionnaire.json), plus inverted and negated frames | [country means](src/tinymfv/data/human/big5_country_factors.csv) | expected 1-5 score per trait |
| 16PF | [162 items](src/tinymfv/data/surveys/16pf/questionnaire.json), plus inverted and negated frames | [country means](src/tinymfv/data/human/16pf_country_factors.csv) | expected 1-5 score per factor |
| Humor Styles | [32 items](src/tinymfv/data/surveys/humor_styles/questionnaire.json), plus inverted and negated frames | [country means](src/tinymfv/data/human/humor_styles_country_factors.csv), originally 1-7 | expected 1-5 score per style |

MFV uses categorical answers: the answer is the foundation. The survey instruments use ordinal answers: the answer is a scale point.

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

print(report["profile"])              # mean forced-choice probability per foundation
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
  --vec-label "Authority steer, PCA (+c = more Authority)" \
  --coherence-frac 0.99 \
  --contrast-frac 0.000001 \
  --margin-frac 0.50
```

The plot gate keeps only coefficients that all plotted instruments can still read. A row passes when answer mass, survey rank-logit contrast, and MFV top-foundation margin stay above the requested fraction of their base values: `pmass(c)/pmass(0) >= coherence-frac`, `mean_abs_C(c)/mean_abs_C(0) >= contrast-frac`, and `mean_margin(c)/mean_margin(0) >= margin-frac`.

## Measurement

The measurement on the maps is the profile.

For MFV, the profile is the model's mean forced-choice probability on each moral foundation:

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
