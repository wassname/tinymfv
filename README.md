# tinymfv

tinymfv is a small set of quick value evals for local LLM steering work. It is sensitive to small answer-probability shifts, so you can see movement before sampled answers flip.

It asks moral vignettes and survey questions, reads the model's answer probabilities, and compares the model profile to human responses. The main use case is simple: after you steer a model, did the intended values move, did nearby values move too, and does the result still look like a coherent answer?

The plots are the point. Range plots show where the model sits relative to human societies. Culture maps show the base model and steered models on a PCA map of human response profiles.

![MFQ-2 range plot: human society ranges beside the fairness steer path](docs/img/showcase/mfq2/range.png)

![MFV culture map: base and fairness-steered points against human countries](docs/img/showcase/mfv/map_pca_ipsative.png)

![MFV range plot: Care moves most under the fairness vector](docs/img/showcase/mfv/range.png)

More maps are in `docs/img/showcase/` for MFQ-2, Big Five, 16PF, and Humor Styles. The plotting script now defaults to clean base / +C / -C maps; use `--show-sweep` when you want the diagnostic `c=-4..+4` trajectory.

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

| dataset | what it asks | model answer | human comparison |
|---|---|---|---|
| MFV classic | 132 Moral Foundations Vignettes from Clifford et al. | one of 7 foundations: Care, Fairness, Loyalty, Authority, Sanctity, Liberty, Social Norms | per-vignette human foundation labels |
| MFV scifi | the same MFV items rewritten as sci-fi scenarios | one of 7 foundations | inherited labels from the classic item |
| MFV ai-actor | the same MFV items rewritten so an AI system is the actor | one of 7 foundations | inherited labels from the classic item |
| MFQ-2 | 36 Moral Foundations Questionnaire items | 1-5 agreement | country foundation means, plus raw Atari et al. respondent data |
| Big Five | 50 personality items | 1-5 agreement | country factor means |
| 16PF | 162 personality items | 1-5 agreement | country factor means |
| Humor Styles | 32 humor-style items | 1-5 agreement | country style means, originally on a 1-7 scale |

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
report = evaluate(model, tok, vignettes=vignettes, return_per_row=True)

print(report["profile"])              # mean probability per foundation
print(report["mean_pmass_allowed"])   # probability mass on valid answer tokens
print(report["per_row"][0]["score"])  # foundation logprobs, in nats
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
print(report["profile_E"])                # human-comparable scale means
print(report["profile_C"])                # steering-sensitive log contrast
print(report["per_item_frame"][0]["lp"])  # raw answer-token logprobs
```

## Metrics

There are two metric families.

Human-comparison metrics are bounded and easy to plot against survey data.

For MFV, the profile is the mean probability of each foundation:

$$p_f = \mathbb{E}_i\,P(f \mid i)$$

For surveys, `profile_E` is the expected scale score, reverse-keyed where needed:

$$E_i = \sum_{k=1}^{M} k p_{i,k}$$

Steering metrics are logprob based. They are more sensitive because they move before the sampled answer flips.

At the answer slot, tinymfv gathers the logprobs for the allowed answers:

$$\ell_k = \log P(a_k \mid \mathrm{prompt}, \mathrm{think}, \mathrm{prefill})$$

For surveys, `profile_C` is the steering-sensitive direction score. It weights high agreement tokens positive and low agreement tokens negative, using logprobs instead of bounded survey means:

$$C_i = \sum_{k=1}^{M} \left(k - \frac{M + 1}{2}\right)\ell_{i,k}$$

Use `delta C` for survey steering effects:

$$\Delta C_i = C_i^{\mathrm{steered}} - C_i^{\mathrm{base}}$$

For MFV, use the paired foundation logit change:

$$\Delta_{i,f} = \operatorname{logit} p_{i,f}^{\mathrm{steered}} - \operatorname{logit} p_{i,f}^{\mathrm{base}}$$

`pmass_allowed` is a format check, not a value score:

$$\mathrm{pmass}_{\mathrm{allowed}} = \sum_{k=1}^{K}\exp(\ell_k)$$

If it drops, the model is leaking probability into invalid answers. If `nll_prefill` rises, the forced answer scaffold itself no longer fits the model well.

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
