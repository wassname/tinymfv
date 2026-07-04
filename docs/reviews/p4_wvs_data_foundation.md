# P4 data foundation: GlobalOpinionQA WVS subset for a WVS/Inglehart-Welzel map

Command: `uv run python scripts/probe_wvs_foundation.py`. Date: 2026-07-04.

`Anthropic/llm_global_opinions` is MC questions with per-country human answer distributions -- the
same shape tinymfv already reads (allowed answer tokens + human anchors), so it drops into our
reader + ipsative-map machinery.

Coverage of the WVS subset:
- source split: GAS (Pew) 2203, WVS 353.
- WVS questions: 353. Distinct countries: 90 (the Economist chart uses 88).
- countries/question: min 1, median 63, max 90. 212 questions have >=40 countries.
- top-coverage countries: Kenya, Zimbabwe, Ethiopia, Lebanon, Jordan, Iraq, Nigeria, South Korea,
  Puerto Rico, Mexico, Malaysia, Colombia (good IW-zone spread).
- `selections` ships as a `defaultdict(...)` repr string; parse recipe is
  `ast.literal_eval(re.search(r"\{.*\}", s, re.S).group(0))` (see probe_wvs_foundation.py).
- `options` includes non-substantive tails ("Don't know", "No answer") that must be dropped or held
  out of the answer space before renormalizing.

Verdict: dense enough (90 countries x 353 questions, 212 questions with >=40 countries) to place
models among human societies at Economist scale. This is the human anchor; the model side reuses the
logprob reader (open models) or read_api sampling reader (frontier models).

## Open research fork (needs a human steer before spending model-run compute)

How to define the two map axes -- this is a research-validity choice, not a mechanical one, so I am
not guessing it:

- (4a) Literal Inglehart-Welzel: use the 10 documented IW questions (Traditional: Q164, Q7-17, Q184,
  Q254, Q4; Survival: Q154-155, Q46, Q182, Q209, Q57), factor-analyze to the two published axes,
  place countries and models on them. Most faithful to the Economist, but fragile: the 10 exact
  questions may not all be present in GlobalOpinionQA with consistent country coverage, and it
  hard-codes Inglehart-Welzel's factor structure.
- (4b) Shared-question ipsative PCA (our existing map method): pick the WVS questions with dense
  country coverage, build a country x question matrix, ipsative-PCA to 2 PCs (reusing
  maps.ipsative_pca), and project models administered the same questions. Drops straight into the
  P1 hull code. Axes are data-derived (not the literal IW Traditional/Survival), so the chart is
  "WVS-values map" not a verbatim Economist reproduction.

Also to confirm before spending: which model set goes on the map (open-weight via the logprob reader
vs frontier via read_api sampling), and the country subset (all 90 vs the >=40-coverage 212-question
core).
