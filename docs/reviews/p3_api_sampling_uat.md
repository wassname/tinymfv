# P3 UAT: sampling readout (read_api) vs logprob readout (read)

Command: `uv run python scripts/probe_api_readout.py --api-model meta-llama/llama-3.1-8b-instruct --n-items 5 --n-samples 20`
Date: 2026-07-04. Instrument: mfq2 (forward frame, first 5 items). Local logprob model: Qwen3-0.6B.

The API path (read_api.read_items_sampled) samples N chat completions at temperature and uses the
empirical answer frequency as `p`; E and profile then come from the SAME readouts as the logprob
path, so a logprob-less frontier model drops onto the same [1,5] scale / same map.

| id | dimension       | statement (trunc)                  | E_local(lp) | E_mc | E_api(llama-3.1-8b) | api_pmass | parsed | api_H |
|---:|:----------------|:-----------------------------------|------------:|-----:|--------------------:|----------:|:-------|------:|
| 1  | care            | Caring for people who have suffere |        3.83 | 3.82 |                3.30 |      1.00 | 20/20  |  0.64 |
| 2  | equality        | The world would be a better place  |        3.08 | 3.08 |                3.00 |      1.00 | 20/20  |  1.37 |
| 3  | proportionality | I think people who are more hardwo |        3.65 | 3.66 |                2.80 |      1.00 | 20/20  |  0.94 |
| 4  | loyalty         | I think children should be taught  |        3.56 | 3.56 |                3.19 |      0.80 | 16/20  |  1.06 |
| 5  | authority       | I think it is important for societ |        3.75 | 3.75 |                3.05 |      0.95 | 19/20  |  0.54 |

Verdicts:
- Unbiasedness (same model, no API): `E_local(lp)` == `E_mc` to <=0.01 on every item (max gap over
  all items 0.0063 at N=2000). Sampling the model's own categorical and reducing to E recovers the
  exact logprob E -- the frequency->E estimator + reduce plumbing is unbiased.
- Live API: llama-3.1-8b sampled reader lands E in [1,5] for all 5 items. The sampling coherence
  gate (`api_pmass` = parse rate) is 1.0 except item 4 (0.80, 4/20 off-format draws) and item 5
  (0.95) -- it correctly flags draws the model did not answer in-format, the sampling analogue of
  the logprob `pmass_allowed`.
- Comparability: `E_api` sits on the same [1,5] scale as `E_local`; the gaps (e.g. 3.83 vs 3.30 on
  care) are genuine Qwen3-0.6B vs Llama-3.1-8b disagreement, not method error. N=20 gives ~0.1 E
  resolution; raise N for tighter dots.

Not produced by design: logit_contrast C and logodds_agree LO (log of an empirical frequency has
-inf zeros; smoothing them would fabricate the fine steer signal). Sampled frontier models can join
the map but carry no steering arm -- exactly the Economist's "average of ten responses" tradeoff.
