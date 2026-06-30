# README plot and metric simplification

## Goal
Make the README readable to a new researcher in one pass. It should say what tinymfv is, what datasets it includes, what single measurement to look at first, and how the plots encode base/+C/-C steering.

## Scope
In: README wording, metric/API simplification, regenerated showcase plots, cold-reader panel review.
Out: changing eval semantics, changing dataset schemas, adding new metrics.

## Requirements
- R1: README has one headline measurement: `profile`. Done means: the Metrics section defines the profile once and treats logprobs/format checks as API details, not competing headline metrics. VERIFY: `rg -n '^###|profile_C|informedness|nll_prefill|pmass_allowed|dlogit_per_foundation' README.md`.
- R2: README explains the plot encoding. Done means: a reader can tell what gray, black, red, and blue marks mean without knowing steering-lite internals. VERIFY: README contains the color/key sentence and no stale "trajectory" caption for the clean maps.
- R3: README still lists all datasets and all image links resolve. VERIFY: a script parses markdown image links and prints `missing_images: []`.
- R4: Cold readers can reconstruct what tinymfv is, which datasets exist, what measurement matters, and why a researcher would use it. VERIFY: external-review-v2 panel JSON files in `docs/reviews/` plus a short triage table in this spec.
- R5: README survey maps show the coherent steering path, not just two endpoints. Done means: MFQ-2, Big Five, and Humor maps draw base -> tested coherent c values, with constant-size c markers and no incoherent lone dots. VERIFY: inspect regenerated PNGs and check the plotter passes `traj=` to `plot_ipsative_pca`.
- R6: README range plots show the coherent coefficient path in sign lanes. Done means: all negative c values share a blue lane, base is black in the middle, and all positive c values share a red lane. Reversals/wraps are visible on the y-axis without marker-size encoding. VERIFY: inspect regenerated range PNGs and check `draw_steer` maps x by sign only.

## Tasks
- [x] T1 (R1, R2): Simplify README around `profile` and update plot captions.
  - verify: `rg -n '^###|profile_C|informedness|nll_prefill|pmass_allowed|dlogit_per_foundation|trajectory' README.md`
  - success: one `### The profile` subsection; no stale plot "trajectory" language.
  - likely_fail: API section still presents six metrics as peers.
  - sneaky_fail: profile is too vague to connect to MFV vs survey data; caught by panel mechanism answers.
- [/] T2 (R2, R3): Keep regenerated showcase plots and verify image refs.
  - verify: parse README image links and check paths exist.
  - success: `missing_images: []`.
  - likely_fail: stale filenames remain.
  - sneaky_fail: images exist but show old sweep encoding; caught by fresh image inspection of MFQ-2 map/range.
- [ ] T3 (R4): Run external-review-v2 comprehension panel and triage.
  - verify: panel outputs valid JSON and summaries answer the probe.
  - success: most panel members correctly identify tinymfv, datasets, `profile`, and researcher use.
  - likely_fail: models still name `profile_C` or `logprobs` as the main metric.
  - sneaky_fail: models parrot phrases without explaining why a researcher would use it; caught by the "own words with inference" probe.
- [/] T4 (R5, R6): Restore coherent c paths while removing misleading geometry.
  - steps: compute one shared answer-mass gate across survey evals; pass the shared coherent path to map plots; make range steer paths use sign lanes.
  - verify: `rg -n "traj=|np.sign\\(c\\)|s=42" scripts/plot_steer_showcase.py src/tinymfv/maps.py`
  - success: maps show red/blue coherent paths through c values; ranges show AI sign-lane paths.
  - likely_fail: maps still show only endpoint dots.
  - sneaky_fail: marker size encodes c, or x offset varies by c magnitude; caught by the source check and image inspection.
  - UAT: when opening `docs/img/showcase/mfq2/map_pca_ipsative.png` and `docs/img/showcase/mfq2/range.png`, the path is visible and the range path can show reversals/wraps.

## Log
- 2026-06-30: The README previously exposed many metric names at once. That made the API complete but caused cold-reader confusion about which measurement to start with.
- 2026-06-30: Final plots are gated on steering-lite task 394, output `outputs/20260630_dignity_authority_strict22_local_sspace_allinstr`. Stale README images from the earlier fairness / two-axis run must not count as UAT.
- 2026-06-30: Coherence gating must be a signed prefix from `c=0`, stopping a side at the first coefficient below the base-relative answer-mass floor. A per-row filter can produce confusing lone high-C dots after collapse.
- 2026-06-30: README structural check before final regeneration: 9 images, `missing_images: []`, 15 local links, `missing_local_links: []`, and no 16PF map link.
- 2026-06-30: The endpoint-only map removed the path readers need. The sign-offset range geometry also made a one-dimensional value plot look like a fake two-dimensional fork.
- 2026-06-30: User wants range plots to show the path as sign lanes, not coefficient-x swim lines. The fix is negative/base/positive lanes, with vertical paths inside each lane.
