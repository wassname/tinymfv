# Authority steering pipeline

## User Goal
Build a clean `+Authority <-> -Authority` steering demo, then show it in tinymfv only after it works.

Order matters:

1. Get the steer good.
2. Get the eval good.
3. Update the README to show the successful steer/eval.

Do not polish README plots from a bad or ambiguous steer.

## User Preferences
- The target axis is Authority only: `+Authority <-> -Authority`.
- The Authority persona pair is fixed by construct, not selected by validation. Validation may choose templates and scenarios, but must not choose a different persona pair or semantic proxy.
- Pure Authority means exactly this premise unless the user changes it:
  - `+Authority`: `authority-respecting`
  - `-Authority`: `authority-disregarding`
- Do not use dignity, tradition, obedience, social norms, care, wellbeing, hierarchy-as-status, or broad social order as the pole definition.
- Do not use `dignity_over_authority` as the steer axis. That is a conflict axis and confounds Authority with dignity/care/wellbeing.
- Dignity, tradition, Social Norms, Care, Fairness, wellbeing, style, verbosity, refusal, and sycophancy are side-effect checks, not the axis definition.
- Persona/template/scenario validation comes before steering.
- Use the closest practical model for validation. Current path: `qwen/qwen3-14b` on DeepInfra, because `qwen/qwen3-8b` has only AtlasCloud/Alibaba endpoints and Alibaba was upstream-rate-limited. Steer target is still `Qwen/Qwen3-4B`.
- Queue GPU steer/eval with `pueue` only after selection looks sane.
- README is for readers. Keep journal, caveats, failed attempts, and work-in-progress methodology in this spec, not in README.
- README should eventually show only the successful artifacts: what was steered, what moved, which eval measured it, and why a researcher might use it.

## Scope
In:
- Fix a pure `+Authority/-Authority` persona pair a priori.
- Test templates and scenarios for that fixed pair using the library validation workflow.
- Export selected scenarios into steering-lite, not only the persona library.
- Queue a steering-lite run only after validation passes clear gates.
- Run tinymfv evals on the resulting steer and inspect direction/noise/side effects.
- Update README plots/table after the steer and eval pass.

Out:
- README prose polish before the steer works.
- Treating strict22 `dignity_over_authority` as the final selection.
- Treating `authority_tradition_obedience` or any other validation-winning proxy as the final axis.
- General automatic sign labeling for arbitrary steering vectors.

## Requirements
- R1: The persona pair is fixed pure Authority. Done means: artifacts show one pair only, `authority-respecting` versus `authority-disregarding`, with no dignity/tradition/Social-Norms/care/wellbeing pole wording. VERIFY: inspect pair JSON and rendered prompt examples.
- R2: Template validation holds R1 fixed. Done means: summary ranks templates for the fixed pair only, with no axis/persona-variant winner. VERIFY: template screen artifact contains one pair id and multiple template ids.
- R3: Scenario validation holds R1 and the chosen template fixed. Done means: source-stratified scenarios are selected because they elicit the fixed Authority contrast cleanly, not because they redefine the axis. VERIFY: selected examples file shows the same pair/template on every row.
- R4: The steering run uses selected data committed in steering-lite. Done means: steering-lite has `data/persona_library_selections/pure_authority_*.jsonl` plus matching summary, and the runner references that file. VERIFY: `rg -n "pure_authority" data/persona_library_selections scripts`.
- R5: tinymfv direction is correct over the evaluated coefficient path. Done means: MFV Authority moves in the intended direction for paired `+c/-c` rows, and MFV coherence evidence shows the readout stayed usable. VERIFY: effect table from steering-lite/tinymfv output.
- R6: Eval reliability is measured before README. Done means: MFV, MFQ-2, Humor, and Big Five report profile shift, reader-logit shift, and noise/CI where available; MFQ-2 uses sampled reads if needed. VERIFY: generated summary table.
- R7: README only shows successful artifacts. Done means: README plot captions name the pure Authority steer and use regenerated images from the final run. VERIFY: README image links and run-dir command point to the final output.

## Tasks
- [x] T1 (R1): Write the fixed pure Authority pair.
  - steps: add the exact pair `authority-respecting` vs `authority-disregarding` to persona-library/steering-lite selection inputs; remove axis-variant selection from the active path.
  - done:
    - [x] pair artifact contains exactly one pair id.
    - [x] rendered example shows only `authority-respecting` and `authority-disregarding`.
  - verify: `rg -n "authority-respecting|authority-disregarding|authority_tradition_obedience|dignity_over_authority" <selection files>`.
  - success: only the pure pair appears in active selection files.
  - likely_fail: old `authority_tradition_obedience` runner or JSON is still referenced.
  - sneaky_fail: pair wording smuggles in tradition, dignity, care, or social norms; catch by reading rendered prompts.
  - UAT: file paths show the exact pure pair and no proxy pair.
- [x] T2 (R2): Validate templates with the fixed pair.
  - steps: run a stage-A screen over fixed pair x candidate templates x source-stratified Authority-affordant scenarios.
  - verify: summary table has one pair id, multiple template ids, and off-axis/style audit columns.
  - success: choose a template because it cleanly elicits the fixed Authority contrast.
  - likely_fail: all templates produce weak separation.
  - sneaky_fail: the top template wins by persona echo or verbosity; catch with echo/style columns and example reads.
  - UAT: selected template artifact plus examples can be inspected without reading code.
- [x] T3 (R3): Validate and export scenarios with the fixed pair and chosen template.
  - steps: run stage B over up to 30 source-ranked scenarios per source where practical; export strict-pass scenarios and examples.
  - verify: selected source-count table plus selected examples file.
  - success: selected scenarios are diverse and elicit pure Authority contrast.
  - likely_fail: many sources produce no strict-pass scenarios.
  - sneaky_fail: scenarios themselves make the contrast about tradition/social norms or care; catch by source-balanced example inspection.
  - UAT: selected examples show the same fixed pair/template on varied scenarios.
- [x] T4 (R4): Export selected data into steering-lite.
  - steps: copy selected JSONL and summary into `steering-lite/data/persona_library_selections/`; add a pure Authority runner.
  - verify: `rg -n "pure_authority|authority-respecting|authority-disregarding|authority_tradition_obedience" /media/wassname/SGIronWolf/projects5/2026/lite/steering-lite/data/persona_library_selections /media/wassname/SGIronWolf/projects5/2026/lite/steering-lite/scripts`.
  - success: runner references pure Authority selection and no proxy/strict22 path.
  - likely_fail: runner still references `authority_tradition_obedience`.
  - sneaky_fail: copied summary and JSONL disagree on pair/template; catch by parsing both.
  - UAT: committed steering-lite files point to the pure Authority selection.
- [x] T5 (R5): Queue pure-Authority steer/eval UAT.
  - steps: queue `sspace` first on `mfv mfq2` at `c=0.5,1`, `admin-n-samples=8`; only broaden after it passes.
  - verify: pueue label includes why/resolve; output contains MFV and MFQ-2 profiles over the signed c-grid.
  - success: small positive `c` raises MFV/MFQ-2 Authority and small negative `c` lowers it, without Social Norms dominating MFV.
  - likely_fail: sign is reversed or flat.
  - sneaky_fail: Authority moves only because high-c rows are incoherent; catch with pmass, MFV margin, and small-c verifier.
  - UAT: verifier table shows signed direction, selectivity, and coherence. Result: failed for all tested methods, with coherence clean.
    - direction: MFV Authority `dlogit(+0.5) > 0` and `dlogit(-0.5) < 0`; MFQ-2 Authority `C(+0.5) > base` and `C(-0.5) < base`.
    - selectivity: MFV Social Norms is not larger than Authority at the same small-c rows.
    - coherence: MFV `frac_unscorable` stays near zero and margin does not collapse; ordinal `pmass` stays near base.
- [x] T6 (R5, R6): Compare runner-supported methods.
  - steps: queue `mean_diff`, `pca`, `sspace`, `directional_ablation`, and `linear_act` with unique timestamped output dirs.
  - verify: each output dir has `summary.json`, `mfv_profiles.csv`, `mfq2_profiles.csv`, and method-specific verifier output.
  - success: choose a method that preserves pure Authority direction and selectivity.
  - likely_fail: all methods reproduce off-axis Social Norms movement.
  - sneaky_fail: a method looks good because retained c values differ; catch by printing retained c values and quality gates.
  - UAT: method comparison table links each output dir and verifier table. Result: no method is README-ready.
- [ ] T7 (R6): Evaluate reliability and side effects.
  - steps: run tinymfv summary over MFV, MFQ-2, Humor, Big Five; estimate noise/CI where available.
  - verify: table includes profile shift/human SD, reader-logit shift, and uncertainty/noise columns.
  - success: Authority signal is larger than noise and side effects are interpretable.
  - likely_fail: MFQ-2 noisy or opposite sign.
  - sneaky_fail: profile shift comes from loss of answer structure; catch with answer mass, survey contrast, and MFV margin.
  - UAT: one table lets the user decide whether the steer is good enough to show.
- [x] T12 (R6): Estimate the minimum MFQ-2 `N` needed for stable steering plots.
  - steps: measure bootstrapped MFQ-2 Authority path variability at subset sizes `N=1,2,4,8` from per-sample readouts, or add a per-sample export if the current aggregate output is insufficient.
  - verify: table reports mean and bootstrap std/CI for Authority `C` deltas at each subset size and c row.
  - success: choose the smallest `N` whose bootstrapped direction and effect size are stable enough for README plots.
  - likely_fail: current `mfq2_profiles.csv` only has aggregate `C_sd`/`C_sem`, not the sample-level trajectories needed for subset bootstrapping.
  - sneaky_fail: bootstrapping across items instead of think trajectories answers the wrong question; catch by inspecting the exported unit of resampling.
  - UAT: committed table or CSV shows MFQ-2 subset stability, with enough detail to choose default `N`. Result: `/media/wassname/SGIronWolf/projects5/2026/lite/tinymfv/docs/reviews/mfq2_authority_pca_n_bootstrap_20260701.csv` shows correct Authority direction for every bootstrap draw at `N=1,2,4,8` and `c in {-1,-0.5,+0.5,+1}`. `N=1` is enough for sign stability in this run; `N=4` or `N=8` gives tighter deltas for publication/readme plots.
- [x] T8 (R1-R5): Repair the pure-Authority data selection.
  - steps: inspect rendered strict25 examples; remove scenarios whose Authority affordance is actually Social Norms, legality, institutional controversy, or welfare/autonomy; rerun scenario validation with stricter source balance and example audit.
  - verify: selected examples file has direct respect/disregard for authority without Social Norms or dignity/care wording, and source counts are not dominated by ValueBench.
  - success: strict selected rows still separate the fixed persona pair, and the rendered examples read like Authority rather than social-order controversy.
  - likely_fail: too few strict rows survive after deconfounding.
  - sneaky_fail: judge score is high because the responses echo `authority-respecting`/`authority-disregarding`; catch by reading examples and echo/style columns.
  - UAT: committed steering-lite selection file plus example table show the corrected data before another GPU run. Result: pass for data export, steer/eval pending.
    - persona-library source commit: `f37cca5` (`Tighten pure authority selection`).
    - steering-lite prompt-transfer commit: `6e05b29` (`Use verbatim persona-library prompts`).
    - steering-lite selection commit: `f6ea42b` (`Add mundane pure-authority selection`).
    - selected data: `/media/wassname/SGIronWolf/projects5/2026/lite/steering-lite/data/persona_library_selections/pure_authority_qwen3_14b_mundane15.jsonl`.
    - selected examples: `/media/wassname/SGIronWolf/projects5/2026/weight-steering-repos/persona-steering-template-library/out/pure_authority_verbatim_mundane5_20260630/selection_score70/selected_examples.md`.
- [x] T10 (R5): Run mundane15 pure-Authority steer/eval UAT across methods.
  - steps: queue `mean_diff`, `pca`, `sspace`, `directional_ablation`, and `linear_act` on MFV/MFQ-2 with `c-grid=0.5,1`, `admin-n-samples=8`, and the verifier.
  - verify: each pueue output dir reports MFV Authority direction over every paired `+c/-c` row and MFV coherence evidence with `scripts/verify_authority_showcase.py <out_dir>`.
  - success: at least one method shows signed MFV Authority direction over the coefficient path while MFV coherence evidence stays usable.
  - likely_fail: smaller cleaner data is underpowered and gives weak/unstable direction.
  - sneaky_fail: method appears to work only because high-c rows lose answer structure; catch with MFV `pmass`, `mean_margin`, `margin/base`, `frac_unscorable`, and matching c grid.
  - UAT: verifier tables for pueue tasks 417-421. Result under the corrected MFV-only verifier: `pca` and `linear_act` show signed MFV Authority direction at `c=0.5` and `c=1.0`; `sspace` is signed at `c=0.5` but not `c=1.0`; `mean_diff` is reversed; `directional_ablation` moves both signs the same way.

| method | output dir | MFV Authority direction | MFV coherence evidence |
|---|---|---|---|
| mean_diff | `/media/wassname/SGIronWolf/projects5/2026/lite/steering-lite/outputs/20260630T163010Z_pure_authority_mundane15_mean_diff_mfv_mfq2_n8` | reversed at both c values | `pmass=1`, `unscorable=0`, margin/base `1.019..1.043` at +c and `1.024..1.029` at -c |
| pca | `/media/wassname/SGIronWolf/projects5/2026/lite/steering-lite/outputs/20260630T163010Z_pure_authority_mundane15_pca_mfv_mfq2_n8` | signed at `c=0.5` and `c=1.0` | `pmass=1`, `unscorable=0`, margin/base `0.927..0.805` at +c and `1.112..0.934` at -c |
| sspace | `/media/wassname/SGIronWolf/projects5/2026/lite/steering-lite/outputs/20260630T163010Z_pure_authority_mundane15_sspace_mfv_mfq2_n8` | signed at `c=0.5`, not at `c=1.0` | `pmass=1`, `unscorable=0`, margin/base `0.972..0.980` at +c and `1.059..1.193` at -c |
| directional_ablation | `/media/wassname/SGIronWolf/projects5/2026/lite/steering-lite/outputs/20260630T163010Z_pure_authority_mundane15_directional_ablation_mfv_mfq2_n8` | both signs raise Authority | `pmass=1`, `unscorable=0`, margin/base `0.812..0.813` at +c and `0.843..0.760` at -c |
| linear_act | `/media/wassname/SGIronWolf/projects5/2026/lite/steering-lite/outputs/20260630T163010Z_pure_authority_mundane15_linear_act_mfv_mfq2_n8` | signed at `c=0.5` and `c=1.0` | `pmass=1`, `unscorable=0`, margin/base `1.133..1.149` at +c and `0.891..0.760` at -c |
- [ ] T9 (R7): Update README only from the final successful run.
  - steps: regenerate all README plots from one final artifact; add concise table and captions.
  - verify: README image links resolve; no 16PF plot; no WIP methodology journal in reader prose.
  - success: reader sees what tinymfv is, what was steered, which datasets measured it, and why it matters.
  - likely_fail: README narrates failed strict22/debug history.
  - sneaky_fail: captions imply a general sign convention; catch by reading README without this spec.
  - UAT: external-review-v2 comprehension panel can explain what/where/why/measurement/dataset in its own words.
- [x] T11 (R5): Render candidate PCA maps from the best current MFV path.
  - steps: choose the method using MFV Authority direction and MFV coherence evidence only; render candidate maps/ranges from the selected steering-lite run.
  - verify: `uv run python scripts/plot_steer_showcase.py --run-dir /media/wassname/SGIronWolf/projects5/2026/lite/steering-lite/outputs/20260630T163010Z_pure_authority_mundane15_pca_mfv_mfq2_n8 --out docs/img/showcase_authority_pca_mundane15 --vec-label "pure Authority, PCA (+c = more Authority)" --coherence-frac 0.99 --margin-frac 0.50 --contrast-frac 0.000001`
  - success: MFV candidate map and range exist, with c path over `[-1.0, -0.5, 0.0, 0.5, 1.0]`.
  - likely_fail: the plotter uses MFQ-2 or arbitrary small-c thresholds to choose the method.
  - sneaky_fail: candidate images silently replace README figures before the steer is accepted; catch by writing to a separate candidate directory.
  - UAT: candidate artifacts are in `/media/wassname/SGIronWolf/projects5/2026/lite/tinymfv/docs/img/showcase_authority_pca_mundane15/`. Result: `pca` is the best current candidate by MFV-only evidence; it has signed Authority movement at `c=0.5` and `c=1.0`, `pmass=1.000`, `frac_unscorable=0.000`, and minimum MFV `margin/base=0.805`.
- [x] T13 (R5): Render steering-method comparison plots.
  - steps: render MFV and MFQ-2 maps/ranges for `mean_diff`, `pca`, `sspace`, `directional_ablation`, and `linear_act` from pueue runs 417-421.
  - verify: `find docs/img/showcase_authority_method_compare_mundane15 -type f`.
  - success: one method-named directory per steering method, each with MFV and MFQ-2 map/range PNG/SVG pairs.
  - likely_fail: method plots overwrite README figures or the candidate PCA directory.
  - sneaky_fail: plots use different c retention rules across methods; catch by plotter output showing coherent c values for each method.
  - UAT: artifacts are in `/media/wassname/SGIronWolf/projects5/2026/lite/tinymfv/docs/img/showcase_authority_method_compare_mundane15/`.
- [x] T14 (R5-R7): Regenerate live README plot images from the successful PCA full run.
  - steps: run the full PCA showcase on MFV, MFQ-2, Humor Styles, and Big Five; verify MFV Authority direction and coherence; regenerate `docs/img/showcase`.
  - verify:
    - `uv run --extra benchmark python scripts/verify_authority_showcase.py outputs/20260630T222000Z_pure_authority_mundane15_pca_readme_mfv_mfq2_humor_big5_n8`
    - `uv run python scripts/plot_steer_showcase.py --run-dir /media/wassname/SGIronWolf/projects5/2026/lite/steering-lite/outputs/20260630T222000Z_pure_authority_mundane15_pca_readme_mfv_mfq2_humor_big5_n8 --out docs/img/showcase --vec-label "pure Authority, PCA (+c = more Authority)" --coherence-frac 0.99 --margin-frac 0.50 --contrast-frac 0.000001`
  - success: all four live showcase plot sets use the same coherent path `[-1, -0.5, 0, +0.5, +1]` and no 16PF plot is regenerated.
  - likely_fail: MFV Authority sign regresses in the full run.
  - sneaky_fail: plotter silently drops different `c` rows per instrument; catch by the shared coherent c-values printout.
  - UAT: live plot files are in `/media/wassname/SGIronWolf/projects5/2026/lite/tinymfv/docs/img/showcase/{mfv,mfq2,humor_styles,big5}/`. MFV verifier result: Authority dlogit `+0.306/-0.327` at `c=0.5` and `+0.906/-0.910` at `c=1.0`; `pmass=1.000`, `frac_unscorable=0.000`, and minimum `margin/base=0.800`. Plotter result: shared coherent c values `[-1.0, -0.5, 0.0, 0.5, 1.0]`.

## Context
- Current wrong path: `dignity_over_authority` strict22. It was selected as the best dignity conflict axis, not the desired Authority-only axis.
- Current target: `+Authority <-> -Authority`.
- Current target model: `Qwen/Qwen3-4B`.
- Existing validation model evidence: `qwen/qwen3-8b`, same family and reasonably close, but weak strict-pass rates are not enough.
- Current active training selection: `pure_authority_qwen3_14b_mundane15`, fixed pair with verbatim persona text and 15 strict-pass, score>=70 scenarios from role-authority mundane sources.
- Sign calibration is run-local. If the axis declares an Authority anchor, positive `c` can be oriented to that anchor. Without a declared anchor, plots should use `+c`/`-c` only.
- Persona-library Authority-only prep files:
  - `/media/wassname/SGIronWolf/projects5/2026/weight-steering-repos/persona-steering-template-library/data/personas/persona_pairs_v2_candidates.jsonl`
  - `/media/wassname/SGIronWolf/projects5/2026/weight-steering-repos/persona-steering-template-library/out/authority_only_20260630/manifest.json`
  - `/media/wassname/SGIronWolf/projects5/2026/weight-steering-repos/persona-steering-template-library/out/authority_only_20260630/stage_a_live.json`
  - `/media/wassname/SGIronWolf/projects5/2026/weight-steering-repos/persona-steering-template-library/out/authority_variants_20260630/manifest.json`
  - `/media/wassname/SGIronWolf/projects5/2026/weight-steering-repos/persona-steering-template-library/out/authority_variants_20260630/stage_a_live.json`
  - `/media/wassname/SGIronWolf/projects5/2026/weight-steering-repos/persona-steering-template-library/out/authority_affordant_20260630/manifest.json`
  - `/media/wassname/SGIronWolf/projects5/2026/weight-steering-repos/persona-steering-template-library/out/authority_affordant_20260630/stage_a_live.json`
  - `/media/wassname/SGIronWolf/projects5/2026/weight-steering-repos/persona-steering-template-library/out/authority_affordant_20260630/alibaba_smoke_dryrun.json`
  - `/media/wassname/SGIronWolf/projects5/2026/weight-steering-repos/persona-steering-template-library/out/authority_affordant_20260630/alibaba_smoke_live.json`
  - `/media/wassname/SGIronWolf/projects5/2026/weight-steering-repos/persona-steering-template-library/out/authority_affordant_20260630/deepinfra_qwen3_14b_nothink_smoke_live.json`
  - `/media/wassname/SGIronWolf/projects5/2026/weight-steering-repos/persona-steering-template-library/out/authority_affordant_20260630/stage_a_live_qwen3_14b_deepinfra.json`

## Log
- 2026-06-30: User corrected the axis. `dignity_over_authority` is not the goal; it confounds Authority with dignity/care/wellbeing. Use Authority-only.
- 2026-06-30: Killed pueue task 397 because it was spending GPU on README/eval plots for the old strict22 steer before the steer was known-good.
- 2026-06-30: Added `authority_only` to persona-library as `authority-respecting` vs `authority-skeptical`, with behavior definitions that avoid dignity/care/wellbeing wording.
- 2026-06-30: `prepare_authority_steering_selection.py` now defaults to `authority_only` and supports explicit `--axis-ids`. Dry-run manifest: 24 stage-A prompts from 12 sources, 10 templates, 240 planned pairs; stage B has 342 candidate scenarios.
- 2026-06-30: First `authority_only` screen was too weak: best template strict-pass 0.083, mean axis delta 1.633. Audit samples showed generic wellbeing/autonomy responses when prompts lacked authority affordance, and several templates were grammatically bad for adjective personas.
- 2026-06-30: Revised screen tests three mirrored Authority-only variants (`authority_only`, `authority_role_duty`, `authority_tradition_obedience`) and ten adjective-compatible templates. Dry-run manifest: 3 axes * 10 templates * 24 prompts = 720 planned pairs.
- 2026-06-30: Switched persona-library validation to `openrouter_wrapper` after direct OpenAI SDK calls hit upstream Qwen 429s. Pushed wrapper fix `d011cd6` because its retry predicate was async and broke stamina/tenacity retries.
- 2026-06-30: Added DeepInfra as explicit OpenRouter provider preference in the validator. That was the wrong abstraction for `qwen/qwen3-8b`: the OpenRouter endpoint list shows only AtlasCloud fp8 and Alibaba. The earlier smoke did not prove DeepInfra routing.
- 2026-06-30: Replaced seed-random scenario selection with source-stratified Authority-affordance ranking. New stage-A rows include explicit boss/rules/commanding-officer/king/captain/protocol cases; weak sources remain visible so stage A can reject them.
- 2026-06-30: Replaced global provider preference with generator-only provider pinning. Generator calls now use `provider.only=["Alibaba"]` and `allow_fallbacks=false`; judge calls remain unpinned. Dry-run artifact `out/authority_affordant_20260630/alibaba_smoke_dryrun.json` records `generator_provider_only=["Alibaba"]`.
- 2026-06-30: Live Alibaba smoke confirmed the routing but hit upstream 429s from Alibaba: artifact `out/authority_affordant_20260630/alibaba_smoke_live.json` has `generator_provider_only=["Alibaba"]`, `n_results=1`, and the row error cites `provider_name="Alibaba"`. Next step is BYOK/wait/retry, or choose another close Qwen endpoint deliberately.
- 2026-06-30: OpenRouter endpoint scan found `qwen/qwen3-14b` has DeepInfra; `qwen/qwen3-8b` does not. Direct mini-test showed DeepInfra Qwen3 returns `content=null` unless prompted with `/no_think`. Patched the validator to add `/no_think` for Qwen-family generator models.
- 2026-06-30: Smoke artifact `out/authority_affordant_20260630/deepinfra_qwen3_14b_nothink_smoke_live.json` completed 3/3 rows with `generator_provider_only=["DeepInfra"]`. Early signal: `authority_tradition_obedience` strict-pass 1/1 and mean axis delta 6.3; generic `authority_only` still mostly collapses toward welfare.
- 2026-06-30: Pushed `openrouter_wrapper` commit `816e2d0` to retry OpenRouter upstream SSE JSON/rate-limit errors, then bumped persona-library to that wrapper commit. Stage-A run resumed from cache and passed the previous Gemini failure point.
- 2026-06-30: Current stage-A artifact `out/authority_affordant_20260630/stage_a_live_qwen3_14b_deepinfra.json` was at 274/720 results, 274 successes, 0 errors when checked.
- 2026-06-30: Stage A completed 720/720 after rerun with `openrouter_wrapper` commit `01190c8`, which retries upstream idle timeouts. Persona-library bump commit: `e61f8a0`.
- 2026-06-30: Stage-A winner is `authority_tradition_obedience` with template `Answer as someone who is {persona}.`, strict pass 0.208 and mean axis delta 3.877. This is weak, so do not steer from stage A directly. Stage B is running over 342 authority-affordant scenarios to select only rows where the axis actually separates.
- 2026-06-30: Stage B completed 342/342 with 49 strict rows. Exported stricter score>=60 selection with 32 scenarios across 8 sources. Committed in steering-lite as `2b59088`: `data/persona_library_selections/authority_only_qwen3_14b_score60.jsonl`, matching summary, and `scripts/run_authority_only_score60_showcase.sh`.
- 2026-06-30: Killed steering-lite pueue task 399 after calibration reached `C=39.758`, `kl_p95=0.547` but then spent 34 minutes in the final calibration snapshot with no eval outputs. Killed task 400 because the runner did not forward CLI overrides, so it would have ignored `--fixed-C`.
- 2026-06-30: Pushed steering-lite `eafae69` so `scripts/run_authority_only_score60_showcase.sh` forwards `"$@"`. Queued pueue task 401 with `--fixed-C 39.758 --out outputs/20260630_authority_only_qwen3_14b_score60_sspace_mfv_mfq2_n8_fixedC39758`, then killed it because the pre-unbuffered runner gave no live stdout/files after five minutes.
- 2026-06-30: Pushed steering-lite `899d932` to set `PYTHONUNBUFFERED=1` in the authority runner. Queued pueue task 402 with output `outputs/20260630_authority_only_qwen3_14b_score60_sspace_mfv_mfq2_n8_fixedC39758_unbuf`; log confirms `fixed C=+39.7580 (iso-KL skipped)`.
- 2026-06-30: Killed pueue task 402 after MFQ-2 gave a mixed result: small `c` had the right local sign (`authority` C base `30.80`, `+0.5` `33.17`, `-0.5` `29.72`) but `+1` doubled back to `27.23` while pmass stayed `1.000`. Queued pueue task 403 as focused MFV-only check at `c=0.5,1`: `outputs/20260630_authority_only_qwen3_14b_score60_sspace_mfv_c05_c1_fixedC39758`.
- 2026-06-30: Pueue task 403 completed. MFV small-c direction passed (`Authority` dlogit `+0.176` at `+0.5`, `-0.392` at `-0.5`; pmass `1.000`, unscorable `0`). MFV `+1/-1` also moved Authority in the expected direction (`+1.454`, `-0.580`). But the vector is not selective enough for README: MFV Social Norms moved more than Authority (`-0.621` at `+0.5`, `+1.121` at `-0.5`; `-1.511/+2.281` at `+1/-1`), and MFQ-2 doubled back at `+1`. Current verdict: local-direction pass, path/selectivity fail.
- 2026-06-30: Regenerated current MFV-only inspection plots from the completed `sspace` run under `docs/img/showcase_authority_score60_sspace_current/`. This is for inspection only, not README replacement.
- 2026-06-30: Queued one comparable showcase run per runner-supported steering method using the Authority-only score60 selection and unique timestamped output dirs:
  - pueue 404: `mean_diff` -> `/media/wassname/SGIronWolf/projects5/2026/lite/steering-lite/outputs/20260630T104202_authority_tradition_obedience_score60_mean_diff_mfv_mfq2_humor_big5_n8`
  - pueue 405: `pca` -> `/media/wassname/SGIronWolf/projects5/2026/lite/steering-lite/outputs/20260630T104202_authority_tradition_obedience_score60_pca_mfv_mfq2_humor_big5_n8`
  - pueue 406: `sspace` -> `/media/wassname/SGIronWolf/projects5/2026/lite/steering-lite/outputs/20260630T104202_authority_tradition_obedience_score60_sspace_mfv_mfq2_humor_big5_n8`
  - pueue 407: `directional_ablation` -> `/media/wassname/SGIronWolf/projects5/2026/lite/steering-lite/outputs/20260630T104202_authority_tradition_obedience_score60_directional_ablation_mfv_mfq2_humor_big5_n8`
  - pueue 408: `linear_act` -> `/media/wassname/SGIronWolf/projects5/2026/lite/steering-lite/outputs/20260630T104202_authority_tradition_obedience_score60_linear_act_mfv_mfq2_humor_big5_n8`
- 2026-06-30: User caught a spec error: validation was supposed to hold the pure Authority persona pair fixed and choose only templates/scenarios. The previous stage-A screen incorrectly let validation choose among axis/persona variants, and selected `authority_tradition_obedience`, a proxy contaminated by tradition/Social Norms. Killed pueue 404 and removed 405-408; old score60 artifacts are invalid evidence for the pure-Authority goal.
- 2026-06-30: Replanned the active path around the fixed premise `authority-respecting` vs `authority-disregarding`. Validation now only selects templates and scenarios; method comparison only happens after a cheap pure-Authority steer/eval UAT passes.
- 2026-06-30: User clarified the verification target from the previous good readout: direction, selectivity, and coherence. Direction means the intended signed Authority movement is present. Selectivity means the movement is not dominated by MFV Social Norms or another off-axis foundation. Coherence means answer mass/margin/unscorable gates do not explain the effect.
- 2026-06-30: Persona-library commit `f355b47` adds `pure_authority` (`authority-respecting` vs `authority-disregarding`) and changes `prepare_authority_steering_selection.py` so the active path has one fixed pair and no axis/persona selection. Dry-run artifact: `/media/wassname/SGIronWolf/projects5/2026/weight-steering-repos/persona-steering-template-library/out/pure_authority_20260630/stage_a_dryrun_qwen3_14b_deepinfra.json`, 240 planned cells, one pair id.
- 2026-06-30: Persona-library commit `6450d51` replaced article templates like `a {persona} person` because they rendered as `a authority-respecting person`. Neutral-template Stage A artifact: `/media/wassname/SGIronWolf/projects5/2026/weight-steering-repos/persona-steering-template-library/out/pure_authority_20260630_neutral_templates/stage_a_live_qwen3_14b_deepinfra.json`. Winner: `Speak with the priorities of someone {persona}.`, strict pass 0.125, mean axis delta 3.460. This is not strong by itself; Stage B is needed.
- 2026-06-30: Stage B artifact: `/media/wassname/SGIronWolf/projects5/2026/weight-steering-repos/persona-steering-template-library/out/pure_authority_20260630_neutral_templates/stage_b_live_qwen3_14b_deepinfra.json`, 342/342 success, strict pass 25 rows (0.073). Exported strict25 selection from 5 sources, source-skewed toward ValueBench: 18 ValueBench, 4 Daily Dilemmas, 1 AIRisk, 1 SocialChem, 1 W2S character.
- 2026-06-30: Steering-lite commit `7e2ca7c` tightened `scripts/verify_authority_showcase.py` to fail if small-c signed Authority direction fails, MFV Social Norms is larger than Authority, or coherence gates fail. Running it on the old invalid score60 MFV output reproduces the useful verdict: signed direction pass, selectivity fail, coherence clean.
- 2026-06-30: Steering-lite commit `b35e269` adds `data/persona_library_selections/pure_authority_qwen3_14b_strict25.jsonl`, matching summary, and `scripts/run_pure_authority_strict25_showcase.sh`. This is the active pure-Authority selection.
- 2026-06-30: Queued pure-Authority strict25 method comparison on MFV/MFQ-2 with `c-grid=0.5,1`, `admin-n-samples=8`: pueue 406 `mean_diff`, 407 `pca`, 408 `sspace`, 409 `directional_ablation`, 410 `linear_act`. Pass criterion in labels: small-c Authority sign correct, MFV Social Norms not larger than Authority, coherence ok.
- 2026-06-30: Pueue 406 `mean_diff` completed at `/media/wassname/SGIronWolf/projects5/2026/lite/steering-lite/outputs/20260630T123208Z_pure_authority_strict25_mean_diff_mfv_mfq2_n8`. Verifier verdict: fail. MFQ-2 small-c direction passed (`authority` C base `30.180`, `+0.5` delta `+5.254`, `-0.5` delta `-4.825`) and coherence passed (`pmass=1.000`). MFV failed direction and selectivity: Authority dlogit deltas were `-0.065` at `+0.5` and `-0.045` at `-0.5`; Social Norms moved more (`-0.632`, `+1.001`). MFV coherence was clean (`pmass=1.000`, `frac_unscorable=0.000`), so this is not answer collapse.
- 2026-06-30: Pueue 407 `pca` completed at `/media/wassname/SGIronWolf/projects5/2026/lite/steering-lite/outputs/20260630T123208Z_pure_authority_strict25_pca_mfv_mfq2_n8`. Verifier verdict: fail. MFV small-c direction barely passed (`Authority` dlogit `+0.024` at `+0.5`, `-0.069` at `-0.5`) but selectivity failed because Social Norms moved more (`-0.005`, `+0.427`). MFQ-2 direction failed: `authority` C base `30.836`, `+0.5` delta `-1.018`, `-0.5` delta `-4.221`; high-c rows also lowered Authority. Coherence was clean for both instruments.
- 2026-06-30: Pueue 408 `sspace` completed at `/media/wassname/SGIronWolf/projects5/2026/lite/steering-lite/outputs/20260630T123208Z_pure_authority_strict25_sspace_mfv_mfq2_n8`. Verifier verdict: fail. MFV small-c direction passed (`Authority` dlogit `+0.187` at `+0.5`, `-0.276` at `-0.5`), but selectivity failed because Social Norms moved more (`-0.497`, `+0.596`). MFQ-2 direction failed: `authority` C base `32.349`, `+0.5` delta `-1.265`, `-0.5` delta `-6.489`. Coherence was clean for both instruments (`pmass=1.000`; MFV `frac_unscorable=0.000`), so this is direction/selectivity failure, not readout collapse.
- 2026-06-30: Pueue 409 `directional_ablation` completed at `/media/wassname/SGIronWolf/projects5/2026/lite/steering-lite/outputs/20260630T123208Z_pure_authority_strict25_directional_ablation_mfv_mfq2_n8`. Verifier verdict: fail. Iso-KL calibration never reached target: bracket floor `C=0.0061` still had `kl_p95=0.971` against target `0.5`. MFV direction failed because both signs raised Authority (`+0.296` at `+0.5`, `+0.219` at `-0.5`), and selectivity failed because Social Norms moved more (`+0.407`, `+0.206`). MFQ-2 direction failed hard: `authority` C base `31.388`, `+0.5` delta `-6.284`, `-0.5` delta `-7.094`. Coherence was clean for both instruments.
- 2026-06-30: Pueue 411 `linear_act` completed at `/media/wassname/SGIronWolf/projects5/2026/lite/steering-lite/outputs/20260630T123208Z_pure_authority_strict25_linear_act_mfv_mfq2_n8`. Verifier verdict: fail. MFQ-2 small-c direction passed weakly (`authority` C base `30.660`, `+0.5` delta `+0.351`, `-0.5` delta `-3.905`), but MFV direction failed (`Authority` dlogit `-0.202` at `+0.5`, `+0.102` at `-0.5`) and selectivity failed because Social Norms moved more (`+0.294`, `+0.111`). Coherence was clean (`pmass=1.000`; MFV `frac_unscorable=0.000`), so this is not answer collapse.
- 2026-06-30: All pure-Authority strict25 method runs failed the same verifier family. Pattern: coherence is clean, but either MFV Authority direction is wrong/weak, MFQ-2 Authority direction is wrong/weak, or MFV Social Norms dominates. This points more to selection/template/scenario contamination than to plotting or answer-token collapse.
- 2026-07-01: Repaired the pure-Authority selection pipeline around the fixed pair and verbatim persona instructions. Removed direct ValueBench, Airisk/Machiavelli, law/police, safety/harm, and lexical false positives from bare `order/orders`, `master`, and `senior`. Final validator artifact: `/media/wassname/SGIronWolf/projects5/2026/weight-steering-repos/persona-steering-template-library/out/pure_authority_verbatim_mundane5_20260630/stage_b_live_qwen3_14b_deepinfra.json`; 125 successes, 0 errors, 33 strict rows, 15 score>=70 selected rows across 4 sources, 0 persona echo/refusal.
- 2026-07-01: Copied the selected data into steering-lite and added verbatim persona-library prompt support so the steer uses the same prompts that validation tested. Queued pueue 417-421 for `mean_diff`, `pca`, `sspace`, `directional_ablation`, and `linear_act` on MFV/MFQ-2 with `c=0.5,1`, `N=8`.
- 2026-07-01: User correctly objected that the `c=0.5`, MFQ-2, Social-Norms-selectivity, and hard coherence thresholds were arbitrary. Replaced the active verifier with an MFV-only evidence report: Authority direction over all paired c values, plus MFV `pmass`, `mean_margin`, `margin/base`, `frac_unscorable`, and `mean_nll_prefill`. Under this corrected view, `pca` and `linear_act` have signed MFV Authority direction across the evaluated path; `sspace` is only locally signed; `mean_diff` and `directional_ablation` are direction failures.
- 2026-07-01: Rendered candidate maps/ranges from the current best MFV-only method, `pca`, into `/media/wassname/SGIronWolf/projects5/2026/lite/tinymfv/docs/img/showcase_authority_pca_mundane15/`. These are candidate artifacts, not README replacements.
- 2026-07-01: Rendered method-comparison maps/ranges into `/media/wassname/SGIronWolf/projects5/2026/lite/tinymfv/docs/img/showcase_authority_method_compare_mundane15/`.
- 2026-07-01: Pueue 423 completed the full PCA run over MFV, MFQ-2, Humor Styles, and Big Five at `N=8`: `/media/wassname/SGIronWolf/projects5/2026/lite/steering-lite/outputs/20260630T222000Z_pure_authority_mundane15_pca_readme_mfv_mfq2_humor_big5_n8`. MFV Authority direction is signed at both evaluated magnitudes (`+0.306/-0.327` at `c=0.5`; `+0.906/-0.910` at `c=1.0`) and coherence is clean (`pmass=1.000`, unscorable `0`, min margin/base `0.800`). The README plot images were regenerated from this run with the shared coherent c path `[-1,-0.5,0,+0.5,+1]`.
- 2026-07-01: MFQ-2 per-sample bootstrap from the pueue 423 run shows sign stability even at `N=1`: every bootstrap draw has the intended Authority direction for `c in {-1,-0.5,+0.5,+1}`. `N=4` or `N=8` tightens uncertainty, but the minimum sign-stable `N` in this run is `1`.
