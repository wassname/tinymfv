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
- Do not use `dignity_over_authority` as the steer axis. That is a conflict axis and confounds Authority with dignity/care/wellbeing.
- Dignity, Care, Fairness, wellbeing, style, verbosity, refusal, and sycophancy are side-effect checks, not the axis definition.
- Persona/template/scenario validation comes before steering.
- Use the closest practical model for validation. Current evidence: persona-library used `qwen/qwen3-8b`, steer target is `Qwen/Qwen3-4B`; same family and close enough for first-pass selection, but weak validation should not be trusted.
- Queue GPU steer/eval with `pueue` only after selection looks sane.
- README is for readers. Keep journal, caveats, failed attempts, and work-in-progress methodology in this spec, not in README.
- README should eventually show only the successful artifacts: what was steered, what moved, which eval measured it, and why a researcher might use it.

## Scope
In:
- Define/test a proper `+Authority/-Authority` persona axis in persona-steering-template-library.
- Test templates and scenarios using the library validation workflow.
- Export selected scenarios into steering-lite, not only the persona library.
- Queue a steering-lite run only after validation passes clear gates.
- Run tinymfv evals on the resulting steer and inspect direction/noise/side effects.
- Update README plots/table after the steer and eval pass.

Out:
- README prose polish before the steer works.
- Treating strict22 `dignity_over_authority` as the final selection.
- General automatic sign labeling for arbitrary steering vectors.

## Requirements
- R1: The selected axis is `authority` only. Done means: selection artifacts name a `+Authority/-Authority` axis and do not define the negative pole as dignity/care/wellbeing. VERIFY: inspect selected axis JSON and prompt examples.
- R2: Template/scenario selection is validated before steering. Done means: summary reports template count, scenario sources, strict pass rate, axis delta, off-axis/style scores, and selected scenario count. VERIFY: selection summary JSON plus examples file.
- R3: The steering run uses selected data committed in steering-lite. Done means: steering-lite has `data/persona_library_selections/authority_only_*.jsonl` plus matching summary, and the runner references that file. VERIFY: `rg -n "authority_only" data/persona_library_selections scripts`.
- R4: Small-c tinymfv direction is correct before high-c plots matter. Done means: MFV Authority moves in the intended direction at the smallest coherent positive coefficient, and the reverse side moves the opposite way. VERIFY: effect table from steering-lite/tinymfv output.
- R5: Eval reliability is measured before README. Done means: MFV, MFQ-2, Humor, and Big Five report profile shift, reader-logit shift, and noise/CI where available; MFQ-2 uses sampled reads if needed. VERIFY: generated summary table.
- R6: README only shows successful artifacts. Done means: README plot captions name the Authority steer and use regenerated images from the final run. VERIFY: README image links and run-dir command point to the final output.

## Tasks
- [/] T1 (R1, R2): Define and validate the right axis in persona-library.
  - steps: add or run a `+Authority/-Authority` axis through the existing template/scenario validation workflow.
  - verify: print top template/axis rows and selected source counts.
  - success: best selected axis is Authority-only, not `dignity_over_authority`; selected scenarios are diverse across sources.
  - likely_fail: best row is still a conflict axis or all scenarios come from one source.
  - sneaky_fail: positive pole is just authoritarian/harmful or negative pole is just caring/helpful; catch by reading selected examples and off-axis scores.
  - UAT: open the selected examples file and see clear Authority contrast without dignity/care as the defining feature.
- [ ] T2 (R2, R3): Export the selected data into steering-lite.
  - steps: copy selected JSONL and summary into `steering-lite/data/persona_library_selections/`; add a runner for the authority-only selection.
  - verify: `rg -n "authority_only" /media/wassname/SGIronWolf/projects5/2026/lite/steering-lite/data/persona_library_selections /media/wassname/SGIronWolf/projects5/2026/lite/steering-lite/scripts`.
  - success: runner references authority-only selection and no strict22 path.
  - likely_fail: runner still references `authority_dignity_strict22`.
  - sneaky_fail: copied summary and JSONL disagree on template/axis; catch by parsing both.
  - UAT: file paths are clickable and committed in steering-lite.
- [ ] T3 (R4): Queue a steering-lite steer/eval only after T1/T2 pass.
  - steps: pueue a GPU job with label stating expected Authority movement and pass/fail resolve.
  - verify: pueue label includes why/resolve; output contains c-grid MFV profiles.
  - success: small positive `c` raises MFV Authority and small negative `c` lowers it.
  - likely_fail: sign is reversed or flat.
  - sneaky_fail: Authority moves only at incoherent high `c`; catch with coherence/path gate.
  - UAT: effect table shows the small-c Authority direction before any README plot update.
- [ ] T4 (R5): Evaluate reliability and side effects.
  - steps: run tinymfv summary over MFV, MFQ-2, Humor, Big Five; estimate noise/CI where available.
  - verify: table includes profile shift/human SD, reader-logit shift, and uncertainty/noise columns.
  - success: Authority signal is larger than noise and side effects are interpretable.
  - likely_fail: MFQ-2 noisy or opposite sign.
  - sneaky_fail: profile shift comes from loss of answer structure; catch with answer mass, survey contrast, and MFV margin.
  - UAT: one table lets the user decide whether the steer is good enough to show.
- [ ] T5 (R6): Update README only from the final successful run.
  - steps: regenerate all README plots from one final artifact; add concise table and captions.
  - verify: README image links resolve; no 16PF plot; no WIP methodology journal in reader prose.
  - success: reader sees what tinymfv is, what was steered, which datasets measured it, and why it matters.
  - likely_fail: README narrates failed strict22/debug history.
  - sneaky_fail: captions imply a general sign convention; catch by reading README without this spec.
  - UAT: external-review-v2 comprehension panel can explain what/where/why/measurement/dataset in its own words.

## Context
- Current wrong path: `dignity_over_authority` strict22. It was selected as the best dignity conflict axis, not the desired Authority-only axis.
- Current target: `+Authority <-> -Authority`.
- Current target model: `Qwen/Qwen3-4B`.
- Existing validation model evidence: `qwen/qwen3-8b`, same family and reasonably close, but weak strict-pass rates are not enough.
- Sign calibration is run-local. If the axis declares an Authority anchor, positive `c` can be oriented to that anchor. Without a declared anchor, plots should use `+c`/`-c` only.

## Log
- 2026-06-30: User corrected the axis. `dignity_over_authority` is not the goal; it confounds Authority with dignity/care/wellbeing. Use Authority-only.
- 2026-06-30: Killed pueue task 397 because it was spending GPU on README/eval plots for the old strict22 steer before the steer was known-good.

