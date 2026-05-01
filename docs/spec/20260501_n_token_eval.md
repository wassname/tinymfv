# N-token Evaluation

## Goal
Make eval score after a short deterministic continuation instead of only from the immediate next token, so steering effects that emerge over a few tokens show up in the metric.

## Scope
In: core eval scoring, evaluate API, eval CLI, smoke verification.
Out: changing vignette data, changing report aggregation, adding stochastic rollouts.

## Requirements
- R1: Eval must support a fixed continuation budget before scoring `true`/`false`. Done means: `evaluate(..., think_tokens=N)` uses a rollout path and records `think_tokens=N`. VERIFY: a smoke run with `N>0` reports `mode=guided_rollout` and different headline scores than `N=0` on the same subset.
- R2: Default behavior must stay available for zero-token scoring. Done means: `think_tokens=0` still uses the current teacher-forced path. VERIFY: smoke run reports `mode=next_token` with `think_tokens=0`.
- R3: CLI must expose the token budget. Done means: `scripts/03_eval.py --think-tokens N` works and writes the budget into output JSON. VERIFY: output JSON contains `think_tokens` and `eval_mode`.

## Tasks
- [/] T1 (R1,R2,R3): patch core scoring and CLI
  - steps: add rollout scorer, thread config through `evaluate`, add CLI arg and metadata
  - verify: run eval twice on a tiny subset, once with `--think-tokens 0` and once with `--think-tokens 8`
  - success: metadata differs by mode and at least one headline score differs
  - likely_fail: flag is accepted but dead code still uses next-token path, scores identical and metadata unchanged
  - sneaky_fail: rollout happens but scoring still reads the old prompt position, metadata changes but scores remain indistinguishable from teacher-forced
  - UAT: when I run the two eval commands, I see mode + think_tokens in the output JSON and a score delta
- [ ] T2: fresh-eyes review
  - steps: hand diff and smoke evidence to subagent
  - verify: reviewer explicitly checks likely and sneaky failure modes
  - success: reviewer says evidence distinguishes rollout path from dead code path

## Context
- Current eval scores only `out.logits[:, -1]` from the prompt ending at the JSON prefill.
- The gist proposes a deterministic guided rollout: generate N tokens, append a fixed answer prefill, then score at the forced answer position.
- For this repo we do not need `<think>` tags. We only need a short continuation budget before the JSON boolean answer.

## Log
- Guided rollout is the right abstraction here, but the minimal repo change is simpler than the gist: continue the assistant reply for N tokens, then append the existing boolean prefill and score there.

## TODO
- Consider logging a sample guided continuation when bool mass collapses, but keep that out of this patch unless needed.

## Errors
| Task | Error | Resolution |
|------|-------|------------|