"""Answer-token readout: administer an Instrument to a local LLM by reading the next-token
distribution at a prefilled answer slot.

Ported from the weight_steer_honesty experiment (`mft_honesty.admin.score_framing`) and
generalized to any ordinal `Instrument`: instead of hard-coding digits 1-5, we gather the mass
on the instrument's `answer_space` tokens after its `prefill`. Per InstrItem:

  build the chat prompt (user turn = schema_hint + statement, assistant prefill = instr.prefill),
  forward once, take the next-token log-probs, gather the prob on each answer token:

  - p        = renormalized distribution over answer_space (sums to 1). This is the per-(item,frame)
               categorical that `instrument.per_item_categorical` canonicalizes + averages.
  - pmass    = sum of raw (full-vocab) mass on the answer tokens: the coherence check. Drops when
               the model leaks to refusals / prose / gibberish, independent of which option it picks.

Framing (forward / inverted / negated) is carried by each InstrItem.frame; canonicalization to a
single forward orientation happens downstream in `per_item_categorical`, NOT here. The reader is
frame-agnostic: it just reports the presented-orientation distribution.

Single-token requirement: every answer token must encode to exactly one id given the tokenizer,
and they must be distinct. Verified for Qwen ('(1' -> ['(','1']). A pmass collapse (not an error
here) is the tell that the prefill merged with the option and the readout went blind.
"""
from __future__ import annotations

import numpy as np
import torch
from loguru import logger

from .guided import _rollout_natural_or_forced
from .instrument import Instrument, InstrItem


def resolve_answer_ids(tok, answer_space: list[str]) -> list[int]:
    ids = [tok.encode(a, add_special_tokens=False) for a in answer_space]
    assert all(len(x) == 1 for x in ids), f"answer token not single-token: {list(zip(answer_space, ids))}"
    flat = [x[0] for x in ids]
    assert len(set(flat)) == len(flat), f"answer token id collision: {dict(zip(answer_space, flat))}"
    return flat


def build_user_content(instr: Instrument, item: InstrItem) -> str:
    """User-turn content (NOT chat-templated, NOT prefilled).

    The user turn is `task\\n\\nStatement: <prompt>`. For ordinal surveys the task (response-scale
    legend) is FRAME-SPECIFIC -- inverted reverses the legend, negated negates the content -- so it
    travels per-item in `meta['task']`. Falls back to the instrument-level `schema_hint`, or the
    bare prompt (nominal vignettes).

    Chat-templating + the assistant prefill that forces the answer slot are applied downstream by
    `_rollout_natural_or_forced` (so the readout shares the nominal MFV generate-think-then-read
    core); this just assembles the legend + statement the model thinks about."""
    task = item.meta.get("task") or instr.schema_hint
    return f"{task}\n\nStatement: {item.prompt}" if task else item.prompt


@torch.no_grad()
def read_items(model, tok, instr: Instrument, items: list[InstrItem], answer_ids: list[int],
               *, max_think_tokens: int, batch_size: int = 36, n_samples: int = 1,
               temperature: float = 0.0, top_p: float = 1.0,
               verbose_first: bool = False) -> list[dict]:
    """Score a list of InstrItems (one frame's worth, or any subset). Returns per-item rows with
    the keys `per_item_categorical` consumes: id, frame, p, pmass_allowed, dimension, sign, human_label.

    Goes through the SAME `_rollout_natural_or_forced` core the nominal MFV forced-choice path uses,
    so steering registers: the model generates up to `max_think_tokens` think tokens, then we read
    the answer slot under the assistant prefill. The only per-instrument difference vs the nominal
    path is the answer-token set (`answer_ids`) + reducer (downstream). The legend already lives in
    the user-turn content, so we pass `schema_hint=""` and mirror the nominal nudge `"Just answer"`.

    `_rollout_natural_or_forced` forwards the whole batch at once, so we chunk `user_prompts` here
    and concatenate, preserving item order.

    Floor: `max_think_tokens >= 1`. The rollout core calls HF `generate(max_new_tokens=...)`, which
    rejects 0 (`max_new_tokens must be greater than 0`). think=1 is the minimum budget.
    """
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    out: list[dict] = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        user_prompts = [build_user_content(instr, it) for it in chunk]
        # ordinal frames are already separate InstrItems, so single-pass (no reversed-enum two-pass;
        # frame debias is downstream in canonicalize_to_forward). force_only: the "(" prefill is too
        # short for natural-emission detection (matches by chance in the think trace), so always read
        # the forced answer slot. n_samples>1 samples independent think traces, then averages the
        # answer-token probabilities below.
        thinks, slots = _rollout_natural_or_forced(
            model, tok, user_prompts,
            schema_hint="", max_think_tokens=max_think_tokens,
            scoring_slots=[("Just answer", instr.prefill)],
            gather_token_ids=answer_ids,
            n_samples=n_samples, temperature=temperature, top_p=top_p, force_only=True,
            verbose=verbose_first and i == 0,
        )
        for j, it in enumerate(chunk):
            sample_idx = [j * n_samples + n for n in range(n_samples)]
            sample_slots = [slots[k][0] for k in sample_idx]
            # lp = lp_gather: the full-vocab log_softmax logprob of each answer token at the answer
            # slot. This is the RAW PRIMITIVE -- every readout (E, the logit contrast C, log-odds,
            # entropy) is a pure function of it, and a steer effect is just a difference of lp. Keep
            # it; do not throw it away by collapsing to a single number here.
            sample_lp = np.asarray([s["lp_gather"] for s in sample_slots], dtype=float)  # [N,A]
            lp = np.log(np.nanmean(np.exp(sample_lp), axis=0))         # [A] BMA over sampled thoughts
            p_a = np.exp(lp)                                           # [A] prob on each answer token
            pmass = float(np.nansum(p_a))                              # mass on allowed tokens (coherence)
            # Renormalize within allowed. INTENTIONALLY NOT NaN-guarded: at full coherence collapse
            # pmass -> 0 so p_norm -> NaN and poisons that item's factor. That is the honest signal, a
            # distribution renormalized from ~zero mass is NOT comparable to one from real mass (the mean
            # of 10 != the mean of 130), so it must not be silently turned into a comparable-looking
            # number. NaN marks "do not compare". Do not "fix" this with a softmax/eps fallback.
            p_norm = p_a / p_a.sum()                                   # [A] within allowed (NaN at collapse, by design)
            sample_thinks = [thinks[k] for k in sample_idx]
            think_text, n_think, emitted_close = sample_thinks[0]
            out.append({
                "id": it.id, "frame": it.frame,
                "lp": lp,                                              # raw logprobs at the M scale tokens
                "sample_lp": sample_lp.tolist(),                        # [N,A] raw logprobs before BMA
                "sample_pmass_allowed": [float(s["pmass_allowed"]) for s in sample_slots],
                "sample_nll_prefill": [float(s["nll_prefill"]) for s in sample_slots],
                "p": p_norm,
                "pmass_allowed": pmass,
                "dimension": it.dimension, "sign": it.sign,
                "human_label": it.human_label,
                "think": think_text, "n_think": n_think,
                "emitted_close": any(t[2] for t in sample_thinks),
            })
        if verbose_first and i == 0:
            slot0 = slots[0][0]
            answer_p = {a: float(np.exp(lp)) for a, lp in zip(instr.answer_space, slot0["lp_gather"])}
            logger.debug(
                f"\n=== TRACE read first item ({instr.name}, think={max_think_tokens}) ===\n"
                f"--- top-5 next tokens at answer slot ---\n{slot0['top5_str']}\n"
                f"--- answer_space probs ---\n{answer_p}\n"
                f"SHOULD: top tokens are the answer_space {instr.answer_space} (format locked by the "
                f"{instr.prefill!r} prefill); pmass_allowed={float(slot0['pmass_allowed']):.3f} near 1.0 "
                f"-> coherent. ELSE the prefill or chat template is off.\n")
    return out
