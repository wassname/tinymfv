"""Diagnostic: reproduce the historical WORD-first-token MFV readout to confirm the
0.826 top-1 in the validation table (journal 2026-05-08), and contrast it with the
current debiased DIGIT readout (0.773).

This does NOT touch the canonical eval. tinymfv.evaluate stays digit-only. This is a
throwaway measurement (like the other probe_* scripts) that reuses the current
_rollout_natural_or_forced core but gathers the first token of each foundation WORD
(care/fair/loy/author/san/lib/social) instead of the option index digit -- the one
structural change commit e4e0f4d made when it switched word->digit to drop the
uneven-first-piece word prior. Everything else (prompt assembly, fwd/rev debias,
think budget, softmax, top-1 vs human argmax) mirrors evaluate().

If word-gather top-1 >> digit-gather top-1 (~0.83 vs 0.77), the gap is the readout
method, and the showcase model DOES match the validation table under the table's
own method.

  uv run python scripts/probe_word_readout.py --model Qwen/Qwen3-4B
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinymfv.data import load_vignettes
from tinymfv.eval import _label_dist
from tinymfv.guided import (
    _DEFAULT_FORCED_FOUNDATIONS, _FORCED_FOUNDATION_DESCS,
    _resolve_first_token_ids, _rollout_natural_or_forced,
)

NUDGE = "Just answer"
PREFILL = 'This is wrong because {"violation": "'


def _word_schema(foundations: list[str]) -> str:
    """The pre-e4e0f4d schema: enum key = foundation WORD, not the index digit."""
    opts = "\n".join('          "%s", # %s' % (f, _FORCED_FOUNDATION_DESCS[f]) for f in foundations)
    return ('Here is the JSON schema for the answer. This is wrong because:\n'
            '  {\n    "properties": {\n      "violation": {\n        "enum": [\n'
            + opts + '\n        ]\n      }\n    }\n  }')


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device).eval()

    foundations = list(_DEFAULT_FORCED_FOUNDATIONS)
    foundations_rev = list(reversed(foundations))
    K = len(foundations)
    # Gather the first token of each foundation WORD, same order both frames (the
    # pre-debias scheme). No reversal of the rev gather (that is the digit-mode fix).
    first_ids, _ = _resolve_first_token_ids(tok, foundations)
    schema_fwd, schema_rev = _word_schema(foundations), _word_schema(foundations_rev)

    vigs = load_vignettes("classic")
    rows = [(v, _label_dist(v, foundations)) for v in vigs]
    rows = [(v, lab) for v, lab in rows if lab is not None]
    prompts = [v["other_violate"] for v, _ in rows]

    y_pred, y_true, pmass = [], [], []
    for i in range(0, len(prompts), args.batch_size):
        chunk = prompts[i:i + args.batch_size]
        _, slots_f = _rollout_natural_or_forced(
            model, tok, chunk, schema_fwd, args.budget,
            scoring_slots=[(NUDGE, PREFILL)], gather_token_ids=first_ids,
            n_samples=1, temperature=0.0)
        _, slots_r = _rollout_natural_or_forced(
            model, tok, chunk, schema_rev, args.budget,
            scoring_slots=[(NUDGE, PREFILL)], gather_token_ids=first_ids,
            n_samples=1, temperature=0.0)
        for j in range(len(chunk)):
            lp_f = np.asarray(slots_f[j][0]["lp_gather"], dtype=float)
            lp_r = np.asarray(slots_r[j][0]["lp_gather"], dtype=float)  # no reversal: word gather
            score = (lp_f + lp_r) / 2.0
            y_pred.append(int(np.argmax(score)))
            pmass.append(0.5 * (slots_f[j][0]["pmass_allowed"] + slots_r[j][0]["pmass_allowed"]))
    y_true = [int(np.argmax(lab)) for _, lab in rows]

    y_pred, y_true = np.array(y_pred), np.array(y_true)
    top1 = float((y_pred == y_true).mean())
    logger.info(f"WORD readout: top1={top1:.3f}  mean_pmass={np.mean(pmass):.3f}  n={len(y_true)}\n"
                f"SHOULD: ~0.83 (matches the 82.6% validation table) if the word gather is the cause "
                f"of the digit readout's 0.77. Canonical evaluate() is unchanged (digit).")


if __name__ == "__main__":
    main()
