"""Sampling readout for API models that do NOT expose token logprobs (OpenRouter / OpenAI-compatible).

The frontier models on the Economist's WVS chart (GPT-5.4, Claude, Gemini, Grok) return no
next-token distribution, so read.py's forced-slot logprob reader cannot touch them. This is the
fallback the Economist itself used ("average of ten responses"): SAMPLE N chat completions at
temperature and take the empirical answer frequency as the per-(item, frame) categorical `p`.

Why this stays comparable to the logprob reader: temperature-1 sampling is an unbiased estimator of
the model's softmax categorical, so the empirical `p` -> the true next-token `p` as N grows, and
E = sum_k k*p_k (readouts.expected_score) converges to the logprob E for the SAME model. Every
downstream summary (per_item_categorical, reduce_ordinal/nominal, expected_score, entropy) is a pure
function of `p`, so a sampled model drops onto the SAME map as a logprob model. N sets the
resolution: SE on a p~=0.5 cell is ~0.5/sqrt(N) (N=10 -> 0.16, coarse; N=100 -> 0.05).

What this CANNOT produce, by design (do not add it back):
- logit_contrast C and logodds_agree LO: log of an empirical frequency has -inf on any unsampled
  token; smoothing those zeros would fabricate the very fine steer signal C/LO exist to measure.
- full-vocab pmass_allowed: we cannot see mass on unsampled tokens. The sampling ANALOGUE we do
  report is the fraction of draws that PARSED to an allowed token -- the empirical refusal /
  off-format rate. At total collapse (nothing parses) `p` is NaN, matching read.py's
  NaN-at-collapse ("do not compare"), not an eps fallback.

Requires OPENROUTER_API_KEY. temperature MUST be > 0: at temp 0 the N draws collapse to one argmax
and E degenerates to an integer.
"""
from __future__ import annotations

import os
import re
from math import inf

import numpy as np
from loguru import logger
from openai import OpenAI

from .instrument import Instrument, InstrItem
from .read import build_user_content

_ANSWER_SUFFIX = ("\n\nAnswer with ONLY one option from [{space}] -- a single token, nothing else: "
                  "no words, no punctuation, no explanation.")


def _client() -> OpenAI:
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


def parse_answer(text: str, answer_space: list[str]) -> str | None:
    """The EARLIEST-appearing answer_space token in `text` (token-boundaried so '3' is not caught
    inside 'x3y', and a longer token wins a tie at the same position). None if no allowed token
    appears -- a refusal or off-format draw, counted against the parse rate, never coerced."""
    best, best_key = None, (inf, 0)
    for a in answer_space:
        m = re.search(rf"(?<![0-9A-Za-z]){re.escape(a)}(?![0-9A-Za-z])", text)
        if m and (m.start(), -len(a)) < best_key:
            best_key = (m.start(), -len(a))
            best = a
    return best


def _sample_texts(client: OpenAI, model: str, prompt: str, n_samples: int, temperature: float,
                  max_tokens: int) -> list[str]:
    """N completions, requesting up to 8 per call via the `n` param and topping up until N."""
    texts: list[str] = []
    while len(texts) < n_samples:
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=temperature, n=min(n_samples - len(texts), 8), max_tokens=max_tokens)
        texts.extend((c.message.content or "") for c in resp.choices)
    return texts


def read_items_sampled(model: str, instr: Instrument, items: list[InstrItem], *,
                       n_samples: int = 20, temperature: float = 1.0, max_tokens: int = 32,
                       verbose_first: bool = False) -> list[dict]:
    """Per-(item, frame) rows shaped for `instrument.per_item_categorical`: id, frame, lp, p,
    pmass_allowed, dimension, sign, human_label. `p` is the empirical answer frequency over
    `instr.answer_space`; `pmass_allowed` is the parse rate; `lp = log(p)` carries -inf on unsampled
    tokens ON PURPOSE so any C/LO consumer fails loudly instead of reading a fabricated number."""
    assert temperature > 0, "sampling readout needs temperature > 0; temp 0 collapses E to an integer"
    client = _client()
    space = instr.answer_space
    A = len(space)
    idx = {a: k for k, a in enumerate(space)}
    out: list[dict] = []
    for it_n, it in enumerate(items):
        user = build_user_content(instr, it)
        prompt = user + _ANSWER_SUFFIX.format(space="/".join(space))
        texts = _sample_texts(client, model, prompt, n_samples, temperature, max_tokens)
        counts = np.zeros(A)
        for t in texts:
            a = parse_answer(t, space)
            if a is not None:
                counts[idx[a]] += 1
        n_parsed = int(counts.sum())
        pmass = n_parsed / len(texts)
        p = counts / n_parsed if n_parsed else np.full(A, np.nan)   # NaN at collapse, by design
        with np.errstate(divide="ignore"):
            lp = np.log(p)
        out.append({
            "id": it.id, "frame": it.frame, "lp": lp, "p": p, "pmass_allowed": pmass,
            "dimension": it.dimension, "sign": it.sign, "human_label": it.human_label,
            "n_samples": len(texts), "n_parsed": n_parsed,
        })
        if verbose_first and it_n == 0:
            logger.debug(
                f"\n=== TRACE read_api first item ({instr.name}, {model}, N={len(texts)}) ===\n"
                f"--- prompt ---\n{prompt}\n"
                f"--- first 3 raw replies ---\n{texts[:3]}\n"
                f"--- counts over {space} ---\n{counts.tolist()}  parsed={n_parsed}/{len(texts)}\n"
                f"SHOULD: replies are a bare token in {space}; parse rate near 1.0 -> coherent. "
                f"ELSE the model is refusing / adding prose / the option set is off.\n")
    return out
