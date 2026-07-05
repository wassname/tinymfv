"""Sampling readout for API models that do NOT expose token logprobs (OpenRouter / OpenAI-compatible).

Chat-completions-only API models return no next-token distribution, so read.py's forced-slot logprob
reader cannot touch them. Fallback (the "average of N responses" approach): SAMPLE N chat completions
at temperature and take the empirical answer frequency as the per-(item, frame) categorical `p`.

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

import asyncio
import json
import re
from math import inf

import numpy as np
from loguru import logger
from openrouter_wrapper.retry import openrouter_request   # stamina backoff on 429/provider/upstream errors

from .instrument import Instrument, InstrItem
from .read import build_user_content

_ANSWER_SUFFIX = ("\n\nAnswer with ONLY one option from [{space}] -- a single token, nothing else: "
                  "no words, no punctuation, no explanation.")


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


def _sample_texts(model: str, prompt: str, n_samples: int, temperature: float,
                  max_tokens: int) -> list[str]:
    """N completions via the openrouter_wrapper (stamina retry handles transient 429/provider/upstream
    errors), requesting up to 8 per call via `n` and topping up until N."""
    texts: list[str] = []
    while len(texts) < n_samples:
        # cap n at 4/call: a big n * max_tokens response is slow enough to trip httpx read timeouts
        # (which then exhaust the wrapper's retry budget); smaller requests are more reliable.
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": temperature, "n": min(n_samples - len(texts), 4),
                   "max_tokens": max_tokens}
        data = asyncio.run(openrouter_request(payload))
        texts.extend((c["message"].get("content") or "") for c in data["choices"])
    return texts


def read_items_sampled(model: str, instr: Instrument, items: list[InstrItem], *,
                       n_samples: int = 20, temperature: float = 1.0, max_tokens: int = 32,
                       verbose_first: bool = False) -> list[dict]:
    """Per-(item, frame) rows shaped for `instrument.per_item_categorical`: id, frame, lp, p,
    pmass_allowed, dimension, sign, human_label. `p` is the empirical answer frequency over
    `instr.answer_space`; `pmass_allowed` is the parse rate; `lp = log(p)` carries -inf on unsampled
    tokens ON PURPOSE so any C/LO consumer fails loudly instead of reading a fabricated number."""
    assert temperature > 0, "sampling readout needs temperature > 0; temp 0 collapses E to an integer"
    space = instr.answer_space
    A = len(space)
    idx = {a: k for k, a in enumerate(space)}
    out: list[dict] = []
    for it_n, it in enumerate(items):
        user = build_user_content(instr, it)
        prompt = user + _ANSWER_SUFFIX.format(space="/".join(space))
        texts = _sample_texts(model, prompt, n_samples, temperature, max_tokens)
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
            "prompt": prompt, "texts": texts,   # raw responses kept so a run's answers are auditable/saveable
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


# --- rated readout (dense, positional-bias-controlled) ------------------------------------------
_RATE_PROMPT = (
    "{question}\n\n"
    "Rate how strongly you personally agree with or endorse EACH of the following answers, on a "
    "scale of 1 (strongly reject) to 5 (strongly endorse):\n{legend}\n\n"
    'Respond with ONLY a JSON object mapping each answer\'s number to its 1-5 rating, e.g. '
    '{{"0": 2, "1": 5}}. Rate every answer, and output nothing but the JSON.')


def _parse_ratings(text: str, n: int) -> dict[int, float] | None:
    """The LAST brace-object in `text` parsed as {presented_label -> 1-5 rating}, or None if it is
    malformed / missing a label / out of range. LAST so a reasoning model that echoes the schema
    earlier doesn't win over its final answer."""
    objs = re.findall(r"\{[^{}]*\}", text)
    if not objs:
        return None
    try:
        raw = json.loads(objs[-1])
    except json.JSONDecodeError:
        return None
    out = {}
    for k in range(n):
        v = raw.get(str(k), raw.get(k))
        if v is None or not (1 <= float(v) <= 5):
            return None
        out[k] = float(v)
    return out


_FORCE_MSG = ('You are out of time. Output ONLY a single-line compact JSON object mapping each answer '
              'number to its 1-5 rating, e.g. {{"0": 1, "1": 5}}. No markdown, no reasoning, nothing else.')


async def _force_answer(model: str, prompt: str, phase1_msg: dict, temperature: float,
                        max_tokens: int, req_timeout: float) -> str:
    """Phase-2 rescue (wassname's bounded-thinking pattern, gist 72eed3a1): a reasoning model that
    spent its whole budget thinking and truncated the JSON mid-object gets a follow-up in the SAME
    conversation -- feed its (truncated) reasoning back as the assistant turn, then demand a compact
    one-line answer NOW. It has already thought, so it just commits. Works even where reasoning can't
    be disabled (some providers, e.g. gemini-2.5-pro, MANDATE it and 400 on reasoning_effort=none), so
    we do NOT pass a reasoning-off knob -- we constrain the OUTPUT instead. A bigger cap than phase 1
    (reasoning models re-think briefly). Still parsed by the caller; may fail again -> dropped sample."""
    tail = (phase1_msg.get("reasoning") or phase1_msg.get("content") or "")[-1500:] or "(thinking truncated)"
    msgs = [{"role": "user", "content": prompt},
            {"role": "assistant", "content": tail},
            {"role": "user", "content": _FORCE_MSG.format()}]
    payload = {"model": model, "messages": msgs, "temperature": temperature, "max_tokens": max(max_tokens, 2048)}
    data = await asyncio.wait_for(openrouter_request(payload), timeout=req_timeout)
    return data["choices"][0]["message"].get("content") or ""


def _rate_plan(items: list[dict], n_samples: int, per_call: int = 1) -> list[dict]:
    """Flatten (item, presented-order, count) into a list of <=per_call requests. A binary item splits
    its draws between the two orders (positional-bias control); an ordinal item keeps natural order
    (shuffling "Never..Always" is nonsense). per_call=1: OpenRouter providers do NOT reliably honour
    n>1 (they return a single completion), so one request per sample -- fine, they fire concurrently."""
    plan = []
    for i, it in enumerate(items):
        opts, n = it["options"], it["n"]
        groups = [([0, 1], (n_samples + 1) // 2), ([1, 0], n_samples // 2)] if n == 2 \
            else [(list(range(n)), n_samples)]
        for perm, tot in groups:
            legend = "\n".join(f"{j}) {opts[perm[j]]}" for j in range(n))
            prompt = _RATE_PROMPT.format(question=it["question"], legend=legend)
            while tot > 0:
                k = min(tot, per_call); tot -= k
                plan.append({"i": i, "perm": perm, "prompt": prompt, "cnt": k})
    return plan


def read_items_rated(model: str, items: list[dict], *, n_samples: int = 12, temperature: float = 1.0,
                     max_tokens: int = 512, concurrency: int = 8, req_timeout: float = 90.0,
                     verbose_first: bool = False) -> list[dict]:
    """Dense Likert readout: per item, ask the model to rate EVERY option 1-5 as JSON, N times, and
    normalize the mean rating to a per-option distribution `p`. Higher signal per call than a single
    forced choice, and positional bias is controlled by permuting the PRESENTED order of BINARY items
    (n==2) across samples then mapping ratings back to the canonical option order. All requests for the
    model fire CONCURRENTLY (asyncio.gather, capped at `concurrency`) so a 12-item panel is ~1 round
    trip, not 24 sequential ones. A reasoning model that burns its token budget thinking and truncates
    the JSON is rescued by a one-shot force-answer follow-up (_force_answer) instead of being dropped.

    `items`: [{"id", "question", "options"(canonical), "n"}]. Returns per item: id, p (mean over valid
    samples, canonical order), p_samples (per-sample canonical p arrays for bootstrap CIs),
    pmass_allowed (valid-JSON fraction), prompt, texts. p is NaN at total parse collapse (do not
    compare), matching the logprob reader. A failed request (network) just drops its samples."""
    plan = _rate_plan(items, n_samples)

    async def run_all() -> list:
        sem = asyncio.Semaphore(concurrency)
        async def call(req):
            async with sem:
                n = items[req["i"]]["n"]
                payload = {"model": model, "messages": [{"role": "user", "content": req["prompt"]}],
                           "temperature": temperature, "n": req["cnt"], "max_tokens": max_tokens}
                # per-request wall-clock cap: one request stuck in the wrapper's stamina backoff (a
                # rate-limited provider) must not stall the whole model's gather -- time it out and drop
                # it as a failed sample (return_exceptions catches the TimeoutError) so the panel moves on.
                data = await asyncio.wait_for(openrouter_request(payload), timeout=req_timeout)
                out = []
                for c in data["choices"]:
                    content = c["message"].get("content") or ""
                    if _parse_ratings(content, n) is None:   # truncated JSON / reasoning ate the budget
                        content = await _force_answer(model, req["prompt"], c["message"],
                                                      temperature, max_tokens, req_timeout)
                    out.append(content)
                return out
        return await asyncio.gather(*(call(r) for r in plan), return_exceptions=True)

    results = asyncio.run(run_all())
    agg = {i: {"p_samples": [], "texts": [], "prompt": ""} for i in range(len(items))}
    n_fail = 0
    for req, res in zip(plan, results):
        i, n, perm = req["i"], items[req["i"]]["n"], req["perm"]
        agg[i]["prompt"] = req["prompt"]
        if isinstance(res, Exception):
            n_fail += 1
            continue
        for text in res:
            agg[i]["texts"].append(text)
            rated = _parse_ratings(text, n)
            if rated is None:
                continue
            r_canon = np.zeros(n)
            for j in range(n):
                r_canon[perm[j]] = rated[j]      # map presented label -> canonical option
            agg[i]["p_samples"].append(r_canon / r_canon.sum())
    if n_fail:
        logger.warning(f"{model}: {n_fail}/{len(plan)} rating calls failed (network) -> fewer samples")

    out = []
    for i, it in enumerate(items):
        ps = agg[i]["p_samples"]
        n = it["n"]
        p = np.mean(ps, axis=0) if ps else np.full(n, np.nan)
        out.append({"id": it["id"], "p": p, "p_samples": [x.tolist() for x in ps],
                    "pmass_allowed": len(ps) / n_samples, "n_samples": n_samples,
                    "prompt": agg[i]["prompt"], "texts": agg[i]["texts"]})
        if verbose_first and i == 0:
            logger.debug(
                f"\n=== TRACE read_items_rated first item ({model}, N={n_samples}) ===\n"
                f"--- prompt ---\n{agg[i]['prompt']}\n"
                f"--- first 2 raw replies ---\n{agg[i]['texts'][:2]}\n"
                f"--- mean p over {it['options']} ---\n{np.round(p, 3).tolist()}  valid={len(ps)}/{n_samples}\n"
                f"SHOULD: replies are a bare JSON dict of 1-5 ratings; valid rate near 1.0 -> coherent. "
                f"ELSE the model is refusing / adding prose / max_tokens too small (empty content).\n")
    return out
