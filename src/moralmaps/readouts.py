"""Ordinal Likert + categorical readouts: pure functions of the raw answer-token logprobs.

One primitive (`lp`, the full-vocab logprobs at the M scale tokens) -> many summaries. The split
matters because the summaries answer different questions and have very different steer-sensitivity:

  expected_score E   = sum_k k * p_k,  in [1, M]          human-comparable scale score
  logit_contrast C   = sum_k (k - mid) * lp_k             primary steer signal (sensitive, signed)
  logodds_agree  LO  = lse(top) - lse(bottom)             readable 2-bin direction summary
  entropy        H   = -sum_k p_k log p_k                 within-allowed coherence (uniform = ln M)

Why E hides steering and C/LO do not (the whole reason this module exists):

  dE/dl_j = p_j (j - E)      -> vanishes when the model is confident (p_j -> 0 in the tails, and
                                j - E -> 0 at the mode). A peaked distribution sits in a flat spot
                                of E, so a small steer that reallocates the tails barely moves it.
  dC/dl_j = (k - mid)        -> a fixed weight, NO p_j factor. Centered weights (sum = 0) also kill
                                the softmax normalizer, so C is normalizer-invariant (raw logits,
                                full-vocab logprobs, or within-M renormalized logprobs all give the
                                same C) and the steer effect is exactly linear: dC = w . dl.

So C is the log-space analog of E: same `sum (weight_k) * (per-token quantity)` shape, but in
logprobs with midpoint-centered weights. LO is C's 2-bin special case (weights +-1 on the poles,
neutral dropped, each pole pooled with logsumexp). Keep E only for landing the model against human
norms; use C (or LO) for "did the steer move it".
"""
from __future__ import annotations

import numpy as np


def _logsumexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    m = float(np.max(x))
    return m + float(np.log(np.exp(x - m).sum()))


def expected_score(p: np.ndarray, scale_max: int) -> float:
    """E = sum_k k * p_k, in [1, scale_max]. Human-comparable; INSENSITIVE near a confident answer."""
    w = np.arange(1, scale_max + 1, dtype=float)
    return float((np.asarray(p, dtype=float) * w).sum())


def logit_contrast(lp: np.ndarray, scale_max: int) -> float:
    """C = sum_k (k - mid) * lp_k, the rank-centered logit contrast (mid = (1 + scale_max) / 2).

    Primary ordinal steer readout. Sensitive (dC/dl_j = weight_j, no probability suppression),
    signed, unbounded, normalizer-invariant (weights sum to 0, so any constant offset on `lp`
    cancels), and linear in the logits so dC across a steer = weights . (lp_steered - lp_base)."""
    mid = (1 + scale_max) / 2.0
    w = np.arange(1, scale_max + 1, dtype=float) - mid
    return float((np.asarray(lp, dtype=float) * w).sum())


def logodds_agree(lp: np.ndarray, scale_max: int) -> float:
    """LO = logsumexp(top n_side) - logsumexp(bottom n_side), n_side = scale_max // 2.

    The readable 2-bin direction summary: nats in favor of agreeing over disagreeing among the
    non-neutral options. Drops the middle category; less information than C but interpretable."""
    lp = np.asarray(lp, dtype=float)
    n_side = scale_max // 2
    return _logsumexp(lp[-n_side:]) - _logsumexp(lp[:n_side])


def clr(score: np.ndarray) -> np.ndarray:
    """Centered log-ratio of a categorical evidence vector: clr_f = score_f - mean_j score_j.

    The nominal-category analog of `logit_contrast` for a K-way forced choice (e.g. the
    moral foundations), where there is no rank to weight by. `score` is the pre-softmax
    per-category evidence (nats); subtracting the mean is the compositional-data CLR
    transform. Like C it kills the softmax normalizer (constant offsets on `score` cancel,
    weights sum to 0), so it is gauge-free and sensitive (no p_j suppression).

    Use clr deltas -- not one-vs-rest log-odds `score_f - logsumexp(score_{j != f})` -- for
    per-category selectivity. The one-vs-rest logsumexp is dominated by the top category, so a
    steer that concentrates evidence on ONE category mechanically depresses every other
    category's one-vs-rest log-odds and fabricates off-axis collateral: a real, selective steer
    then scores as LESS selective than a diffuse random poke (jsteer authority gate, job 95).
    clr spreads a single-category shift as +(1 - 1/K) on that category and -1/K on each other,
    so genuine selectivity survives. -- added 2026-07-15 for the jsteer selectivity gate
    """
    s = np.asarray(score, dtype=float)
    return s - s.mean()


def entropy(p: np.ndarray, scale_max: int | None = None) -> float:
    """Shannon entropy of the within-allowed distribution, nats. Coherence the pmass gate misses:
    a uniform answer has pmass ~ 1 but entropy = ln(scale_max), the max."""
    p = np.asarray(p, dtype=float)
    return float(-(p * np.log(p + 1e-12)).sum())
