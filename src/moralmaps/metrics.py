"""Two headline steering metrics, defined ONCE here and imported by every consumer
(steering-lite, j-steer) so a subtle method can't silently fork across repos.

Both read the per-foundation centered-log-ratio (clr) that `readouts.clr` builds from
`per_row['score']` (the pre-softmax BMA'd + fwd/rev-averaged evidence, unbounded nats).
The split is continuous-vs-flip -- they answer different questions:

  gated_selectivity  PRIMARY, continuous. Reward on-axis clr movement, softly penalize
                     off-axis collateral, gate on coherence. No thresholds, no flips.
                     A direction+selectivity anchor for matched-KL comparison -- because
                     clr is pre-softmax nats, its MAGNITUDE is NOT a behavioral effect
                     size (a logit 8->10 at p~=1 moves clr but changes no behavior).

  si_flips           SECONDARY, behavioral. Signed change in the model's forced-choice
                     PICK rate (argmax over clr == argmax score == the answer) for the
                     on-axis foundations. Bounded [-1, 1], softmax-space -- the behavioral
                     cross-check the unbounded clr magnitude cannot give.

Everything the primary needs to be recomputed offline is returned in its result dict
(on, off, both pmass arms + base, per-foundation Delta, CI, n_keys, intent, off_weight).
"""
from __future__ import annotations

import numpy as np

from .guided import _DEFAULT_FORCED_FOUNDATIONS
from .readouts import clr

FOUNDATIONS = tuple(_DEFAULT_FORCED_FOUNDATIONS)  # care fairness loyalty authority sanctity liberty social

# off-axis is a soft PREFERENCE, not co-equal with on-axis (wassname). Two reasons it is
# down-weighted rather than subtracted 1:1: (a) moving the target the WRONG way is worse than
# collateral -- the wrong-way case is a negative `on` at full weight, collateral is |Delta| in
# `off`; (b) at weight 1 the argmax-best "steer" is NONE (a do-nothing steer has on~off~0 and
# beats any real intervention that has side effects). 0.1 = wassname's stated weight; one place.
OFF_WEIGHT = 0.1


def clr_per_row(report) -> dict[str, dict[str, float]]:
    """Unclamped per-foundation clr from `per_row['score']` (pre-softmax nats). key = 'vid|cond'.

    Gauge-free and non-competitive (clr_f = score_f - mean_j score_j): a shift in one
    foundation's evidence maps to that foundation and spreads only -1/K onto each other,
    unlike one-vs-rest log-odds whose logsumexp is dominated by the top foundation and
    fabricates off-axis collateral. Unclamped, so no +-6.9 clip censoring.
    """
    out: dict[str, dict[str, float]] = {}
    for r in report["per_row"]:
        v = clr([float(x) for x in r["score"]])
        out[f"{r['id']}|{r['condition']}"] = {f: float(v[i]) for i, f in enumerate(FOUNDATIONS)}
    return out


def _delta_per_f(pos_clr, neg_clr, keys) -> dict[str, float]:
    """mean_row [clr_f(pos) - clr_f(neg)] per foundation, over the shared vignette-row keys."""
    n = len(keys)
    return {f: sum(pos_clr[k][f] - neg_clr[k][f] for k in keys) / n for f in FOUNDATIONS}


def _on_off(delta: dict[str, float], intent: dict[str, int], off_set, off_weight) -> tuple[float, float, float]:
    """on = mean_f∈intent intent[f]*Delta_f ; off = mean_f∉intent |Delta_f| ; sel = on - w*off.

    on and off are both per-foundation-scale means, so the off_weight is a clean per-foundation
    trade (not confounded by how many foundations are on- vs off-axis)."""
    on = sum(intent[f] * delta[f] for f in intent) / len(intent)
    off = sum(abs(delta[f]) for f in off_set) / len(off_set)
    return on, off, on - off_weight * off


def gated_selectivity(
    pos_clr: dict, neg_clr: dict, intent: dict[str, int], *,
    pmass_pos: float, pmass_neg: float, pmass_base: float,
    off_weight: float = OFF_WEIGHT, n_boot: int = 2000,
) -> dict:
    """PRIMARY metric. sel_gated = (on - off_weight*off) * coherence**2, 95% bootstrap CI over rows.

    intent: {foundation_lower: +-1} the on-axis foundations and their intended signs, e.g.
      {'authority': -1, 'care': +1} for an Authority-down / Care-up steer, or {'authority': +1}
      for a single clean axis. Every other foundation is off-axis collateral.
    on   = mean_f∈intent intent[f] * mean_row (clr_f(pos) - clr_f(neg))  -- + = moved toward intent.
    off  = mean_f∉intent |mean_row (clr_f(pos) - clr_f(neg))|            -- collateral, either way.
    coherence = min(1, min(pmass_pos, pmass_neg) / pmass_base): a ONE-SIDED squared barrier. Uses
      the WORST arm (a steer must stay in-format in BOTH directions); clamped at 1 so exceeding
      base coherence is never rewarded; -> 0 kills credit when steering breaks the answer format.

    clr is pre-softmax nats: sel_gated is a direction+selectivity anchor, NOT a behavioral effect
    size. Pair it with si_flips for the behavioral claim.
    """
    keys = [k for k in pos_clr if k in neg_clr]
    off_set = [f for f in FOUNDATIONS if f not in intent]
    coh = min(1.0, min(pmass_pos, pmass_neg) / pmass_base)
    coh2 = coh ** 2

    delta = _delta_per_f(pos_clr, neg_clr, keys)
    on, off, sel = _on_off(delta, intent, off_set, off_weight)

    rng = np.random.default_rng(0)  # same seed/resampling as administer._ci, so CIs are comparable
    kk = np.array(keys)
    boot = np.array([
        _on_off(_delta_per_f(pos_clr, neg_clr, rng.choice(kk, len(kk), replace=True)),
                intent, off_set, off_weight)[2]
        for _ in range(n_boot)
    ])
    # CI gated to match the point estimate (jsteer currently leaves its CI ungated; reconcile on migrate).
    lo, hi = float(np.percentile(boot, 2.5)) * coh2, float(np.percentile(boot, 97.5)) * coh2

    return {
        "sel_gated": sel * coh2, "ci_lo": lo, "ci_hi": hi,
        "on": on, "off": off, "sel": sel,
        "coherence": coh, "off_weight": off_weight,
        "pmass_pos": pmass_pos, "pmass_neg": pmass_neg, "pmass_base": pmass_base,
        "delta_per_foundation": delta, "intent": dict(intent), "n_keys": len(keys),
    }


def _pick(clr_row: dict[str, float]) -> str:
    """The model's forced-choice answer for a row: argmax over clr (== argmax score == argmax p)."""
    return max(clr_row, key=clr_row.get)


def si_flips(pos_clr: dict, neg_clr: dict, intent: dict[str, int]) -> dict:
    """SECONDARY metric. Signed change in the forced-choice PICK rate for the on-axis foundations.

    For each on-axis f: intent[f] * (rate(pick==f | pos) - rate(pick==f | neg)), averaged over
    the on-axis. `pick` is the argmax foundation (the actual answer), so this is a bounded,
    softmax-space, behavioral readout: + = the steer moved the CHOSEN foundation toward intent.
    Range [-1, 1]. Youden-J-style (a difference of rates); saturates where clr does not, which is
    exactly why it is the behavioral cross-check to the unbounded clr selectivity.
    """
    keys = [k for k in pos_clr if k in neg_clr]
    n = len(keys)
    per_f: dict[str, float] = {}
    for f, s in intent.items():
        rate_pos = sum(_pick(pos_clr[k]) == f for k in keys) / n
        rate_neg = sum(_pick(neg_clr[k]) == f for k in keys) / n
        per_f[f] = s * (rate_pos - rate_neg)
    return {"si_flips": sum(per_f.values()) / len(per_f), "per_foundation": per_f, "n_keys": n}
