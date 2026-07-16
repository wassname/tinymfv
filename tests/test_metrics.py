"""Unit guard for the two shared steering metrics (moralmaps.metrics). No model.

Builds synthetic pos/neg clr rows for a selective Authority-down / Care-up steer and
checks: on>0, off small, the off_weight trade, the squared coherence barrier, and that
si_flips is a bounded behavioral pick-rate change. -- Claude
"""
from __future__ import annotations

import numpy as np
import pytest

from moralmaps.metrics import FOUNDATIONS, gated_selectivity, si_flips, clr_per_row

INTENT = {"authority": -1, "care": +1}  # steer Authority down, Care up


def _row(auth: float, care: float) -> dict[str, float]:
    """A clr row: authority/care set, others 0. (clr rows need not sum to 0 for the metric,
    which only reads per-foundation Deltas; realistic rows are centered but that is irrelevant here.)"""
    d = {f: 0.0 for f in FOUNDATIONS}
    d["authority"] = auth
    d["care"] = care
    return d


def _sweep(auth_p, care_p, auth_n, care_n, others_p=0.0, others_n=0.0, n=8):
    """n paired rows. pos = intended pole (auth low, care high), neg = opposite pole."""
    pos, neg = {}, {}
    for i in range(n):
        p = _row(auth_p, care_p)
        m = _row(auth_n, care_n)
        for f in FOUNDATIONS:
            if f not in INTENT:
                p[f] = others_p
                m[f] = others_n
        pos[f"vid{i}|other_violate"] = p
        neg[f"vid{i}|other_violate"] = m
    return pos, neg


def test_selective_steer_scores_high():
    # pos drives auth DOWN (-2) and care UP (+2); neg is the mirror; off-axis flat.
    pos, neg = _sweep(auth_p=-2.0, care_p=+2.0, auth_n=+2.0, care_n=-2.0)
    r = gated_selectivity(pos, neg, INTENT, pmass_pos=0.9, pmass_neg=0.9, pmass_base=0.9)
    # on = mean[ -1*(-2 - +2), +1*(+2 - -2) ] / 2 = mean[+4, +4] = +4 ; off = 0
    assert abs(r["on"] - 4.0) < 1e-9
    assert r["off"] < 1e-9
    assert r["coherence"] == 1.0                      # pmass preserved -> no discount
    assert abs(r["sel_gated"] - 4.0) < 1e-9
    assert r["ci_lo"] == r["ci_hi"] == 4.0            # zero row variance -> tight CI


def test_off_axis_is_soft_preference():
    """A sloppy steer (equal collateral on every off foundation) must still beat doing nothing,
    and only lose a little to a surgical steer -- off is weighted 0.1."""
    surgical, neg = _sweep(auth_p=-2.0, care_p=+2.0, auth_n=+2.0, care_n=-2.0, others_p=0.0)
    sloppy, neg2 = _sweep(auth_p=-2.0, care_p=+2.0, auth_n=+2.0, care_n=-2.0, others_p=4.0, others_n=0.0)
    rs = gated_selectivity(surgical, neg, INTENT, pmass_pos=0.9, pmass_neg=0.9, pmass_base=0.9)
    rl = gated_selectivity(sloppy, neg2, INTENT, pmass_pos=0.9, pmass_neg=0.9, pmass_base=0.9)
    assert rl["off"] > 0                               # collateral registered
    assert rl["sel_gated"] > 0                         # still beats inaction (sel=0)
    assert rs["sel_gated"] > rl["sel_gated"]           # surgical wins
    assert (rs["sel_gated"] - rl["sel_gated"]) < 0.5 * rs["sel_gated"]  # but only mildly (0.1 weight)


def test_coherence_barrier_squared():
    pos, neg = _sweep(auth_p=-2.0, care_p=+2.0, auth_n=+2.0, care_n=-2.0)
    # worst arm pmass halves vs base -> coherence 0.5 -> sel scaled by 0.25.
    r = gated_selectivity(pos, neg, INTENT, pmass_pos=0.45, pmass_neg=0.9, pmass_base=0.9)
    assert abs(r["coherence"] - 0.5) < 1e-9
    assert abs(r["sel_gated"] - 4.0 * 0.25) < 1e-9
    # exceeding base coherence never rewards (clamped at 1).
    r2 = gated_selectivity(pos, neg, INTENT, pmass_pos=0.99, pmass_neg=0.99, pmass_base=0.9)
    assert r2["coherence"] == 1.0


@pytest.mark.parametrize("invalid_pmass", [float("nan"), float("inf"), -0.1])
def test_coherence_inputs_must_be_valid(invalid_pmass):
    pos, neg = _sweep(auth_p=-2.0, care_p=+2.0, auth_n=+2.0, care_n=-2.0)
    with pytest.raises(AssertionError, match="pmass inputs"):
        gated_selectivity(
            pos, neg, INTENT,
            pmass_pos=invalid_pmass, pmass_neg=0.9, pmass_base=0.9,
        )


def test_metrics_require_exact_nonempty_row_pairing():
    pos, neg = _sweep(auth_p=-2.0, care_p=+2.0, auth_n=+2.0, care_n=-2.0)
    neg.pop(next(iter(neg)))
    with pytest.raises(AssertionError, match="row keys differ"):
        gated_selectivity(pos, neg, INTENT, pmass_pos=0.9, pmass_neg=0.9, pmass_base=0.9)
    with pytest.raises(AssertionError, match="row keys differ"):
        si_flips(pos, neg, INTENT)

    with pytest.raises(AssertionError, match="rows are empty"):
        gated_selectivity({}, {}, INTENT, pmass_pos=0.9, pmass_neg=0.9, pmass_base=0.9)
    with pytest.raises(AssertionError, match="rows are empty"):
        si_flips({}, {}, INTENT)


def test_si_flips_behavioral_bounded():
    # pos makes care the argmax pick, neg makes authority the pick.
    pos, neg = _sweep(auth_p=-2.0, care_p=+3.0, auth_n=+3.0, care_n=-2.0)
    r = si_flips(pos, neg, INTENT)
    # care: intent +1, rate(pos)=1, rate(neg)=0 -> +1 ; authority: intent -1, rate(pos)=0,
    # rate(neg)=1 -> -1*(0-1)=+1 ; mean = +1.
    assert abs(r["si_flips"] - 1.0) < 1e-9
    assert -1.0 <= r["si_flips"] <= 1.0
