"""Named 2-axis "value map" groupings per instrument -- the interpretable alternative to the blind
ipsative PCA map. Each instrument gets two axes with FOUR named pole directions (like the WVS
Inglehart-Welzel map), so a point's position reads directly. -- authored by Claude

An axis is a list of (factor, sign): the axis score of a 0-1 fraction profile is the mean over its
factors of the endorsement (sign +1) or its complement 1-endorsement (sign -1). So a factor entered
with -1 is reverse-scored (big5 Stability reverses neuroticism; a contrast axis puts one pole's
factors at -1). High score = the axis's POSITIVE (second) pole.

Sources: MFT individualizing/binding -- Graham & Haidt; MFQ-2 equality/proportionality fairness split
-- Atari et al. 2023. Big Five meta-traits Plasticity/Stability -- DeYoung 2007. HSQ 2x2
adaptive/maladaptive x self/other -- Martin et al. 2003. WVS -- Inglehart-Welzel (see tinymfv.iw_axes;
the WVS map builds its own item-level axes, this table is for the psychometric instruments).

The debatable calls (flagged): the SECOND MFT axis is not canonical -- mfq2 uses the documented
equality(egalitarian) vs proportionality(meritocratic) fairness split; mfv (no equality/proportionality
factors) uses liberty(autonomy) vs authority(hierarchy). Rename/retune freely; this is just data.
"""
from __future__ import annotations

import numpy as np

# instrument -> (x_axis, y_axis); each axis = (neg_pole_label, pos_pole_label, [(factor, sign)]).
VALUE_AXES: dict[str, tuple] = {
    "mfq2": (
        ("Individualizing", "Binding",
         [("care", -1), ("equality", -1), ("proportionality", -1),
          ("loyalty", 1), ("authority", 1), ("purity", 1)]),
        ("Equality", "Proportionality", [("equality", -1), ("proportionality", 1)]),
    ),
    "mfv": (
        ("Individualizing", "Binding",
         [("care", -1), ("fairness", -1), ("liberty", -1),
          ("authority", 1), ("loyalty", 1), ("sanctity", 1)]),
        ("Liberty", "Authority", [("liberty", -1), ("authority", 1)]),
    ),
    "big5": (
        ("low Plasticity", "Plasticity", [("extraversion", 1), ("openness", 1)]),
        ("low Stability", "Stability",
         [("agreeableness", 1), ("conscientiousness", 1), ("neuroticism", -1)]),
    ),
    "humor_styles": (
        ("Maladaptive", "Adaptive",
         [("affiliative", 1), ("selfenhancing", 1), ("aggressive", -1), ("selfdefeating", -1)]),
        ("Other-directed", "Self-directed",
         [("selfenhancing", 1), ("selfdefeating", 1), ("affiliative", -1), ("aggressive", -1)]),
    ),
}


def axis_score(profile_frac: np.ndarray, dims: list[str], axis: list[tuple]) -> float:
    """Mean signed endorsement over an axis's factors (sign -1 -> 1 - endorsement). Input is a 0-1
    fraction profile in `dims` order."""
    idx = {d: i for i, d in enumerate(dims)}
    vals = [profile_frac[idx[f]] if s > 0 else 1.0 - profile_frac[idx[f]] for f, s in axis]
    return float(np.mean(vals))


def value_coords(M: np.ndarray, dims: list[str], name: str) -> tuple[np.ndarray, tuple[str, str, str, str]]:
    """(P[n,2], pole_labels) for instrument `name`: project each 0-1 fraction row of M onto its two
    named value axes. pole_labels = (x_neg, x_pos, y_neg, y_pos)."""
    (xn, xp, xa), (yn, yp, ya) = VALUE_AXES[name]
    P = np.array([[axis_score(row, dims, xa), axis_score(row, dims, ya)] for row in M])
    return P, (xn, xp, yn, yp)
