"""Approximate Inglehart-Welzel axes over the GlobalOpinionQA WVS items. -- authored by Claude

NOT the verbatim WVS factor scores. The canonical IW battery is 10 items (5 per axis); national
pride, respect-for-authority, materialist/postmaterialist priorities and happiness are absent or too
sparse in GlobalOpinionQA, so this is a keyword-selected APPROXIMATION: two axes, a few items each,
each item oriented to its axis-positive pole by reading the OPTION text -- so a reversed option order
(one child-quality row is stored ['Not mentioned','Important']) cannot silently flip a country.

  X = Survival (0) <-> Self-expression (1)
  Y = Traditional (0) <-> Secular-Rational (1)

An entity's axis value = mean over that axis's covered items of `positiveness` in [0,1], where
positiveness is the answer's expected position toward the axis-positive pole (0 = opposite pole,
1 = the pole). Each item is keyed by a unique trailing substring of the full WVS question (the
sub-item name GlobalOpinionQA appends after the shared stem).
"""
from __future__ import annotations

import re

import numpy as np

# drop non-substantive options before renormalizing (DK / refusals / the "Missing"/"INAP" fillers
# GlobalOpinionQA leaves in some option lists).
SKIP = re.compile(r"don'?t know|no answer|refus|decline|none of|not applicable|^other|missing|inap",
                  re.I)

X_AXIS = "Survival <-> Self-expression"
Y_AXIS = "Traditional <-> Secular-Rational"

# axis -> [(question-text suffix uniquely naming the WVS sub-item, option substring naming the
# axis-POSITIVE pole)]. Suffix is matched by str.endswith on the full question; pole must be an
# endpoint option (both asserted in resolve_items).
AXIS_ITEMS = {
    Y_AXIS: [
        ("Religion", "Not at all important"),          # importance of religion (secular = not important)
        ("God", "No"),                                 # believe in God (secular = No)
        ("Abortion", "Always justifiable"),            # abortion (secular = justifiable)
        ("Obedience", "Not mentioned"),                # child quality: obedience (secular = don't teach)
        ("Independence", "Important"),                 # child quality: independence (secular = teach)
        ("Determination, perseverance", "Important"),  # child quality: determination (secular = teach)
        ("Imagination", "Important"),                  # child quality: imagination (secular = teach)
    ],
    X_AXIS: [
        ("Homosexuality", "Always justifiable"),                 # homosexuality (self-expr = justifiable)
        ("dealing with people?", "Most people can be trusted"),  # interpersonal trust (self-expr = trust)
        ("Signing a petition", "Have done"),                     # political action (self-expr = have done)
        ("Attending peaceful demonstrations", "Have done"),      # political action
        ("Joining in boycotts", "Have done"),                    # political action
    ],
}


def e_frac(dist: np.ndarray, n: int) -> float:
    """Expected normalized option position in [0,1]: sum_k (k/(n-1)) * p_k, k = 0..n-1."""
    k = np.arange(n, dtype=float) / (n - 1)
    return float((np.asarray(dist, dtype=float) * k).sum())


def positiveness(dist: np.ndarray, pole_idx: int, n: int) -> float:
    """Position toward the axis-positive pole in [0,1]. pole at the LAST option -> e_frac; at the
    FIRST option -> 1 - e_frac. Only endpoint poles are meaningful for a signed axis."""
    e = e_frac(dist, n)
    if pole_idx == n - 1:
        return e
    if pole_idx == 0:
        return 1.0 - e
    raise ValueError(f"pole must be an endpoint option, got idx {pole_idx} of {n}")


def resolve_items(recs: list[dict]) -> dict[str, list[dict]]:
    """recs: [{q, opts, dist}]. Resolve each AXIS_ITEMS entry to exactly one question; fail loud on
    0 or >1 selector matches, an ambiguous pole, or a non-endpoint pole. Returns axis -> list of
    {suffix, rec, pole_idx, n}."""
    resolved: dict[str, list[dict]] = {}
    for axis, items in AXIS_ITEMS.items():
        rl = []
        for suffix, pole_sub in items:
            hits = [r for r in recs if r["q"].strip().endswith(suffix)]
            assert len(hits) == 1, f"{axis}: suffix {suffix!r} matched {len(hits)} questions (want 1)"
            r = hits[0]
            n = len(r["opts"])
            pole = [i for i, o in enumerate(r["opts"]) if pole_sub.lower() in o.lower()]
            assert len(pole) == 1, f"{suffix!r}: pole {pole_sub!r} matched options {pole} of {r['opts']}"
            assert pole[0] in (0, n - 1), f"{suffix!r}: pole not an endpoint (idx {pole[0]} of {n})"
            rl.append({"suffix": suffix, "rec": r, "pole_idx": pole[0], "n": n})
        resolved[axis] = rl
    return resolved
