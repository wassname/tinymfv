"""Unit tests for the instrument abstraction's pure functions (no model needed).

Covers the panel's flagged risks: frame canonicalization, the keying-vs-framing composition
(NOT a double-flip), renormalized p, ordinal expectation, nominal choice frequency, and the
negative control.
"""
import numpy as np

from tinymfv.instrument import (
    Instrument, InstrItem, canonicalize_to_forward, per_item_categorical,
    reduce_ordinal, shuffle_dimensions,
)

M = 5
ONEHOT = {d: np.eye(M)[d - 1] for d in range(1, M + 1)}  # ONEHOT[4] = mass on scale point 4


def _E(p, scale_max):  # expected scale point; reduce_ordinal computes this inline now
    return float((np.asarray(p) * np.arange(1, scale_max + 1)).sum())


def _ord_instr(items):
    return Instrument("t", "endorsement", "ordinal", ["1", "2", "3", "4", "5"],
                      ["care", "authority"], items, prefill="(")


def test_canonicalize():
    # forward + nominal -> identity; ordinal inverted/negated -> reversed vector
    p = ONEHOT[5]
    assert np.array_equal(canonicalize_to_forward(p, "forward", "ordinal"), p)
    assert np.array_equal(canonicalize_to_forward(p, "inverted", "ordinal"), ONEHOT[1])
    assert np.array_equal(canonicalize_to_forward(p, "negated", "ordinal"), ONEHOT[1])
    assert np.array_equal(canonicalize_to_forward(p, "inverted", "nominal"), p)  # nominal identity


def test_forward_expectation():
    rows = [{"id": "1", "frame": "forward", "p": ONEHOT[4], "pmass_allowed": 1.0,
             "dimension": "care", "sign": 1, "human_label": None}]
    items = per_item_categorical(rows, "ordinal")
    assert abs(_E(items["1"]["p"], M) - 4.0) < 1e-9
    prof = reduce_ordinal(items, _ord_instr([]))
    assert abs(prof[0] - 4.0) < 1e-9  # care
    assert np.isnan(prof[1])          # authority has no items


def test_frame_consistency():
    # Same item, forward vs inverted: after canonicalization the per-item categorical must agree.
    fwd = [{"id": "1", "frame": "forward", "p": ONEHOT[1], "pmass_allowed": 1.0,
            "dimension": "care", "sign": 1, "human_label": None}]
    inv = [{"id": "1", "frame": "inverted", "p": ONEHOT[5], "pmass_allowed": 1.0,
            "dimension": "care", "sign": 1, "human_label": None}]
    e_fwd = _E(per_item_categorical(fwd, "ordinal")["1"]["p"], M)
    e_inv = _E(per_item_categorical(inv, "ordinal")["1"]["p"], M)
    assert abs(e_fwd - 1.0) < 1e-9 and abs(e_inv - 1.0) < 1e-9  # canonicalization makes them agree


def test_keying_is_not_double_flip():
    # Reverse-keyed item ("I keep in the background", sign=-1) on an INVERTED scale.
    # Model puts mass on presented "5". Panel claimed frame+keying double-corrects; show it does not.
    # inverted: presented 5 -> canonical 1 (disagrees with the item) -> E_canon = 1.
    # keying sign<0 (pool-reflect): agreement = M+1 - 1 = 5 = HIGH on the factor. Correct:
    # disagreeing with "I keep in the background" == extraverted.
    rows = [{"id": "x", "frame": "inverted", "p": ONEHOT[5], "pmass_allowed": 1.0,
             "dimension": "care", "sign": -1, "human_label": None}]
    items = per_item_categorical(rows, "ordinal")
    assert abs(_E(items["x"]["p"], M) - 1.0) < 1e-9      # canonical agreement-with-item = 1
    prof = reduce_ordinal(items, _ord_instr([]))
    assert abs(prof[0] - 5.0) < 1e-9                                 # keyed factor score = 5, not 1 (no double flip)

    # And it matches the SAME reverse-keyed item shown forward with the mirrored answer (mass on 1):
    rows_fwd = [{"id": "x", "frame": "forward", "p": ONEHOT[1], "pmass_allowed": 1.0,
                 "dimension": "care", "sign": -1, "human_label": None}]
    prof_fwd = reduce_ordinal(per_item_categorical(rows_fwd, "ordinal"), _ord_instr([]))
    assert abs(prof[0] - prof_fwd[0]) < 1e-9


def test_frame_spread_diagnostic():
    # forward mass on 1, inverted mass on presented 1 (-> canonical 5): canonical frames disagree
    # maximally -> frame_spread = 2.0 (L1 between two disjoint one-hots).
    rows = [{"id": "1", "frame": "forward", "p": ONEHOT[1], "pmass_allowed": 1.0,
             "dimension": "care", "sign": 1, "human_label": None},
            {"id": "1", "frame": "inverted", "p": ONEHOT[1], "pmass_allowed": 1.0,
             "dimension": "care", "sign": 1, "human_label": None}]
    items = per_item_categorical(rows, "ordinal")
    assert abs(items["1"]["frame_spread"] - 2.0) < 1e-9


def test_negative_control_shuffle():
    rng = np.random.default_rng(0)
    rows = [{"id": str(i), "frame": "forward", "p": ONEHOT[(i % 5) + 1], "pmass_allowed": 1.0,
             "dimension": "care" if i < 5 else "authority", "sign": 1, "human_label": None}
            for i in range(10)]
    items = per_item_categorical(rows, "ordinal")
    shuffled = shuffle_dimensions(items, rng)
    # same item ids, dimensions permuted
    assert set(shuffled) == set(items)
    assert [shuffled[k]["dimension"] for k in items] != [items[k]["dimension"] for k in items]


def test_cross_scale_guard():
    # HSQ-like: human on 1-7, model on 1-5, with a human_label attached -> must raise.
    import pytest
    items = [InstrItem("1", "q", dimension="care", human_label=np.ones(5) / 5)]
    with pytest.raises(AssertionError):
        Instrument("hsq", "endorsement", "ordinal", ["1", "2", "3", "4", "5"],
                   ["care"], items, prefill="(", human_scale_max=7)
