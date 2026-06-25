"""Instrument spec: one answer-token reader, many questionnaires (Option 3).

Every instrument is the SAME measurement: a softmax over an answer-token set at a prefilled
slot, with a think budget, debias, BMA over sampled traces, a pmass coherence check, and
temperature calibration to human soft-labels. Per-instrument variation is small:

  1. answer_space : tokens gathered. Nominal = foundation words (forced-choice MFV);
                    ordinal = scale points ['1'..'M'] (MFQ-2, Big5, 16PF, HSQ).
  2. reducer      : per-item canonical distribution -> profile over `dimensions`.
                    Nominal -> mean choice frequency (the answer IS the dimension).
                    Ordinal -> E[scale point] grouped by item dimension (expectation over the
                    integer distribution, not argmax), with reverse-keying applied HERE only.
  3. scaffold     : prompt + assistant prefill that forces the answer slot.
  4. human_label  : per-item forward-orientation distribution over answer_space.

The frame reflection must NOT live in the reducer: that makes nominal and ordinal debias
asymmetric and risks double-flipping. Both kinds unify on one rule:

  CANONICALIZE-AT-READER. Every frame's gathered distribution is mapped to ONE forward
  orientation before anything else (`canonicalize_to_forward`): nominal reorder frames reindex
  to canonical answer order (the reader already gathers in canonical order, so this is identity);
  ordinal `inverted`/`negated` frames reverse the probability vector (the distribution-level
  analog of agreement = M+1 - E). The per-item object is then the mean of these canonical
  distributions over frames -- a single forward-orientation categorical, used IDENTICALLY for:
    - metrics: soft-NLL / temperature T vs the forward human histogram (both kinds), plus
      an ORDINAL metric (mean |E_model - E_human|) so the certifying metric is sensitive to the
      expectation the profile actually uses; top1 / informedness only for nominal (an argmax
      flip metric is meaningless on an ordered scale: 4-vs-5 != 1-vs-5).
    - profile: the reducer. Keying (reverse-keyed items) is applied ONLY in reduce_ordinal,
      because it is a pooling-into-factor correction, orthogonal to and composable with the
      framing canonicalization -- not a double-correction.

`p` everywhere is renormalized over the allowed answer tokens (sums to 1); `pmass_allowed` is
the separate coherence check, never mixed into the distribution.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Kind = Literal["nominal", "ordinal"]
ORDINAL_REFLECT_FRAMES = {"inverted", "negated"}  # frames whose meaning is flipped vs forward


@dataclass
class InstrItem:
    id: str
    prompt: str                            # user-turn content (vignette, or task + statement)
    dimension: str | None = None           # ordinal: the factor; nominal: None (read from answer)
    sign: int = 1                          # ordinal keying: -1 = reverse-keyed (pool-reflect, profile only)
    frame: str = "forward"                 # forward | inverted | negated (canonicalized before use)
    human_label: np.ndarray | None = None  # forward-orientation dist over answer_space, sums to 1
    meta: dict = field(default_factory=dict)


@dataclass
class Instrument:
    name: str
    construct: str                         # map-layer tag: "salience" | "endorsement" | "wrongness" ...
    kind: Kind
    answer_space: list[str]                # tokens gathered at the answer slot
    dimensions: list[str]                  # profile axes for the map
    items: list[InstrItem]
    prefill: str                           # assistant prefill that forces the answer slot
    schema_hint: str | None = None         # instruction appended to the user turn
    answer_to_dim: dict[str, str] | None = None  # nominal only: answer token -> dimension
    scale_max: int = 5                     # ordinal: top of the Likert scale (== len(answer_space))
    human_scale_max: int = 5               # human reference scale; HSQ humans on 1-7 (see caveat below)
    display: str = ""                      # human-readable name for plot titles ("Big Five")
    human_csv: str | None = None           # per-country subscale means CSV (the map's cross-cultural cloud)

    def __post_init__(self):
        if self.kind == "ordinal":
            assert self.answer_space == [str(i) for i in range(1, self.scale_max + 1)], (
                f"ordinal answer_space must be ['1'..'{self.scale_max}'] IN ORDER -- reduce_ordinal "
                f"weights by position (w = 1..scale_max), so a reordered space silently inverts E; "
                f"got {self.answer_space}")
        if self.kind == "nominal" and self.answer_to_dim is None:
            self.answer_to_dim = {a: a for a in self.answer_space}
        # Cross-scale caveat: a 1-7 human histogram (HSQ) cannot share a 5-way soft-NLL
        # with a 1-5 model directly. Calibration for such
        # instruments must project both to a common support (or report 0-1 endorsement only).
        # Enforced loudly rather than silently mis-comparing:
        if self.kind == "ordinal" and self.human_scale_max != self.scale_max:
            assert all(it.human_label is None for it in self.items), (
                f"{self.name}: human_scale_max={self.human_scale_max} != scale_max={self.scale_max}; "
                "per-item categorical calibration needs a common support. Project the human "
                "histogram to the model scale before attaching human_label, or leave it None.")


# --- canonicalization: every frame -> one forward orientation, BEFORE metrics or reducer ---

def canonicalize_to_forward(p: np.ndarray, frame: str, kind: Kind) -> np.ndarray:
    """Map a presented-orientation answer distribution to forward orientation.

    nominal: reader gathers in canonical answer order already -> identity (reorder frames are
             averaged as aligned probability vectors upstream).
    ordinal: `inverted` (scale legend reversed) and `negated` (content negated) flip meaning, so
             reverse the vector: p_forward[d] = p_presented[M+1-d]. This is the distribution-level
             form of agreement = M+1 - E. NB negation is not a guaranteed-exact semantic
             complement (panel caveat); the frame-disagreement diagnostic surfaces items where it
             breaks. `forward` -> identity.
    """
    p = np.asarray(p, dtype=float)
    if kind == "ordinal" and frame in ORDINAL_REFLECT_FRAMES:
        return p[::-1].copy()
    return p


def per_item_categorical(per_row: list[dict], kind: Kind) -> dict[str, dict]:
    """Collapse (item, frame) rows to one forward-orientation categorical per item id.

    Each row has: id, frame, lp (raw logprobs at the M tokens), p (renormalized over answer_space),
    pmass_allowed, dimension, sign, human_label. Returns {id: {lp, p, pmass, dimension, sign,
    human_label, n_frames, frame_spread}} where p is the mean of canonicalized frame probability
    vectors (kept for E + the human-comparison maps, preserving the NaN-at-collapse signal), lp is
    the mean of canonicalized frame logprobs (the log-space primitive for the contrast C + log-odds;
    averaging in log space is exact for the linear contrast), and frame_spread is the max L1 gap
    between any two canonical probability frames (the acquiescence/negation diagnostic).
    """
    by_id: dict[str, list[dict]] = defaultdict(list)
    for r in per_row:
        by_id[r["id"]].append(r)
    # Each item collapses to ONE averaged distribution, so every item contributes EQUALLY to the
    # factor mean. That matches pooling all (item, frame) rows only if every item has the same frame
    # count. Assert it loudly rather than silently reweighting a factor if a future instrument gives
    # some items fewer frames.
    frame_counts = {len(rows) for rows in by_id.values()}
    assert len(frame_counts) == 1, f"heterogeneous frame counts per item: {frame_counts}"
    out: dict[str, dict] = {}
    for iid, rows in by_id.items():
        # Per-item rows differ only by frame; dimension + sign must agree, frames must be distinct.
        # Else we would silently average incompatible distributions under rows[0]'s metadata.
        assert len({r.get("dimension") for r in rows}) == 1, f"{iid}: inconsistent dimension across frames"
        assert len({r.get("sign", 1) for r in rows}) == 1, f"{iid}: inconsistent sign across frames"
        assert len({r["frame"] for r in rows}) == len(rows), f"{iid}: duplicate frame in rows"
        canon = [canonicalize_to_forward(r["p"], r["frame"], kind) for r in rows]
        canon_lp = [canonicalize_to_forward(r["lp"], r["frame"], kind) for r in rows]
        C = np.stack(canon)
        spread = float(max((np.abs(C[i] - C[j]).sum()
                            for i in range(len(C)) for j in range(i + 1, len(C))), default=0.0))
        r0 = rows[0]
        out[iid] = {
            "lp": np.stack(canon_lp).mean(axis=0),
            "p": C.mean(axis=0),
            "pmass": float(np.mean([r["pmass_allowed"] for r in rows])),
            "dimension": r0.get("dimension"),
            "sign": r0.get("sign", 1),
            "human_label": r0.get("human_label"),
            "n_frames": len(rows),
            "frame_spread": spread,
        }
    return out


# --- profile reducers: same per-item categorical, different profile summary ---

def reduce_nominal(items: dict[str, dict], instr: Instrument) -> np.ndarray:
    """Nominal profile = mean category probability per dimension.

    For MFV, the answer is the foundation. The reducer maps answer tokens into profile dimensions
    and averages the canonical per-item categorical distributions.
    """
    assert instr.kind == "nominal", "reduce_nominal expects a nominal instrument"
    assert instr.answer_to_dim is not None, f"{instr.name}: nominal instrument needs answer_to_dim"
    by_dim: dict[str, list[float]] = {d: [] for d in instr.dimensions}
    for it in items.values():
        for answer, p in zip(instr.answer_space, it["p"]):
            by_dim[instr.answer_to_dim[answer]].append(float(p))
    return np.array([float(np.mean(by_dim[d])) for d in instr.dimensions])

def reduce_ordinal(items: dict[str, dict], instr: Instrument) -> np.ndarray:
    """Likert profile = mean keyed agreement per dimension; agreement = E[scale point].

    Keying lives here and ONLY here: a reverse-keyed item (sign<0) means agreeing with it scores
    LOW on the factor, so reflect agreement = M+1 - E. The frame canonicalization already happened
    in per_item_categorical, so this is the single, orthogonal keying step (no double-flip).
    """
    w = np.arange(1, instr.scale_max + 1, dtype=float)
    by_dim: dict[str, list[float]] = defaultdict(list)
    for it in items.values():
        assert it["dimension"] is not None, "ordinal item has dimension=None; it would pool into a phantom factor"
        E = float((it["p"] * w).sum())
        agr = (instr.scale_max + 1 - E) if it["sign"] < 0 else E
        by_dim[it["dimension"]].append(agr)
    return np.array([float(np.mean(by_dim[d])) for d in instr.dimensions])
