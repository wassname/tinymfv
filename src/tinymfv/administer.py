"""Run an ordinal Instrument end-to-end on a local model -> profile + coherence check.

This is the survey counterpart to `tinymfv.evaluate` (the vignette forced-choice eval). It ties:

    read_items            (answer-token readout, all frames)
    per_item_categorical  (canonicalize each frame to forward, average -> one dist per item)
    reduce_ordinal        (E[scale point] per item, reverse-key, pool to a per-factor profile)

The profile vector (per `instr.dimensions`) is the main output; `per_item_categorical`'s
canonicalization makes it algebraically identical to the experiment's `admin.administer` per-factor
means (verified by the reducer parity check), so this is a drop-in for the maps.

`mean_pmass_allowed` is the coherence check: mass on valid answer tokens. A sharp drop (especially
after steering) means the profile is untrustworthy even if every digit is in-format.
"""
from __future__ import annotations

from typing import TypedDict

import numpy as np

from .instrument import Instrument, per_item_categorical, reduce_ordinal, canonicalize_to_forward
from .read import read_items, resolve_answer_ids
from .readouts import expected_score, logit_contrast, logodds_agree, entropy


class ItemRow(TypedDict):
    id: str
    foundation: str
    E: float                        # expected scale point, 1..scale_max (human-comparable, insensitive)
    keyed_E: float                  # E reverse-keyed for sign<0 items (== old keyed_agreement)
    C: float                        # rank-centered logit contrast (primary steer signal)
    keyed_C: float                  # C reverse-keyed (negated) for sign<0 items
    logodds_agree: float            # agree-vs-disagree log-odds (readable direction summary)
    keyed_logodds_agree: float
    entropy: float                  # within-allowed entropy, nats (coherence the pmass gate misses)
    pmass_allowed: float
    frame_spread: float


class ItemFrameRow(TypedDict):
    id: str
    framing: str                    # forward | inverted | negated
    foundation: str
    lp: list[float]                 # raw logprobs at the M scale tokens (presented orientation)
    E: float                        # forward-canonicalized E toward the original statement
    C: float                        # forward-canonicalized logit contrast
    keyed_E: float
    keyed_C: float
    pmass_allowed: float
    think: str                      # the model's think trace for this (item, frame)
    n_think: int
    emitted_close: bool


class AdministerResult(TypedDict):
    # profiles: per-factor means in factor order (`dimensions`). E is the human-comparable score;
    # C (the rank-centered logit contrast) is the steering-legible one (E saturates at confidence).
    profile_E: np.ndarray           # [len(dimensions)] per-factor keyed E (the human-comparison map input)
    profile_C: np.ndarray           # [len(dimensions)] per-factor keyed contrast (the steer map input)
    profile: np.ndarray             # alias of profile_E (kept so existing E maps keep working)
    dimensions: list[str]           # factor order, matches the profiles
    foundations: list[dict]         # one per factor: foundation, mean(E), C, logodds_agree, sd, ci95*, f_<frame>
    per_item: list[ItemRow]         # one per item, frame-averaged readouts
    per_item_frame: list[ItemFrameRow]  # one per (item, frame): raw lp + think + readouts
    mean_pmass_allowed: float       # coherence check (mass on valid answer tokens)


def administer(model, tok, instr: Instrument, *, batch_size: int = 36,
               max_think_tokens: int = 64) -> AdministerResult:
    assert instr.kind == "ordinal", "administer() is the ordinal survey readout; use evaluate() for nominal MFV"
    # Every ordinal item must carry its frame-specific response-scale legend in meta['task']; without
    # it build_user_content would silently emit a bare statement (no legend) and the profile would be
    # junk while pmass still looks fine. Fail loud.
    assert all("task" in it.meta for it in instr.items), f"{instr.name}: ordinal items need meta['task']"
    answer_ids = resolve_answer_ids(tok, instr.answer_space)
    # max_think_tokens=64 is the spec's "light" default: the model thinks before the prefilled answer
    # slot, so an activation steer accrues over the trace before being read. Floor is 1 (the shared
    # rollout core's HF generate() rejects max_new_tokens=0).
    per_row = read_items(model, tok, instr, instr.items, answer_ids,
                         max_think_tokens=max_think_tokens, batch_size=batch_size, verbose_first=True)
    items = per_item_categorical(per_row, instr.kind)            # {id: {p, pmass, dimension, sign, ...}}

    M = instr.scale_max
    profile_E = reduce_ordinal(items, instr)                     # per-factor keyed E (human comparison)
    # nanmean: an unscorable item (blown-up forced read -> NaN pmass) drops out rather
    # than poisoning the mean, matching the forced-choice eval path.
    mean_pmass = float(np.nanmean([it["pmass"] for it in items.values()]))

    # per-item frame-averaged readouts. E and entropy come from the averaged probability vector (so the
    # NaN-at-collapse signal survives); C and log-odds come from the averaged logprobs (the sensitive
    # log-space readouts). Reverse-keying (sign<0): E reflects to M+1-E, while the midpoint-centered
    # contrast C and the agree-vs-disagree log-odds negate.
    per_item_rows = []
    for iid, it in items.items():
        lp, p, sign = it["lp"], it["p"], it["sign"]
        E, Cval, LO = expected_score(p, M), logit_contrast(lp, M), logodds_agree(lp, M)
        per_item_rows.append({
            "id": iid, "foundation": it["dimension"],
            "E": E, "keyed_E": (M + 1 - E) if sign < 0 else E,
            "C": Cval, "keyed_C": -Cval if sign < 0 else Cval,
            "logodds_agree": LO, "keyed_logodds_agree": -LO if sign < 0 else LO,
            "entropy": entropy(p, M), "pmass_allowed": it["pmass"], "frame_spread": it["frame_spread"],
        })

    # per-(item, frame) rows: the raw granularity (lp + think + per-frame readouts) downstream analyses
    # need -- framing-bias diagnostic, paired base-vs-steer deltas, the pro-vs-anti interaction control.
    # E/C are forward-canonicalized toward the original statement; keyed_* reflect reverse-keyed items.
    frames = sorted({r["frame"] for r in per_row})
    by_dim_frame: dict[tuple[str, str], list[float]] = {}        # keyed E per (factor, frame): framing diagnostic
    per_item_frame: list[dict] = []
    for r in per_row:
        sign = r["sign"]
        p_fwd = canonicalize_to_forward(r["p"], r["frame"], instr.kind)
        lp_fwd = canonicalize_to_forward(r["lp"], r["frame"], instr.kind)
        E, Cval = expected_score(p_fwd, M), logit_contrast(lp_fwd, M)
        by_dim_frame.setdefault((r["dimension"], r["frame"]), []).append((M + 1 - E) if sign < 0 else E)
        per_item_frame.append({
            "id": r["id"], "framing": r["frame"], "foundation": r["dimension"],
            "lp": list(map(float, r["lp"])),
            "E": E, "C": Cval,
            "keyed_E": (M + 1 - E) if sign < 0 else E,
            "keyed_C": -Cval if sign < 0 else Cval,
            "pmass_allowed": r["pmass_allowed"],
            "think": r["think"], "n_think": r["n_think"], "emitted_close": r["emitted_close"],
        })

    # per-factor means + bootstrap CIs for BOTH E and C (the uncertainty the user wants alongside the
    # mean), the mean log-odds, and the per-frame E means (framing-bias diagnostic).
    rng = np.random.default_rng(0)
    def _ci(vals: np.ndarray) -> tuple[float, float]:
        boot = rng.choice(vals, size=(2000, len(vals)), replace=True).mean(axis=1)
        return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    profile_C = np.zeros(len(instr.dimensions))
    foundations = []
    for j, d in enumerate(instr.dimensions):
        e_vals = np.array([row["keyed_E"] for row in per_item_rows if row["foundation"] == d])
        c_vals = np.array([row["keyed_C"] for row in per_item_rows if row["foundation"] == d])
        lo_vals = np.array([row["keyed_logodds_agree"] for row in per_item_rows if row["foundation"] == d])
        profile_C[j] = float(np.mean(c_vals))
        e_lo, e_hi = _ci(e_vals); c_lo, c_hi = _ci(c_vals)
        per_fr = {fr: float(np.mean(by_dim_frame[(d, fr)])) for fr in frames}
        foundations.append({
            "foundation": d,
            "mean": float(profile_E[j]), "sd": float(e_vals.std(ddof=1)),
            "ci95_lo": e_lo, "ci95_hi": e_hi,
            "C": float(profile_C[j]), "C_sd": float(c_vals.std(ddof=1)),
            "C_ci95_lo": c_lo, "C_ci95_hi": c_hi,
            "logodds_agree": float(np.mean(lo_vals)),
            "framing_spread": float(max(per_fr.values()) - min(per_fr.values())),
            **{f"f_{fr}": v for fr, v in per_fr.items()},
        })
    return {"profile_E": profile_E, "profile_C": profile_C, "profile": profile_E,
            "dimensions": instr.dimensions, "foundations": foundations,
            "per_item": per_item_rows, "per_item_frame": per_item_frame,
            "mean_pmass_allowed": mean_pmass}
