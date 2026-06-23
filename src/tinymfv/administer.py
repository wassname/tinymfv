"""Run an ordinal Instrument end-to-end on a local model -> profile + coherence canary.

This is the survey counterpart to `tinymfv.evaluate` (the vignette forced-choice eval). It ties:

    read_items            (answer-token readout, all frames)
    per_item_categorical  (canonicalize each frame to forward, average -> one dist per item)
    reduce_ordinal        (E[scale point] per item, reverse-key, pool to a per-factor profile)

The profile vector (per `instr.dimensions`) is the load-bearing output; `per_item_categorical`'s
canonicalization makes it algebraically identical to the experiment's `admin.administer` per-factor
means (verified by the reducer parity check), so this is a drop-in for the maps.

`mean_pmass_allowed` is the coherence canary: mass on valid answer tokens. A sharp drop (especially
after steering) means the profile is untrustworthy even if every digit is in-format.
"""
from __future__ import annotations

import numpy as np

from .instrument import Instrument, per_item_categorical, reduce_ordinal, canonicalize_to_forward
from .read import read_items, resolve_answer_ids


def administer(model, tok, instr: Instrument, *, batch_size: int = 36) -> dict:
    assert instr.kind == "ordinal", "administer() is the ordinal survey readout; use evaluate() for nominal MFV"
    w = np.arange(1, instr.scale_max + 1, dtype=float)
    answer_ids = resolve_answer_ids(tok, instr.answer_space)
    per_row = read_items(model, tok, instr, instr.items, answer_ids,
                         batch_size=batch_size, verbose_first=True)
    items = per_item_categorical(per_row, instr.kind)            # {id: {p, pmass, dimension, sign, ...}}

    profile = reduce_ordinal(items, instr)                       # per-factor keyed agreement
    mean_pmass = float(np.mean([it["pmass"] for it in items.values()]))

    # per-item frame-averaged keyed agreement (for the maps' bootstrap uncertainty cross)
    per_item_rows = []
    for iid, it in items.items():
        E = float((it["p"] * w).sum())
        keyed = (instr.scale_max + 1 - E) if it["sign"] < 0 else E
        per_item_rows.append({"id": iid, "foundation": it["dimension"], "keyed_agreement": keyed,
                              "E": E, "pmass_allowed": it["pmass"], "frame_spread": it["frame_spread"]})

    # per-frame factor means + framing spread (acquiescence/wording diagnostic): for each frame,
    # canonicalize that frame's presented distribution to forward, key it, pool per factor.
    frames = sorted({r["frame"] for r in per_row})
    by_dim_frame: dict[tuple[str, str], list[float]] = {}
    for r in per_row:
        p_fwd = canonicalize_to_forward(r["p"], r["frame"], instr.kind)
        E = float((p_fwd * w).sum())
        keyed = (instr.scale_max + 1 - E) if r["sign"] < 0 else E
        by_dim_frame.setdefault((r["dimension"], r["frame"]), []).append(keyed)

    rng = np.random.default_rng(0)
    foundations = []
    for j, d in enumerate(instr.dimensions):
        vals = np.array([row["keyed_agreement"] for row in per_item_rows if row["foundation"] == d])
        boot = rng.choice(vals, size=(2000, len(vals)), replace=True).mean(axis=1)
        per_fr = {fr: float(np.mean(by_dim_frame[(d, fr)])) for fr in frames}
        foundations.append({
            "foundation": d, "mean": float(profile[j]), "sd": float(vals.std(ddof=1)),
            "ci95_lo": float(np.percentile(boot, 2.5)), "ci95_hi": float(np.percentile(boot, 97.5)),
            "framing_spread": float(max(per_fr.values()) - min(per_fr.values())),
            **{f"f_{fr}": v for fr, v in per_fr.items()},
        })
    return {"profile": profile, "dimensions": instr.dimensions, "foundations": foundations,
            "per_item": per_item_rows, "mean_pmass_allowed": mean_pmass}
