"""Instrument registry: build `Instrument` objects (instrument.py schema) from the bundled survey
data. MFQ-2 plus three reused questionnaires (Big-Five, 16PF, Humor Styles), all ordinal 1-5.

Ported from the weight_steer_honesty experiment (`mft_honesty.instruments`). Each instrument is
three framing files (forward / inverted / negated) of the SAME items reworded; we load all three
into InstrItems tagged with `frame`, so the reader scores every (item, frame) and
`per_item_categorical` canonicalizes them to one forward distribution per item.

Per-instrument variation:
- data dir + framing filenames (MFQ-2: forward/inverted/negated.json; the others:
  questionnaire{,_inverted,_negated}.json).
- dimensions: derived from the forward file in first-appearance order.
- keying: MFQ-2 items are all +1; IPIP/HSQ are mixed-key within a factor, sign read from
  keying.json (written by the experiment's human-IPIP pipeline). Reverse items are reflected at
  reduce_ordinal time, not here.
- human reference scale: all administered at 1-5; the HSQ human reference is on 1-7
  (`human_scale_max`), reconciled to a 0-1 fraction at MAP time, not here. Per-item human_label is
  left None (the maps use the per-country CSVs in `data/human/`, not per-item histograms).
"""
from __future__ import annotations
import json
from pathlib import Path

from .instrument import Instrument, InstrItem

DATA = Path(__file__).resolve().parent / "data"
SURVEYS = DATA / "surveys"
HUMAN = DATA / "human"

DIGITS_1_5 = ["1", "2", "3", "4", "5"]
# Same response-format prefill the experiment used: the model answers "(N)", we prefill "(".
PREFILL = "("

# instrument name -> (survey subdir, {frame: filename stem}, human_csv, display, human_scale_max)
_F3 = {"forward": "questionnaire", "inverted": "questionnaire_inverted", "negated": "questionnaire_negated"}
_SPECS = {
    "mfq2": ("mfq2", {"forward": "forward", "inverted": "inverted", "negated": "negated"},
             "mfq2_country_foundations.csv", "Moral Foundations", 5),
    "big5": ("big5", _F3, "big5_country_factors.csv", "Big Five", 5),
    "16pf": ("16pf", _F3, "16pf_country_factors.csv", "16PF", 5),
    "humor_styles": ("humor_styles", _F3, "humor_styles_country_factors.csv", "Humor Styles", 7),
}


def _load_keying(survey_dir: Path, fallback_ids: list[str]) -> dict[str, int]:
    p = survey_dir / "keying.json"
    if not p.exists():
        return {i: 1 for i in fallback_ids}          # MFQ-2: no file -> every item +1
    return {str(k): int(v) for k, v in json.loads(p.read_text()).items()}


def build_instrument(name: str) -> Instrument:
    sub, frame_files, human_csv, display, hmax = _SPECS[name]
    sdir = SURVEYS / sub
    frames = {fr: json.loads((sdir / f"{stem}.json").read_text())[0] for fr, stem in frame_files.items()}

    fwd = frames["forward"]
    dims: list[str] = []                             # subscales in first-appearance order (forward file)
    for q in fwd["questions"]:
        if q["dimension"] not in dims:
            dims.append(q["dimension"])
    keying = _load_keying(sdir, [str(q["id"]) for q in fwd["questions"]])

    items: list[InstrItem] = []
    for fr, doc in frames.items():
        for q in doc["questions"]:
            items.append(InstrItem(
                id=str(q["id"]), prompt=q["content"], dimension=q["dimension"],
                sign=keying[str(q["id"])], frame=fr, human_label=None,
                meta={"task": doc["task"]}))          # frame-specific response-scale legend
    return Instrument(
        name=name, construct="endorsement", kind="ordinal",
        answer_space=DIGITS_1_5, dimensions=dims, items=items, prefill=PREFILL,
        scale_max=5, human_scale_max=hmax,
        display=display, human_csv=str(HUMAN / human_csv))


INSTRUMENTS: dict[str, Instrument] = {}


def get(name: str) -> Instrument:
    if name not in INSTRUMENTS:
        INSTRUMENTS[name] = build_instrument(name)
    return INSTRUMENTS[name]
