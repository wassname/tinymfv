"""Dataset loading. Reads per-condition jsonls and inner-joins by id, returning
the packed structure the eval consumes.

Files used by eval (both paraphrased rewrites, both OOD relative to verbatim source):
    data/vignettes[_<name>]_other_violate.jsonl   (3rd-person paraphrase of origin)
    data/vignettes[_<name>]_self_violate.jsonl    (1st-person rewrite)

Side artifact (not used by eval, kept for human-correlation sanity check):
    data/vignettes[_<name>]_origin.jsonl          (verbatim source, train-set risk)

Each row: {id, foundation, foundation_coarse, wrong, text}.

Dual-axis design
================
Each vignette produces 4 prompts from two independent binary axes:

    **cond** (scenario framing — which text variant the model reads):
        `other_violate`  — 3rd-person ("You see someone doing X")
        `self_violate`   — 1st-person ("You do X")

    **frame** (question framing — how the JSON probe is phrased):
        `wrong`   — '{"is_wrong": '    → true means wrong
        `accept`  — '{"is_acceptable": ' → true means right (inverted)

Both axes are paired-out in `analyse()`:
    - The two *frames* cancel the additive JSON-true prior (training data has
      more `"true"` than `"false"` in JSON contexts).
    - The two *conds* let you measure perspective bias: the gap between how
      harshly the model judges others vs itself for the same scenario.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[2]
HF_REPO = "wassname/tiny-mfv"
CONDITIONS = ["other_violate", "self_violate"]

# Canonical config names.
CONFIGS: tuple[str, ...] = ("classic", "scifi", "ai-actor")

ConfigName = Literal["classic", "scifi", "ai-actor", "all"]


def _local_path(name: str, condition: str) -> Path:
    return ROOT / "data" / f"vignettes_{name}_{condition}.jsonl"


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


# Legacy column names from the source jsonls -> normalised `human_*` keys.
# "Not Wrong" is the Clifford et al. (2015) social-norms control option
# ("the act is morally fine"), which maps to our SocialNorms foundation.
_HUMAN_LEGACY: dict[str, str] = {
    "Care": "human_Care",
    "Fairness": "human_Fairness",
    "Loyalty": "human_Loyalty",
    "Authority": "human_Authority",
    "Sanctity": "human_Sanctity",
    "Liberty": "human_Liberty",
    "Not Wrong": "human_SocialNorms",
}


def _parse_pct(v) -> float | None:
    """Parse '83 %' / '83%' / 83.0 -> 83.0; return None on missing/blank."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().rstrip("%").strip()
    if not s:
        return None
    return float(s)


def load_condition(name: str, condition: str) -> list[dict]:
    """Load one condition file."""
    p = _local_path(name, condition)
    if not p.exists():
        raise FileNotFoundError(f"Missing required data file: {p}")
    return _load_jsonl(p)


def load_vignettes(name: ConfigName = "classic") -> list[dict]:
    """Load vignettes by config name.

    Args:
        name: ``'classic'`` (Clifford et al. 2015), ``'scifi'``, ``'ai-actor'``
              (the same source items transcribed onto AI-as-actor scenarios),
              or ``'all'`` to concat with a ``set`` column.

    Returns:
        List of dicts with keys: ``id``, ``foundation``, ``foundation_coarse``,
        ``wrong``, ``other_violate``, ``self_violate``, ``set``.

    The two condition columns (*cond* axis) contain the scenario text:
        - ``other_violate``: 3rd-person framing ("You see someone doing X")
        - ``self_violate``:  1st-person framing ("You do X")

    Eval reads these directly as scenarios for the forced-choice foundation
    probe. The `human_*` label distribution is inherited from the source item.
    """
    if name.lower() == "all":
        return load_all_vignettes()

    cfg = name.lower()
    if cfg not in CONFIGS:
        raise ValueError(f"Unknown config {cfg!r}; expected one of {CONFIGS} or 'all'")

    by_cond = {c: {r["id"]: r for r in load_condition(cfg, c)} for c in CONDITIONS}
    common = set.intersection(*[set(d) for d in by_cond.values()])
    rows = []
    anchor = by_cond["other_violate"]
    _CORE_KEYS = {"id", "foundation", "foundation_coarse", "wrong", "text"}
    for vid, ov in anchor.items():
        if vid not in common:
            continue
        row = {
            "id": vid,
            "foundation": ov["foundation"],
            "foundation_coarse": ov["foundation_coarse"],
            "wrong": ov.get("wrong"),
            "other_violate": ov["text"],
            "self_violate": by_cond["self_violate"][vid]["text"],
            "set": cfg,
        }
        # Pass through extra keys (ai_*, human_*, etc.); also normalise the
        # legacy stringified percent columns ('Care': '83 %') into numeric
        # `human_*` keys so eval can read a single label schema.
        for k, v in ov.items():
            if k in _CORE_KEYS or k in row:
                continue
            if k in _HUMAN_LEGACY:
                row[_HUMAN_LEGACY[k]] = _parse_pct(v)
            else:
                row[k] = v
        rows.append(row)
    return rows


def load_all_vignettes() -> list[dict]:
    """Load and concatenate all three configs with a ``set`` column."""
    all_rows = []
    for cfg in CONFIGS:
        all_rows.extend(load_vignettes(cfg))
    return all_rows
