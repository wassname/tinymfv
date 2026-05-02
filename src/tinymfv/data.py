"""Dataset loading. Reads per-condition jsonls and inner-joins by id, returning
the packed structure the eval consumes.

Files used by eval (both paraphrased rewrites, both OOD relative to verbatim source):
    data/vignettes[_<name>]_other_violate.jsonl   (3rd-person paraphrase of origin)
    data/vignettes[_<name>]_self_violate.jsonl    (1st-person rewrite)

Side artifact (not used by eval, kept for human-correlation sanity check):
    data/vignettes[_<name>]_origin.jsonl          (verbatim source, train-set risk)

Each row: {id, foundation, foundation_coarse, wrong, text}.
Falls back to HuggingFace `wassname/tiny-mfv` if local files absent.

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

ROOT = Path(__file__).resolve().parents[2]
HF_REPO = "wassname/tiny-mfv"
CONDITIONS = ["other_violate", "self_violate"]

# Canonical config names and aliases.
CONFIGS = ("classic", "scifi", "airisk")
_ALIASES = {"classic": "clifford", "clifford": "clifford"}  # classic→clifford on disk


def _resolve_name(name: str) -> str:
    """Map user-facing name to the canonical file/HF key.

    Accepts 'classic' (preferred) or 'clifford' (legacy).
    Returns the file-system key ('clifford' → empty suffix, others as-is).
    """
    low = name.lower()
    if low in _ALIASES:
        return _ALIASES[low]
    return low  # scifi, airisk pass through


def _local_path(name: str, condition: str) -> Path:
    suf = f"_{name}" if name else ""
    return ROOT / "data" / f"vignettes{suf}_{condition}.jsonl"


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def load_condition(name: str, condition: str) -> list[dict]:
    """Load one condition file."""
    p = _local_path(name, condition)
    if p.exists():
        return _load_jsonl(p)
    from datasets import load_dataset
    cfg = name or "clifford"
    return list(load_dataset(HF_REPO, cfg, split=condition))


def load_vignettes(name: str | None = None) -> list[dict]:
    """Inner-join the 2 violate conditions by id.

    Returns rows with `other_violate`, `self_violate` keys plus
    id / foundation / foundation_coarse / wrong.

    Args:
        name: One of 'classic' (alias: 'clifford'), 'scifi', 'airisk', or 'all'.
              Must be specified — calling with no argument raises ValueError.

    The two condition columns (*cond* axis) contain the scenario text:
        - ``other_violate``: 3rd-person framing ("You see someone doing X")
        - ``self_violate``:  1st-person framing ("You do X")

    These are crossed with the *frame* axis (``wrong`` / ``accept``) at eval
    time in ``format_prompts`` → ``analyse`` to cancel the JSON-true prior
    and measure perspective bias.  See module docstring for details.
    """
    if name is None:
        raise ValueError(
            "load_vignettes() requires a config name. "
            f"Choose one of: {', '.join(repr(c) for c in CONFIGS)}, or 'all'."
        )

    if name.lower() == "all":
        return load_all_vignettes()

    resolved = _resolve_name(name)
    # clifford files have no suffix (legacy naming: vignettes_other_violate.jsonl)
    file_name = "" if resolved == "clifford" else resolved

    by_cond = {c: {r["id"]: r for r in load_condition(file_name, c)} for c in CONDITIONS}
    common = set.intersection(*[set(d) for d in by_cond.values()])
    rows = []
    anchor = by_cond["other_violate"]
    for vid, ov in anchor.items():
        if vid not in common:
            continue
        rows.append({
            "id": vid,
            "foundation": ov["foundation"],
            "foundation_coarse": ov["foundation_coarse"],
            "wrong": ov.get("wrong"),
            "other_violate": ov["text"],
            "self_violate": by_cond["self_violate"][vid]["text"],
            "set": name.lower() if name.lower() != "clifford" else "classic",
        })
    return rows


def load_all_vignettes() -> list[dict]:
    """Load and concatenate all three configs (classic, scifi, airisk).

    Returns the union with a ``set`` column indicating the source config.
    """
    all_rows = []
    for cfg in CONFIGS:
        all_rows.extend(load_vignettes(cfg))
    return all_rows
