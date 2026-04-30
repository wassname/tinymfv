"""Dataset loading. Reads per-condition jsonls and inner-joins by id, returning
the packed structure the eval consumes.

Files used by eval (both paraphrased rewrites, both OOD relative to verbatim source):
    data/vignettes[_<name>]_other_violate.jsonl   (3rd-person paraphrase of origin)
    data/vignettes[_<name>]_self_violate.jsonl    (1st-person rewrite)

Side artifact (not used by eval, kept for human-correlation sanity check):
    data/vignettes[_<name>]_origin.jsonl          (verbatim source, train-set risk)

Each row: {id, foundation, foundation_coarse, wrong, text}.
Falls back to HuggingFace `wassname/tiny-mfv` if local files absent.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HF_REPO = "wassname/tiny-mfv"
CONDITIONS = ["other_violate", "self_violate"]


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


def load_vignettes(name: str = "") -> list[dict]:
    """Inner-join the 2 violate conditions by id. Returns rows with `other_violate`,
    `self_violate` keys plus id/foundation/foundation_coarse/wrong."""
    by_cond = {c: {r["id"]: r for r in load_condition(name, c)} for c in CONDITIONS}
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
        })
    return rows
