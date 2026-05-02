"""Backfill rater-distribution columns from clifford into scifi/airisk.

The Clifford et al. 2015 vignettes carry per-foundation rater % columns
(Care, Fairness, Loyalty, Authority, Sanctity, Liberty, Not Wrong) from the
original survey. The scifi/airisk variants are 1:1 ports (132 rows, same
order, same coarse foundation per row), so we copy the rater distribution
across by row index. Each variant keeps its own `Wrong` (different scale).

Updates two derived layers:

1. The CSVs (`data/vignettes_<name>.csv`) gain the 7 new columns.
2. The condition jsonls (`data/vignettes_<name>_{other,self}_violate.jsonl`)
   gain matching keys, joined by id (md5(scenario)). This mirrors what
   `02_rewrite.py:make_rec` would emit on a re-run, without needing to call
   the LLM again -- cache already covers all 132 scenarios.

Fails loudly if row counts disagree or if any row's coarse foundation
diverges across the three files.
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "vignettes.csv"
NAMES = ["scifi", "airisk"]
CONDITIONS = ["other_violate", "self_violate"]
COPY_COLS = ["Care", "Fairness", "Loyalty", "Authority", "Sanctity", "Liberty", "Not Wrong"]


def coarse(f: str) -> str:
    return re.split(r"\s*\(", f, maxsplit=1)[0].strip()


def hkey(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def patch_csv(src: pd.DataFrame, tgt_path: Path) -> pd.DataFrame:
    tgt = pd.read_csv(tgt_path)
    assert len(tgt) == len(src), f"{tgt_path.name}: {len(tgt)} rows, clifford has {len(src)}"
    src_coarse = src["Foundation"].map(coarse)
    tgt_coarse = tgt["Foundation"].map(coarse)
    bad = [(i, src_coarse[i], tgt_coarse[i]) for i in range(len(src)) if src_coarse[i] != tgt_coarse[i]]
    if bad:
        for b in bad[:10]:
            logger.error(f"{tgt_path.name} row {b[0]}: clifford={b[1]!r} target={b[2]!r}")
        raise ValueError(f"{tgt_path.name}: {len(bad)} foundation_coarse mismatches")

    for col in COPY_COLS:
        tgt[col] = src[col].values
    tgt = tgt[["Scenario", "Foundation", *COPY_COLS, "Wrong"]]
    tgt.to_csv(tgt_path, index=False)
    logger.info(f"wrote {tgt_path} cols={list(tgt.columns)}")
    return tgt


def patch_jsonl(tgt: pd.DataFrame, jsonl_path: Path) -> None:
    """Augment each record with the new % columns, joined by id = hkey(Scenario)."""
    by_id = {hkey(r["Scenario"]): {c: r[c] for c in COPY_COLS} for _, r in tgt.iterrows()}
    lines = jsonl_path.read_text().splitlines()
    out = []
    n_match = 0
    for line in lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        extra = by_id.get(rec["id"])
        if extra is None:
            raise ValueError(f"{jsonl_path.name}: id {rec['id']} not in CSV")
        n_match += 1
        # preserve key order: original keys first, then new
        for k, v in extra.items():
            rec[k] = v
        out.append(json.dumps(rec))
    jsonl_path.write_text("\n".join(out) + "\n")
    logger.info(f"patched {jsonl_path.name}: {n_match} records")


def main() -> None:
    src = pd.read_csv(SRC)
    assert all(c in src.columns for c in COPY_COLS), f"clifford missing cols: {set(COPY_COLS) - set(src.columns)}"

    for name in NAMES:
        tgt_csv = ROOT / "data" / f"vignettes_{name}.csv"
        tgt_df = patch_csv(src, tgt_csv)
        for cond in CONDITIONS:
            jp = ROOT / "data" / f"vignettes_{name}_{cond}.jsonl"
            if not jp.exists():
                logger.warning(f"missing {jp}, skipping")
                continue
            patch_jsonl(tgt_df, jp)


if __name__ == "__main__":
    main()
