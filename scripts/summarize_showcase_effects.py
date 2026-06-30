"""Summarize a steering-lite all-instrument showcase for the README table."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import plot_steer_showcase as P
from tinymfv import get_instrument

DISPLAY = {
    "mfv": "MFV vignettes",
    "humor_styles": "Humor Styles",
    "big5": "Big Five",
    "mfq2": "MFQ-2 survey",
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _fmt(x: float, digits: int = 2) -> str:
    return f"{x:+.{digits}f}"


def _fmt_pct(x: float) -> str:
    return f"{x:+.0f}%"


def _ci_sem(lo: float, hi: float) -> float:
    return (hi - lo) / (2 * 1.96)


def _cs_label(cs: list[float]) -> str:
    return ", ".join(f"{c:+g}" if c else "0" for c in sorted(cs))


def _coherent_cs(run_dir: Path, instruments: list[str], coherence_frac: float,
                 contrast_frac: float, margin_frac: float) -> list[float]:
    ordinal = [name for name in instruments if name != "mfv"]
    quality = P.shared_quality_score(run_dir, ordinal, pmass_frac=coherence_frac,
                                     contrast_frac=contrast_frac, margin_frac=margin_frac)
    return P.coherent_prefix_cs(sorted(quality), quality, 1.0)


def _survey_rows(run_dir: Path, name: str, cs: list[float]) -> list[dict[str, str]]:
    instr = get_instrument(name)
    rows = _rows(run_dir / f"{name}_profiles.csv")
    by_key = {(r["foundation"], float(r["c"])): r for r in rows}
    humans = P.human_strip(instr)
    pos_c = max(c for c in cs if c > 0)
    neg_c = min(c for c in cs if c < 0)

    out = []
    for dim in instr.dimensions:
        neg = by_key[(dim, neg_c)]
        pos = by_key[(dim, pos_c)]
        human_vals = np.array([v for _country, v in humans[dim]], dtype=float)
        human_sd = float(human_vals.std(ddof=1))
        profile_delta = float(pos["mean"]) - float(neg["mean"])
        logit_delta = float(pos["C"]) - float(neg["C"])
        sem = float(np.hypot(
            _ci_sem(float(pos["C_ci95_lo"]), float(pos["C_ci95_hi"])),
            _ci_sem(float(neg["C_ci95_lo"]), float(neg["C_ci95_hi"])),
        ))
        out.append({
            "dataset": DISPLAY[name],
            "axis": dim,
            "c path": _cs_label(cs),
            "profile shift / human SD": _fmt_pct(100 * profile_delta / human_sd),
            "profile shift": _fmt(profile_delta),
            "reader-logit shift": f"{_fmt(logit_delta)} ± {sem:.2f}",
        })
    return out


def _mfv_rows(run_dir: Path, cs: list[float]) -> list[dict[str, str]]:
    founds, countries, human_M, prof, _pmass = P._mfv_zspace(run_dir)
    rows = _rows(run_dir / "mfv_profiles.csv")
    by_key = {(r["foundation"], float(r["c"])): r for r in rows}
    pos_c = max(c for c in cs if c > 0)
    neg_c = min(c for c in cs if c < 0)

    out = []
    for j, foundation in enumerate(founds):
        neg = by_key[(foundation, neg_c)]
        pos = by_key[(foundation, pos_c)]
        human_sd = float(human_M[:, j].std(ddof=1))
        profile_delta = float(prof[pos_c][j] - prof[neg_c][j])
        logit_delta = float(pos["dlogit"]) - float(neg["dlogit"])
        sem = float(np.hypot(float(pos["dlogit_sem"]), float(neg["dlogit_sem"])))
        out.append({
            "dataset": DISPLAY["mfv"],
            "axis": foundation,
            "c path": _cs_label(cs),
            "profile shift / human SD": _fmt_pct(100 * profile_delta / human_sd),
            "profile shift": _fmt(profile_delta),
            "reader-logit shift": f"{_fmt(logit_delta)} ± {sem:.2f}",
        })
    return out


def _markdown_table(rows: list[dict[str, str]]) -> str:
    cols = ["dataset", "axis", "c path", "profile shift / human SD", "profile shift", "reader-logit shift"]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row[c] for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--coherence-frac", type=float, default=0.99)
    ap.add_argument("--contrast-frac", type=float, default=0.50)
    ap.add_argument("--margin-frac", type=float, default=0.50)
    ap.add_argument("--instruments", nargs="+", default=["mfv", "humor_styles", "big5", "mfq2"])
    args = ap.parse_args()

    cs = _coherent_cs(args.run_dir, args.instruments, args.coherence_frac,
                     args.contrast_frac, args.margin_frac)
    assert any(c > 0 for c in cs) and any(c < 0 for c in cs), f"need both signed arms, got {cs}"

    rows: list[dict[str, str]] = []
    if "mfv" in args.instruments:
        rows.extend(_mfv_rows(args.run_dir, cs))
    for name in args.instruments:
        if name != "mfv":
            rows.extend(_survey_rows(args.run_dir, name, cs))
    print(_markdown_table(rows))


if __name__ == "__main__":
    main()
