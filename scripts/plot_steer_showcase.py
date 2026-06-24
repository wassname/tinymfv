"""Showcase tinymfv's plotting on a real steering run (the dogfood before publishing the lib).

Consumes a steering-lite `run_allinstr_showcase.py` output dir (one calibrated
activation-steering vector administered across every instrument, 3-point
base/+C/-C) and renders, per instrument, the tinymfv figures:

  - ordinal (mfq2/big5/16pf/humor_styles): ipsative culture map + range + zoom,
    via tinymfv.maps, against the bundled human cross-cultural cloud.
  - nominal MFV: a per-foundation Delta-logit dumbbell (pos vs neg pole).

The steer is a 3-point sweep, so cs = [-1, 0, +1] are SYMBOLIC pole indices
(the real calibrated coefficient C is in the title/caption, not the y-units).

  uv run python scripts/plot_steer_showcase.py \
    --run-dir ../steering-lite/outputs/allinstr_qwen35_4b --out docs/img/showcase
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import tinymfv as T
from tinymfv import get_instrument

ORDINAL = ["mfq2", "big5", "16pf", "humor_styles"]


def _frac(x, scale_max: int) -> np.ndarray:
    return (np.asarray(x, float) - 1) / (scale_max - 1)


def read_human_csv(path: str) -> dict[tuple[str, str], float]:
    """{(country, foundation): mean} from a tinymfv human_<instrument>.csv."""
    out: dict[tuple[str, str], float] = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[(r["country"], r["foundation"])] = float(r["mean"])
    return out


def human_matrix(instr) -> tuple[list[str], np.ndarray]:
    """(countries, M[countries x factors] as 0-1 fraction). Mirrors mft_honesty.maps.human_matrix."""
    dims = instr.dimensions
    h = read_human_csv(instr.human_csv)
    countries = sorted({c for (c, _f) in h})
    raw = np.array([[h[(c, f)] for f in dims] for c in countries])
    return countries, _frac(raw, instr.human_scale_max)


def human_strip(instr) -> dict[str, list[tuple[str, float]]]:
    """{factor: [(country, mean_on_model_scale)]}. Human 1-H rescaled to model 1-M for the range."""
    h = read_human_csv(instr.human_csv)
    H, M = instr.human_scale_max, instr.scale_max
    def rescale(v: float) -> float:
        return 1.0 + (v - 1.0) / (H - 1) * (M - 1) if H != M else v
    strip: dict[str, list[tuple[str, float]]] = {}
    for f in instr.dimensions:
        strip[f] = sorted(((c, rescale(v)) for (c, ff), v in h.items() if ff == f),
                          key=lambda t: t[1])
    return strip


def read_profiles(run_dir: Path, name: str, dims: list[str]) -> dict[str, np.ndarray]:
    """{pole: profile-vector in instrument factor order} from <name>_profiles.csv (model-scale means)."""
    by_pole: dict[str, dict[str, float]] = {}
    with open(run_dir / f"{name}_profiles.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            by_pole.setdefault(r["pole"], {})[r["foundation"]] = float(r["mean"])
    return {pole: np.array([d[f] for f in dims]) for pole, d in by_pole.items()}


def plot_ordinal(run_dir: Path, out: Path, name: str, vec_label: str, C: float) -> list[Path]:
    instr = get_instrument(name)
    dims = instr.dimensions
    prof_pole = read_profiles(run_dir, name, dims)
    base, pos, neg = prof_pole["base"], prof_pole["pos"], prof_pole["neg"]
    humans = human_strip(instr)
    cs = [-1.0, 0.0, 1.0]
    prof = {-1.0: neg, 0.0: base, 1.0: pos}

    countries, Mfrac = human_matrix(instr)
    labels = (f"base (c=0)", f"+C={C:+.2f}", f"-C={-C:+.2f}")
    # mfq2 has per-respondent Atari data -> scatter the individual cloud behind the societies and
    # fit the ipsative PCA on PEOPLE (better-conditioned, the real envelope). Other instruments: None.
    respondents = T.maps.respondent_profiles(dims, instr.scale_max) if name == "mfq2" else None
    figm = T.maps.plot_ipsative_pca(instr, dims, countries, Mfrac,
                                    _frac(base, instr.scale_max), _frac(pos, instr.scale_max),
                                    _frac(neg, instr.scale_max), respondents=respondents, labels=labels)
    paths = [T.maps.save_both(figm, out / name, "map_pca_ipsative")]
    plt.close(figm)

    figr = T.maps.plot_range(instr, dims, cs, prof, humans, None, vec_label)
    paths.append(T.maps.save_both(figr, out / name, "range"))
    plt.close(figr)

    figz = T.maps.plot_range_zoom(instr, dims, cs, prof, humans, vec_label)
    paths.append(T.maps.save_both(figz, out / name, "range_zoom"))
    plt.close(figz)
    return paths


def plot_mfv(run_dir: Path, out: Path, vec_label: str, C: float) -> Path:
    """Per-foundation Delta-logit dumbbell: each foundation's +C (red) and -C (blue) shift vs bare."""
    d = json.loads((run_dir / "mfv.json").read_text())
    order = d["foundation_order"]
    pos = d["pos"]["dlogit_per_foundation"]
    neg = d["neg"]["dlogit_per_foundation"]
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axvline(0, color="0.6", lw=0.8, zorder=1)
    for f, yi in zip(order, y):
        pm, ps = pos[f]["mean"], pos[f]["std"] / max(1, pos[f]["n"]) ** 0.5
        nm, ns = neg[f]["mean"], neg[f]["std"] / max(1, neg[f]["n"]) ** 0.5
        ax.plot([nm, pm], [yi, yi], color="0.8", lw=1.0, zorder=2)
        ax.errorbar(pm, yi, xerr=1.96 * ps, fmt="o", color="#c0392b", ms=5, capsize=2, zorder=3)
        ax.errorbar(nm, yi, xerr=1.96 * ns, fmt="o", color="#2c6fbb", ms=5, capsize=2, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(order)
    ax.set_xlabel("Delta logit(violation) vs bare  (nats)")
    ax.set_title(f"Steered MFV vignettes: {vec_label}", fontsize=11)
    ax.scatter([], [], color="#c0392b", label=f"+C={C:+.2f}")
    ax.scatter([], [], color="#2c6fbb", label=f"-C={-C:+.2f}")
    ax.legend(fontsize=8, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = out / "mfv"
    path = T.maps.save_both(fig, p, "foundation_dlogit")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/img/showcase"))
    args = ap.parse_args()
    summary = json.loads((args.run_dir / "summary.json").read_text())
    C = float(summary["calibrated_C"])
    method = summary["method"]
    vec_label = f"{method} (Authority/Care axis)"
    args.out.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for name in ORDINAL:
        if (args.run_dir / f"{name}_profiles.csv").exists():
            written += [str(p) for p in plot_ordinal(args.run_dir, args.out, name, vec_label, C)]
    if (args.run_dir / "mfv.json").exists():
        written.append(str(plot_mfv(args.run_dir, args.out, vec_label, C)))
    print(f"wrote {len(written)} figures under {args.out}:")
    for w in written:
        print(" ", w)


if __name__ == "__main__":
    main()
