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


def human_haze(instr, n_per_country: int = 200, seed: int = 0) -> np.ndarray:
    """Synthetic individual-respondent cloud (n x K, 0-1 fraction) for instruments that ship only
    society-level stats (big5/16pf/humor: no raw per-person data like mfq2's Atari file). For each
    (country, factor) we resample n Normal(mean, sd) draws from the published country mean+sd, so the
    cloud carries BOTH between-country (different means) and within-country (sd) human spread. Caveat:
    factors are drawn independently, so this marginal resample loses the cross-factor correlation a
    real respondent matrix has -- it is a backdrop envelope, not a covariance estimate, and is NOT
    used as the PCA basis (that stays the society means M)."""
    dims = instr.dimensions
    rng = np.random.default_rng(seed)
    stats: dict[tuple[str, str], tuple[float, float]] = {}
    with open(instr.human_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            stats[(r["country"], r["foundation"])] = (float(r["mean"]), float(r["sd"]))
    countries = sorted({c for (c, _f) in stats})
    blocks = []
    for c in countries:
        cols = [rng.normal(stats[(c, f)][0], stats[(c, f)][1], n_per_country) for f in dims]
        blocks.append(np.clip(np.stack(cols, axis=1), 1.0, instr.human_scale_max))
    return _frac(np.concatenate(blocks, axis=0), instr.human_scale_max)


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


def read_profiles(run_dir: Path, name: str, dims: list[str]) -> tuple[dict[float, np.ndarray], dict[float, float]]:
    """({c: profile-vector in factor order}, {c: pmass}) from <name>_profiles.csv. `c` is the signed
    multiplier of calibrated C (0 = base); a single-multiplier run yields just {-1, 0, +1}."""
    by_c: dict[float, dict[str, float]] = {}
    pmass: dict[float, float] = {}
    with open(run_dir / f"{name}_profiles.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            c = float(r["c"])
            by_c.setdefault(c, {})[r["foundation"]] = float(r["mean"])
            pmass[c] = float(r["pmass"])
    return {c: np.array([d[f] for f in dims]) for c, d in by_c.items()}, pmass


def plot_ordinal(run_dir: Path, out: Path, name: str, vec_label: str, C: float) -> list[Path]:
    instr = get_instrument(name)
    dims = instr.dimensions
    prof_c, pmass = read_profiles(run_dir, name, dims)
    cs = sorted(prof_c)
    base = prof_c[0.0]
    # headline arrows = the calibrated coefficient (c=+-1); the trajectory dots at |c|>1 extend
    # BEYOND the arrowheads, so a multi-C run shows deployment point + where stronger steer drifts.
    pos = prof_c[1.0] if 1.0 in prof_c else prof_c[max(cs)]
    neg = prof_c[-1.0] if -1.0 in prof_c else prof_c[min(cs)]
    humans = human_strip(instr)
    prof = prof_c

    countries, Mfrac = human_matrix(instr)
    labels = (f"base (c=0)", f"+C={C:+.2f}", f"-C={-C:+.2f}")
    # mfq2 has per-respondent Atari data -> scatter the REAL individual cloud behind the societies AND
    # fit the ipsative PCA on it (better-conditioned, the true envelope). Other instruments have no raw
    # per-person data, so scatter a marginal resample from each country's published mean+sd as the haze
    # while keeping the PCA basis on the society means M.
    if name == "mfq2":
        respondents, haze = T.maps.respondent_profiles(dims, instr.scale_max), None
    else:
        respondents, haze = None, human_haze(instr)
    # trajectory overlay only when the run swept more than the 3-point base/+-C (else the arrows suffice)
    traj = {c: _frac(prof_c[c], instr.scale_max) for c in cs} if len(cs) > 3 else None
    traj_inco = {c for c, pm in pmass.items() if pm < 0.9} if traj else None
    figm = T.maps.plot_ipsative_pca(instr, dims, countries, Mfrac,
                                    _frac(base, instr.scale_max), _frac(pos, instr.scale_max),
                                    _frac(neg, instr.scale_max), respondents=respondents, haze=haze,
                                    traj=traj, traj_incoherent=traj_inco, labels=labels)
    paths = [T.maps.save_both(figm, out / name, "map_pca_ipsative")]
    plt.close(figm)

    # SPLOM only for mfq2: real per-respondent joint (others ship independent-marginal haze, whose
    # off-diagonals would fabricate the correlation structure). Full + AI-zoom (macro + micro).
    if name == "mfq2":
        proffrac = {c: _frac(prof_c[c], instr.scale_max) for c in cs}
        for zoom, tag in [(False, "splom"), (True, "splom_zoom")]:
            figs = T.maps.plot_splom(instr, dims, respondents, Mfrac, _frac(base, instr.scale_max),
                                     proffrac, zoom=zoom, vec_label=vec_label)
            paths.append(T.maps.save_both(figs, out / name, tag))
            plt.close(figs)

    figr = T.maps.plot_range(instr, dims, cs, prof, humans, None, vec_label)
    paths.append(T.maps.save_both(figr, out / name, "range"))
    plt.close(figr)

    figz = T.maps.plot_range_zoom(instr, dims, cs, prof, humans, vec_label)
    paths.append(T.maps.save_both(figz, out / name, "range_zoom"))
    plt.close(figz)
    return paths


def _zscore(v: np.ndarray) -> np.ndarray:
    """Relative emphasis: centre and scale a profile across foundations, so a logit profile (model)
    and a 1-5 wrongness profile (human cultures) are comparable by PATTERN regardless of units."""
    return (v - v.mean()) / (v.std() + 1e-9)


def read_human_mfv() -> tuple[list[str], dict[str, dict[str, float]]]:
    """(countries, {country: {foundation: mean_1to5}}) from the bundled MFV human norms.
    JimenezLeal2025 (LatAm) + Yamada2025 (MFV-J): 5 countries x 6 foundations (no Social Norms)."""
    path = T.maps.DATA / "human" / "mfv_country_factors.csv"
    by_country: dict[str, dict[str, float]] = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            by_country.setdefault(r["country"], {})[r["foundation"]] = float(r["mean"])
    return sorted(by_country), by_country


def plot_mfv_map(run_dir: Path, out: Path, vec_label: str, C: float) -> Path:
    """Bespoke MFV map: per-foundation RELATIVE EMPHASIS (z across foundations) of the model's base /
    +C / -C reads against the human MFV cultures. MFV is nominal (model emits logit(violation) per
    foundation, humans rate wrongness 1-5), so absolute scales differ; z-scoring each profile within
    itself compares the PATTERN -- which foundations a reader weights as more violation-worthy than
    their own average -- which is exactly what the steer is meant to move. Social Norms is dropped (no
    human norm). The steer shows as base->+C (red) and base->-C (blue) arrows per foundation."""
    d = json.loads((run_dir / "mfv.json").read_text())
    base_l = d["base_logit_per_foundation"]
    pos_dl, neg_dl = d["pos"]["dlogit_per_foundation"], d["neg"]["dlogit_per_foundation"]
    countries, human = read_human_mfv()
    hfounds = set(next(iter(human.values())))
    founds = [f for f in d["foundation_order"] if f.lower() in hfounds]   # 6 shared, model order
    fl = [f.lower() for f in founds]

    base = _zscore(np.array([base_l[f]["mean"] for f in founds]))
    posz = _zscore(np.array([base_l[f]["mean"] + pos_dl[f]["mean"] for f in founds]))
    negz = _zscore(np.array([base_l[f]["mean"] + neg_dl[f]["mean"] for f in founds]))
    Hz = {c: _zscore(np.array([human[c][f] for f in fl])) for c in countries}

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    POS, NEG, GREY = T.maps.POS_COL, T.maps.NEG_COL, T.maps.COUNTRY_GREY
    for i, f in enumerate(founds):
        hvals = np.array([Hz[c][i] for c in countries])
        ax.scatter(i - 0.18 + (rng.random(len(hvals)) - 0.5) * 0.12, hvals, s=26, color=GREY,
                   alpha=0.9, edgecolor="white", linewidth=0.3, zorder=3)
        ax.plot([i - 0.30, i - 0.06], [np.median(hvals)] * 2, color=T.maps.MEDIAN_GREY, lw=1.4, zorder=4)
        xs = i + 0.18
        ax.plot(xs, base[i], "o", ms=4, color="black", zorder=7)
        for pole, col in [(posz[i], POS), (negz[i], NEG)]:
            if abs(pole - base[i]) > 1e-9:
                ax.plot([xs, xs], [base[i], pole], color=col, lw=2.0, zorder=6, solid_capstyle="round")
                ax.plot(xs, pole, marker=("^" if pole >= base[i] else "v"), color=col, ms=7,
                        markeredgecolor="none", zorder=8)
    ax.scatter([], [], marker="o", color=GREY, label=f"human culture (n={len(countries)})")
    ax.scatter([], [], marker="o", color="black", label="model base")
    ax.scatter([], [], marker="^", color=POS, label=f"steer +C={C:+.2f}")
    ax.scatter([], [], marker="v", color=NEG, label=f"steer -C={-C:+.2f}")
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9, ncol=2)
    ax.set_xticks(range(len(founds)))
    ax.set_xticklabels(founds, rotation=20, ha="right", fontsize=8)
    ax.set_xlim(-0.6, len(founds) - 0.4)
    ax.set_ylabel("relative emphasis  (z across foundations)")
    ax.set_title(f"MFV foundation emphasis vs human cultures: {vec_label}", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = T.maps.save_both(fig, out / "mfv", "map_emphasis")
    plt.close(fig)
    return path


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
        written.append(str(plot_mfv_map(args.run_dir, args.out, vec_label, C)))
        written.append(str(plot_mfv(args.run_dir, args.out, vec_label, C)))
    print(f"wrote {len(written)} figures under {args.out}:")
    for w in written:
        print(" ", w)


if __name__ == "__main__":
    main()
