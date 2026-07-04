"""Showcase tinymfv's plotting on a real steering run.

Consumes a steering-lite `run_allinstr_showcase.py` output dir (one calibrated
activation-steering vector administered across every instrument over a signed
c-sweep) and renders the SAME two figures for every instrument, uniformly:

  - map  : ipsative culture map (PCA), AI coherent +/-c path vs the human cloud.
  - range: per-factor range, AI base + coherent +/-c path vs the human society strip.

Ordinal instruments read <name>_profiles.csv. MFV reads mfv_profiles.csv and is
projected into z-scored relative-emphasis space, because its nominal foundation
probabilities cannot share a raw axis with 1-5 survey scores. It still goes
through the same plot_ipsative_pca / plot_range functions.

cs are SIGNED multipliers of the calibrated coefficient C (0 = base). The public
README plots show the coherent path: c=0 plus each +/-c row whose tinymfv answer mass
stays above the requested fraction of base. Incoherent rows are dropped.

  uv run python scripts/plot_steer_showcase.py \
    --run-dir ../steering-lite/outputs/allinstr_qwen35_4b \
    --out docs/img/showcase \
    --vec-label "MFV Authority anchor (+c intended higher Authority)"
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import tinymfv as T
from tinymfv import get_instrument
from tinymfv.zones import zones_for

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


def human_haze(instr, n_per_country: int = 200, seed: int = 0) -> tuple[np.ndarray, list[str]]:
    """Synthetic individual-respondent cloud (n x K, 0-1 fraction) + the country of each row, for
    instruments that ship only society-level stats (big5/16pf/humor: no raw per-person data like
    mfq2's Atari file). For each (country, factor) we resample n Normal(mean, sd) draws from the
    published country mean+sd, so the cloud carries BOTH between-country (different means) and
    within-country (sd) human spread. Caveat: factors are drawn independently, so this marginal
    resample loses the cross-factor correlation a real respondent matrix has -- it is a backdrop
    envelope, not a covariance estimate, and is NOT used as the PCA basis (that stays the society
    means M). The returned country-per-row list lets the map contour it by IW zone."""
    dims = instr.dimensions
    rng = np.random.default_rng(seed)
    stats: dict[tuple[str, str], tuple[float, float]] = {}
    with open(instr.human_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            stats[(r["country"], r["foundation"])] = (float(r["mean"]), float(r["sd"]))
    countries = sorted({c for (c, _f) in stats})
    blocks, row_country = [], []
    for c in countries:
        cols = [rng.normal(stats[(c, f)][0], stats[(c, f)][1], n_per_country) for f in dims]
        blocks.append(np.clip(np.stack(cols, axis=1), 1.0, instr.human_scale_max))
        row_country.extend([c] * n_per_country)
    return _frac(np.concatenate(blocks, axis=0), instr.human_scale_max), row_country


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


def read_profiles(run_dir: Path, name: str, dims: list[str], value_col: str = "mean"
                  ) -> tuple[dict[float, np.ndarray], dict[float, float]]:
    """({c: profile-vector in factor order}, {c: pmass}) from <name>_profiles.csv. `c` is the signed
    multiplier of calibrated C (0 = base); a single-multiplier run yields just {-1, 0, +1}.
    value_col selects the readout: 'mean' = E (human-comparable, for the map/range vs human band);
    'C' = the rank-centered logit contrast (the steer-legible signal, for the steer-effect plot)."""
    by_c: dict[float, dict[str, float]] = {}
    pmass: dict[float, float] = {}
    with open(run_dir / f"{name}_profiles.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            c = float(r["c"])
            by_c.setdefault(c, {})[r["foundation"]] = float(r[value_col])
            pmass[c] = float(r["pmass"])
    return {c: np.array([d[f] for f in dims]) for c, d in by_c.items()}, pmass


def coherent_prefix_cs(cs: list[float], quality: dict[float, float], floor: float) -> list[float]:
    """c=0 plus each signed arm until the shared quality score first falls below `floor`."""
    kept = [0.0]
    for side in (1.0, -1.0):
        for c in sorted([c for c in cs if np.sign(c) == side], key=abs):
            if quality[c] < floor:
                break
            kept.append(c)
    return sorted(kept)


def shared_quality_score(run_dir: Path, names: list[str], *, pmass_frac: float,
                         contrast_frac: float, margin_frac: float) -> dict[float, float]:
    """Worst base-relative quality across instruments, keyed by signed calibrated multiplier.

    For ordinal surveys, answer mass can stay ~1 while the within-answer distribution becomes
    generic. The rank-logit contrast retention catches that failure mode. For MFV, answer mass is
    structurally pinned by the forced-choice scaffold, so mean top1-vs-top2 margin is the live OOD
    signal when present.
    """
    scores: list[dict[float, float]] = []
    pmasses: list[dict[float, float]] = []
    for name in names:
        instr = get_instrument(name)
        _, pmass = read_profiles(run_dir, name, instr.dimensions)
        c_prof, _ = read_profiles(run_dir, name, instr.dimensions, value_col="C")
        pmasses.append(pmass)
        base_contrast = float(np.mean(np.abs(c_prof[0.0])))
        scores.append({
            c: min(pmass[c] / pmass[0.0] / pmass_frac,
                   float(np.mean(np.abs(c_prof[c]))) / base_contrast / contrast_frac)
            for c in pmass
        })
    if (run_dir / "mfv_profiles.csv").exists():
        mfv_pmass: dict[float, float] = {}
        mfv_margin: dict[float, float] = {}
        with open(run_dir / "mfv_profiles.csv", newline="") as fh:
            for r in csv.DictReader(fh):
                c = float(r["c"])
                mfv_pmass[c] = float(r["pmass"])
                mfv_margin[c] = float(r["mean_margin"])
        scores.append({
            c: min(mfv_pmass[c] / mfv_pmass[0.0] / pmass_frac,
                   mfv_margin[c] / mfv_margin[0.0] / margin_frac)
            for c in mfv_pmass
        })
    assert scores, "no profile CSVs found"
    cs = sorted(set.intersection(*(set(s) for s in scores)))
    return {c: min(s[c] for s in scores) for c in cs}


def plot_ordinal(run_dir: Path, out: Path, name: str, vec_label: str, C: float,
                 coh_cs: list[float]) -> list[Path]:
    instr = copy.copy(get_instrument(name))
    if name == "mfq2":
        instr.display = "MFQ-2 survey"
    dims = instr.dimensions
    prof_c, _pmass = read_profiles(run_dir, name, dims)
    base = prof_c[0.0]
    pos_c = max(c for c in coh_cs if c > 0.0)
    neg_c = min(c for c in coh_cs if c < 0.0)
    pos = prof_c[pos_c]
    neg = prof_c[neg_c]
    humans = human_strip(instr)
    prof = prof_c

    countries, Mfrac = human_matrix(instr)
    labels = ("base (c=0)", f"c={pos_c:+g}", f"c={neg_c:+g}")
    # mfq2 has per-respondent Atari data -> scatter the REAL individual cloud behind the societies AND
    # fit the ipsative PCA on it (better-conditioned, the true envelope). Other instruments have no raw
    # per-person data, so scatter a marginal resample from each country's published mean+sd as the haze
    # while keeping the PCA basis on the society means M.
    # Each IW zone is a covariance ellipse over its member COUNTRY-MEAN dots (drawn in maps). mfq2
    # scatters its real Atari respondents behind; the others scatter a per-country resample.
    zones, emph = zones_for(countries)
    if name == "mfq2":
        _, respondents = T.maps.respondent_profiles(dims, instr.scale_max)
        haze = None
    else:
        respondents, (haze, _) = None, human_haze(instr)
    traj = {c: _frac(prof_c[c], instr.scale_max) for c in coh_cs}
    figm = T.maps.plot_ipsative_pca(instr, dims, countries, Mfrac,
                                    _frac(base, instr.scale_max), _frac(pos, instr.scale_max),
                                    _frac(neg, instr.scale_max), respondents=respondents, haze=haze,
                                    traj=traj, emphasize=emph, zones=zones, labels=labels)
    figm.axes[0].set_title(f"{instr.display}: humans vs LLMs steered for {vec_label}", fontsize=10)
    paths = [T.maps.save_both(figm, out / name, "map_pca_ipsative")]
    plt.close(figm)

    prof_plot = {c: prof_c[c] for c in coh_cs}
    figr = T.maps.plot_range(instr, dims, coh_cs, prof_plot, humans, None, vec_label)
    paths.append(T.maps.save_both(figr, out / name, "range"))
    plt.close(figr)
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

def read_mfv_profiles(run_dir: Path) -> tuple[list[str], dict[float, np.ndarray], dict[float, float]]:
    rows = list(csv.DictReader((run_dir / "mfv_profiles.csv").open()))
    foundation_order = []
    for r in rows:
        if r["foundation"] not in foundation_order:
            foundation_order.append(r["foundation"])
    countries, human = read_human_mfv()
    hfounds = set(next(iter(human.values())))
    founds = [f for f in foundation_order if f.lower() in hfounds]
    dropped = [f for f in foundation_order if f.lower() not in hfounds]
    assert dropped == ["Social Norms"], f"unexpected MFV foundations without a human norm: {dropped}"
    by_c: dict[float, dict[str, float]] = {}
    pmass: dict[float, float] = {}
    for r in rows:
        c = float(r["c"])
        by_c.setdefault(c, {})[r["foundation"]] = float(r["mean"])
        pmass[c] = float(r["pmass"])
    prof = {c: _zscore(np.array([vals[f] for f in founds])) for c, vals in by_c.items()}
    return founds, prof, pmass


def _mfv_zspace(run_dir: Path):
    """Shared MFV adapter -> z-scored relative-emphasis profiles + human MFV culture matrix."""
    founds, prof, pmass = read_mfv_profiles(run_dir)
    countries, human = read_human_mfv()
    fl = [f.lower() for f in founds]
    M = np.array([_zscore(np.array([human[c][f] for f in fl])) for c in countries])
    return founds, countries, M, prof, pmass


# MFV has no ordinal Instrument (it goes through evaluate_multibool, not administer), but the shared
# plotters only read .name/.display off it -- a shim supplies those. The y-values are z-scores, not a
# 1-M scale, so it carries no scale_max and the range passes its own ylabel.
_MFV_INSTR = SimpleNamespace(name="mfv", display="MFV vignettes")
_MFV_YLABEL = "relative emphasis  (z across foundations)"


def plot_mfv_map(run_dir: Path, out: Path, vec_label: str, C: float, coh_cs: list[float]) -> Path:
    """MFV ipsative culture map via the SAME plot_ipsative_pca the ordinal instruments use, in the
    z-scored relative-emphasis space (logit-violation and 1-5 wrongness cannot share a raw axis).
    Red/blue endpoint points show where the steer moves the AI among human cultures."""
    founds, countries, M, prof, _pmass = _mfv_zspace(run_dir)
    pos_c = max(c for c in coh_cs if c > 0.0)
    neg_c = min(c for c in coh_cs if c < 0.0)
    labels = ("base (c=0)", f"c={pos_c:+g}", f"c={neg_c:+g}")
    traj = {c: prof[c] for c in coh_cs}
    zones, emph = zones_for(countries)                 # MFV: 5 country dots, no cloud
    fig = T.maps.plot_ipsative_pca(_MFV_INSTR, founds, countries, M, prof[0.0], prof[pos_c], prof[neg_c],
                                   traj=traj, emphasize=emph, zones=zones, labels=labels)
    fig.axes[0].set_title(f"MFV vignettes: humans vs LLMs steered for {vec_label}", fontsize=10)
    path = T.maps.save_both(fig, out / "mfv", "map_pca_ipsative")
    plt.close(fig)
    return path


def plot_mfv_range(run_dir: Path, out: Path, vec_label: str, C: float, coh_cs: list[float]) -> Path:
    """MFV range via the SAME plot_range the ordinal instruments use, in z relative-emphasis space."""
    founds, countries, M, prof, _pmass = _mfv_zspace(run_dir)
    humans = {f: sorted(((countries[ci], float(M[ci, fi])) for ci in range(len(countries))), key=lambda t: t[1])
              for fi, f in enumerate(founds)}
    fig = T.maps.plot_range(_MFV_INSTR, founds, coh_cs, {c: prof[c] for c in coh_cs},
                            humans, None, vec_label, ylabel=_MFV_YLABEL)
    path = T.maps.save_both(fig, out / "mfv", "range")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/img/showcase"))
    ap.add_argument("--vec-label", required=True,
                    help="short run-local steering label for plot titles; declare the anchor explicitly")
    ap.add_argument("--coherence-frac", type=float, default=0.99,
                    help="keep c rows whose pmass is above this fraction of base")
    ap.add_argument("--contrast-frac", type=float, default=0.50,
                    help="for ordinal surveys, also keep only rows whose mean |C| stays above this fraction of base")
    ap.add_argument("--margin-frac", type=float, default=0.50,
                    help="for MFV, also keep only rows whose mean forced-choice margin stays above this fraction of base")
    args = ap.parse_args()
    summary = json.loads((args.run_dir / "summary.json").read_text())
    C = float(summary["calibrated_C"])
    method = summary["method"]
    vec_label = args.vec_label
    args.out.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    ordinal_names = [name for name in ORDINAL if (args.run_dir / f"{name}_profiles.csv").exists()]
    quality = shared_quality_score(args.run_dir, ordinal_names, pmass_frac=args.coherence_frac,
                                   contrast_frac=args.contrast_frac, margin_frac=args.margin_frac)
    coh_cs = coherent_prefix_cs(sorted(quality), quality, 1.0)
    print(f"shared coherent c values at pmass>={args.coherence_frac:.2%}, "
          f"survey |C|>={args.contrast_frac:.0%}, MFV margin>={args.margin_frac:.0%}: {coh_cs}")
    for name in ordinal_names:
        written += [str(p) for p in plot_ordinal(args.run_dir, args.out, name, vec_label, C, coh_cs)]
    if (args.run_dir / "mfv_profiles.csv").exists():
        written.append(str(plot_mfv_map(args.run_dir, args.out, vec_label, C, coh_cs)))    # shared ipsative map (z-space)
        written.append(str(plot_mfv_range(args.run_dir, args.out, vec_label, C, coh_cs)))  # shared range (z-space)
    print(f"wrote {len(written)} figures under {args.out}:")
    for w in written:
        print(" ", w)


if __name__ == "__main__":
    main()
